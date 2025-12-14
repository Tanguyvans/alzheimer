#!/usr/bin/env python3
"""
Cross-Validation Training for 3D Vision Transformer

Runs K-fold cross-validation with multiple seeds for robust evaluation.

Usage:
    python train_cv.py --config config.yaml --n-folds 5 --seeds 42 123 456
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from model import ViT3DClassifier
from dataset import MRIDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_layer_wise_lr_decay_params(model, base_lr: float, weight_decay: float, lr_decay: float = 0.75):
    """Create parameter groups with layer-wise learning rate decay."""
    param_groups = []
    num_blocks = len(model.blocks)

    # Head (highest LR)
    head_params = list(model.head.parameters())
    if head_params:
        param_groups.append({'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay})

    # Norm
    norm_params = list(model.norm.parameters())
    if norm_params:
        param_groups.append({'params': norm_params, 'lr': base_lr * lr_decay, 'weight_decay': weight_decay})

    # Transformer blocks (reverse order)
    for i in range(num_blocks - 1, -1, -1):
        block_params = list(model.blocks[i].parameters())
        layer_idx = num_blocks - i + 1
        lr = base_lr * (lr_decay ** layer_idx)
        if block_params:
            param_groups.append({'params': block_params, 'lr': lr, 'weight_decay': weight_decay})

    # Embeddings (lowest LR)
    embed_layer_idx = num_blocks + 2
    embed_lr = base_lr * (lr_decay ** embed_layer_idx)
    embed_params = [model.cls_token, model.pos_embed] + list(model.patch_embed.parameters())
    embed_params = [p for p in embed_params if isinstance(p, torch.nn.Parameter)]
    if embed_params:
        param_groups.append({'params': embed_params, 'lr': embed_lr, 'weight_decay': weight_decay})

    return param_groups


class CVTrainer:
    """Cross-validation trainer for 3D ViT"""

    def __init__(self, config: Dict, fold: int, seed: int, output_dir: Path):
        self.config = config
        self.fold = fold
        self.seed = seed
        self.output_dir = output_dir
        self.device = self._setup_device()

        set_seed(seed)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_str = self.config['hardware']['device']
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_str == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def build_model(self) -> nn.Module:
        """Build the model"""
        cfg = self.config['model']
        pretrained_path = None
        if cfg['use_pretrained']:
            p = Path(cfg['pretrained_path'])
            if p.exists():
                pretrained_path = str(p)

        model = ViT3DClassifier(
            architecture=cfg['architecture'],
            num_classes=cfg['num_classes'],
            in_channels=cfg['in_channels'],
            image_size=cfg['image_size'],
            pretrained_path=pretrained_path,
            dropout=cfg['dropout'],
            classifier_dropout=cfg['classifier_dropout'],
            drop_path_rate=cfg.get('drop_path_rate', 0.1),
        )
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        """Build optimizer and scheduler"""
        cfg = self.config['training']
        use_layer_lr_decay = cfg.get('layer_wise_lr_decay', 0.0)

        if use_layer_lr_decay > 0:
            param_groups = get_layer_wise_lr_decay_params(
                model, cfg['learning_rate'], cfg['weight_decay'], use_layer_lr_decay
            )
            optimizer = optim.AdamW(param_groups)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0), eta_min=cfg['lr_min']
        )

        warmup_epochs = cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        else:
            scheduler = main_scheduler

        return optimizer, scheduler

    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build loss function with class weights"""
        if self.config['training']['use_weighted_loss']:
            labels = [label for _, label in train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {'loss': total_loss / len(train_loader), 'accuracy': 100. * correct / total}

    @torch.no_grad()
    def validate(self, model, dataloader, criterion):
        """Validate model"""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Balanced accuracy
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        class_accs = []
        for cls in np.unique(all_labels):
            mask = all_labels == cls
            class_accs.append((all_preds[mask] == all_labels[mask]).mean())

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'balanced_accuracy': 100. * np.mean(class_accs),
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, model, train_loader, val_loader, optimizer, scheduler, criterion):
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']
        best_model_state = None

        pbar = tqdm(range(num_epochs), desc=f"Fold {self.fold+1} Seed {self.seed}")
        for epoch in pbar:
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.validate(model, val_loader, criterion)
            scheduler.step()

            pbar.set_postfix({
                'train_acc': f"{train_metrics['accuracy']:.1f}%",
                'val_acc': f"{val_metrics['accuracy']:.1f}%"
            })

            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
            else:
                self.epochs_without_improvement += 1

            if self.config['callbacks']['early_stopping']['enabled']:
                if self.epochs_without_improvement >= patience:
                    break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return self.best_val_acc


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path):
    """Run K-fold cross-validation with multiple seeds"""

    # Load all data
    logger.info("Loading data...")
    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    test_df = pd.read_csv(config['data']['test_csv'])

    # Combine all data for CV
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples: {len(all_df)}")

    # Get labels for stratification
    all_labels = all_df['label'].values
    all_indices = np.arange(len(all_df))

    # Save combined CSV temporarily
    combined_csv = output_dir / 'combined_data.csv'
    all_df.to_csv(combined_csv, index=False)

    # Create full dataset
    image_size = config['model']['image_size']
    preproc = config.get('preprocessing', {})

    full_dataset = MRIDataset(
        str(combined_csv),
        target_shape=(image_size, image_size, image_size),
        augment=False,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    # Results storage
    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED: {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(all_indices, all_labels)):
            logger.info(f"\n--- Fold {fold+1}/{n_folds} ---")

            # Split train into train/val (90/10)
            set_seed(seed + fold)
            n_train = len(train_idx)
            val_size = int(0.1 * n_train)
            perm = np.random.permutation(n_train)
            val_idx_local = train_idx[perm[:val_size]]
            train_idx_final = train_idx[perm[val_size:]]

            # Create subsets
            train_subset = Subset(full_dataset, train_idx_final)
            val_subset = Subset(full_dataset, val_idx_local)
            test_subset = Subset(full_dataset, test_idx)

            # Enable augmentation for training
            train_dataset_aug = MRIDataset(
                str(combined_csv),
                target_shape=(image_size, image_size, image_size),
                augment=config['augmentation']['enabled'],
                use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
                target_spacing=preproc.get('target_spacing', 1.75)
            )
            train_subset_aug = Subset(train_dataset_aug, train_idx_final)

            # Create dataloaders
            batch_size = config['training']['batch_size']
            train_loader = DataLoader(train_subset_aug, batch_size=batch_size, shuffle=True,
                                     num_workers=config['hardware']['num_workers'], pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                   num_workers=config['hardware']['num_workers'], pin_memory=True)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=config['hardware']['num_workers'], pin_memory=True)

            logger.info(f"Train: {len(train_idx_final)}, Val: {len(val_idx_local)}, Test: {len(test_idx)}")

            # Create trainer
            fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            trainer = CVTrainer(config, fold, seed, fold_dir)

            # Build components
            model = trainer.build_model()
            optimizer, scheduler = trainer.build_optimizer(model)
            criterion = trainer.build_criterion(train_loader)

            # Train
            best_val_acc = trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)

            # Test
            test_metrics = trainer.validate(model, test_loader, criterion)

            fold_result = {
                'seed': seed,
                'fold': fold,
                'val_accuracy': best_val_acc,
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy']
            }
            fold_results.append(fold_result)
            all_results.append(fold_result)

            logger.info(f"Fold {fold+1} - Val: {best_val_acc:.2f}%, Test: {test_metrics['accuracy']:.2f}%")

        # Seed summary
        seed_accs = [r['test_accuracy'] for r in fold_results]
        logger.info(f"\nSeed {seed} Summary: {np.mean(seed_accs):.2f}% ± {np.std(seed_accs):.2f}%")

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*60)

    results_df = pd.DataFrame(all_results)

    # Per-seed summary
    for seed in seeds:
        seed_results = results_df[results_df['seed'] == seed]
        mean_acc = seed_results['test_accuracy'].mean()
        std_acc = seed_results['test_accuracy'].std()
        logger.info(f"Seed {seed}: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Overall summary
    overall_mean = results_df['test_accuracy'].mean()
    overall_std = results_df['test_accuracy'].std()
    balanced_mean = results_df['test_balanced_accuracy'].mean()
    balanced_std = results_df['test_balanced_accuracy'].std()

    logger.info(f"\n{'='*40}")
    logger.info(f"OVERALL: {overall_mean:.2f}% ± {overall_std:.2f}%")
    logger.info(f"BALANCED: {balanced_mean:.2f}% ± {balanced_std:.2f}%")
    logger.info(f"{'='*40}")

    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    summary = {
        'n_folds': n_folds,
        'seeds': seeds,
        'overall_accuracy_mean': float(overall_mean),
        'overall_accuracy_std': float(overall_std),
        'balanced_accuracy_mean': float(balanced_mean),
        'balanced_accuracy_std': float(balanced_std),
        'per_fold_results': all_results
    }

    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation Training for 3D ViT')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run cross-validation
    summary = run_cross_validation(config, args.n_folds, args.seeds, output_dir)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
