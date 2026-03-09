#!/usr/bin/env python3
"""
Cross-Validation Training for 3D CNN Alzheimer's Classification

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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import CNN3DClassifier
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


class CNN3DCVTrainer:
    """Cross-validation trainer for CNN 3D"""

    def __init__(self, config: Dict, fold: int, seed: int, output_dir: Path, use_wandb: bool = False):
        self.config = config
        self.fold = fold
        self.seed = seed
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.device = self._setup_device()

        set_seed(seed)

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def _setup_device(self) -> torch.device:
        device_str = self.config['hardware']['device']
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_str == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def build_model(self) -> nn.Module:
        cfg = self.config['model']
        model = CNN3DClassifier(
            architecture=cfg['architecture'],
            num_classes=cfg['num_classes'],
            in_channels=cfg['in_channels'],
            dropout=cfg.get('dropout', 0.1),
            classifier_dropout=cfg.get('classifier_dropout', 0.5),
        )
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        cfg = self.config['training']

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0),
            eta_min=cfg['lr_min']
        )

        warmup_epochs = cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler

        return optimizer, scheduler

    def build_criterion(self, train_labels: List[int]) -> nn.Module:
        if self.config['training']['use_weighted_loss']:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

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
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

        # Balanced accuracy
        class_accs = []
        for cls in np.unique(all_labels):
            mask = all_labels == cls
            class_accs.append((all_preds[mask] == all_labels[mask]).mean())

        # Sensitivity / Specificity
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.5

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'balanced_accuracy': 100. * np.mean(class_accs),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probs': all_probs
        }

    def train(self, model, train_loader, val_loader, optimizer, scheduler, criterion):
        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']
        min_epochs = self.config['callbacks']['early_stopping'].get('min_epochs', 0)
        best_model_state = None

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.validate(model, val_loader, criterion)
            scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{num_epochs} - Train Acc: {train_metrics['accuracy']:.1f}%, "
                            f"Val Acc: {val_metrics['accuracy']:.1f}%, Val AUC: {val_metrics['auc']:.3f}")

            if self.use_wandb:
                wandb.log({
                    f'fold_{self.fold}/train_loss': train_metrics['loss'],
                    f'fold_{self.fold}/train_accuracy': train_metrics['accuracy'],
                    f'fold_{self.fold}/val_loss': val_metrics['loss'],
                    f'fold_{self.fold}/val_accuracy': val_metrics['accuracy'],
                    f'fold_{self.fold}/val_balanced_accuracy': val_metrics['balanced_accuracy'],
                    f'fold_{self.fold}/val_sensitivity': val_metrics['sensitivity'],
                    f'fold_{self.fold}/val_specificity': val_metrics['specificity'],
                    f'fold_{self.fold}/val_auc': val_metrics['auc'],
                    f'fold_{self.fold}/epoch': epoch + 1,
                    f'fold_{self.fold}/lr': optimizer.param_groups[0]['lr']
                })

            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.epochs_without_improvement += 1

            if self.config['callbacks']['early_stopping']['enabled']:
                if epoch >= min_epochs and self.epochs_without_improvement >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, self.best_val_acc


def create_fold_datasets(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Dict,
    temp_dir: Path
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """Create dataloaders for a single fold"""

    image_size = config['model']['image_size']
    preproc = config.get('preprocessing', {})
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_csv = temp_dir / 'train_fold.csv'
    val_csv = temp_dir / 'val_fold.csv'
    test_csv = temp_dir / 'test_fold.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    target_shape = (image_size, image_size, image_size)
    use_paper = preproc.get('use_paper_preprocessing', True)
    spacing = preproc.get('target_spacing', 1.75)

    train_dataset = MRIDataset(
        str(train_csv), target_shape=target_shape, augment=True,
        use_paper_preprocessing=use_paper, target_spacing=spacing
    )
    val_dataset = MRIDataset(
        str(val_csv), target_shape=target_shape, augment=False,
        use_paper_preprocessing=use_paper, target_spacing=spacing
    )
    test_dataset = MRIDataset(
        str(test_csv), target_shape=target_shape, augment=False,
        use_paper_preprocessing=use_paper, target_spacing=spacing
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    train_labels = train_df['label'].tolist()
    return train_loader, val_loader, test_loader, train_labels


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path,
                         use_wandb: bool = False):
    """Run K-fold cross-validation with multiple seeds"""

    if use_wandb and WANDB_AVAILABLE:
        wandb_cfg = config.get('wandb', {})
        wandb.init(
            project=wandb_cfg.get('project', 'alzheimer-cnn3d'),
            name=wandb_cfg.get('name', f'cnn3d_cv_{n_folds}fold'),
            config=config
        )
    else:
        use_wandb = False

    # Load and combine all data for CV
    logger.info("Loading data...")
    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    test_df = pd.read_csv(config['data']['test_csv'])

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples: {len(all_df)}")
    logger.info(f"Label distribution:\n{all_df['label'].value_counts()}")

    all_labels = all_df['label'].values
    all_indices = np.arange(len(all_df))

    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

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

            logger.info(f"Train: {len(train_idx_final)}, Val: {len(val_idx_local)}, Test: {len(test_idx)}")

            train_loader, val_loader, test_loader, train_labels = create_fold_datasets(
                all_df, train_idx_final, val_idx_local, test_idx, config, temp_dir
            )

            fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            trainer = CNN3DCVTrainer(config, fold, seed, fold_dir, use_wandb=use_wandb)

            model = trainer.build_model()
            optimizer, scheduler = trainer.build_optimizer(model)
            criterion = trainer.build_criterion(train_labels)

            model, best_val_acc = trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)

            test_metrics = trainer.validate(model, test_loader, criterion)

            fold_result = {
                'seed': seed,
                'fold': fold,
                'val_accuracy': best_val_acc,
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                'test_sensitivity': test_metrics['sensitivity'],
                'test_specificity': test_metrics['specificity'],
                'test_auc': test_metrics['auc'],
            }

            fold_results.append(fold_result)
            all_results.append(fold_result)

            if use_wandb:
                wandb.log({
                    f'test/seed_{seed}_fold_{fold}_accuracy': test_metrics['accuracy'],
                    f'test/seed_{seed}_fold_{fold}_balanced_accuracy': test_metrics['balanced_accuracy'],
                    f'test/seed_{seed}_fold_{fold}_sensitivity': test_metrics['sensitivity'],
                    f'test/seed_{seed}_fold_{fold}_specificity': test_metrics['specificity'],
                    f'test/seed_{seed}_fold_{fold}_auc': test_metrics['auc']
                })

            logger.info(f"  Val: {best_val_acc:.1f}% | "
                        f"Test: Acc={test_metrics['accuracy']:.1f}%, "
                        f"BalAcc={test_metrics['balanced_accuracy']:.1f}%, "
                        f"Sens={test_metrics['sensitivity']:.1f}%, "
                        f"Spec={test_metrics['specificity']:.1f}%, "
                        f"AUC={test_metrics['auc']:.3f}")

        # Seed summary
        test_accs = [r['test_accuracy'] for r in fold_results]
        balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]

        logger.info(f"\nSeed {seed} Summary:")
        logger.info(f"  Acc: {np.mean(test_accs):.1f} +/- {np.std(test_accs):.1f}%")
        logger.info(f"  BalAcc: {np.mean(balanced_accs):.1f} +/- {np.std(balanced_accs):.1f}%")
        logger.info(f"  AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS (CNN 3D)")
    logger.info(f"{'='*60}")

    results_df = pd.DataFrame(all_results)

    acc_mean = results_df['test_accuracy'].mean()
    acc_std = results_df['test_accuracy'].std()
    bal_mean = results_df['test_balanced_accuracy'].mean()
    bal_std = results_df['test_balanced_accuracy'].std()
    sens_mean = results_df['test_sensitivity'].mean()
    sens_std = results_df['test_sensitivity'].std()
    spec_mean = results_df['test_specificity'].mean()
    spec_std = results_df['test_specificity'].std()
    auc_mean = results_df['test_auc'].mean()
    auc_std = results_df['test_auc'].std()

    logger.info(f"  Accuracy:     {acc_mean:.1f} +/- {acc_std:.1f}%")
    logger.info(f"  Balanced Acc: {bal_mean:.1f} +/- {bal_std:.1f}%")
    logger.info(f"  Sensitivity:  {sens_mean:.1f} +/- {sens_std:.1f}%")
    logger.info(f"  Specificity:  {spec_mean:.1f} +/- {spec_std:.1f}%")
    logger.info(f"  AUC:          {auc_mean:.3f} +/- {auc_std:.3f}")

    if use_wandb:
        wandb.log({
            'final/accuracy_mean': acc_mean,
            'final/accuracy_std': acc_std,
            'final/balanced_accuracy_mean': bal_mean,
            'final/balanced_accuracy_std': bal_std,
            'final/sensitivity_mean': sens_mean,
            'final/sensitivity_std': sens_std,
            'final/specificity_mean': spec_mean,
            'final/specificity_std': spec_std,
            'final/auc_mean': auc_mean,
            'final/auc_std': auc_std
        })
        wandb.finish()

    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    summary = {
        'n_folds': n_folds,
        'seeds': seeds,
        'architecture': config['model']['architecture'],
        'accuracy': float(acc_mean),
        'accuracy_std': float(acc_std),
        'balanced_accuracy': float(bal_mean),
        'balanced_accuracy_std': float(bal_std),
        'sensitivity': float(sens_mean),
        'sensitivity_std': float(sens_std),
        'specificity': float(spec_mean),
        'specificity_std': float(spec_std),
        'auc': float(auc_mean),
        'auc_std': float(auc_std),
        'per_fold_results': all_results
    }

    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation for CNN 3D')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_wandb = config.get('wandb', {}).get('enabled', False) and not args.no_wandb

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_cross_validation(config, args.n_folds, args.seeds, output_dir, use_wandb=use_wandb)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
