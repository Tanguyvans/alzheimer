#!/usr/bin/env python3
"""
Cross-Validation Training for Multi-Modal Fusion (MRI + Tabular)

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
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import build_model
from dataset import MultiModalDataset

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


class MultiModalCVTrainer:
    """Cross-validation trainer for Multi-Modal Fusion"""

    def __init__(self, config: Dict, fold: int, seed: int, output_dir: Path, use_wandb: bool = False):
        self.config = config
        self.fold = fold
        self.seed = seed
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
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
        """Build multi-modal fusion model"""
        num_tabular_features = len(self.config['data']['tabular_features'])
        model = build_model(self.config, num_tabular_features)
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        """Build optimizer and scheduler"""
        cfg = self.config['training']

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
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
        """Build loss function with class weights"""
        if self.config['training']['use_weighted_loss']:
            class_counts = np.bincount(train_labels)
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

        for mri, tabular, labels in train_loader:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(mri, tabular)
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
        all_preds, all_labels, all_probs = [], [], []

        for mri, tabular, labels in dataloader:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            outputs = model(mri, tabular)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        # Compute metrics
        all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

        # Balanced accuracy
        class_accs = []
        for cls in np.unique(all_labels):
            mask = all_labels == cls
            class_accs.append((all_preds[mask] == all_labels[mask]).mean())

        # Sensitivity (recall for positive class) and Specificity (recall for negative class)
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
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
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']
        min_epochs = self.config['callbacks']['early_stopping'].get('min_epochs', 0)
        best_model_state = None

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.validate(model, val_loader, criterion)
            scheduler.step()

            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{num_epochs} - Train Acc: {train_metrics['accuracy']:.1f}%, "
                           f"Val Acc: {val_metrics['accuracy']:.1f}%, Val AUC: {val_metrics['auc']:.3f}")

            # WandB logging
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

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return self.best_val_acc


class MultiModalSubset(Subset):
    """Subset that preserves access to underlying dataset attributes"""

    def __init__(self, dataset, indices, scaler=None):
        super().__init__(dataset, indices)
        self._scaler = scaler

    def set_scaler(self, scaler):
        """Update the scaler for this subset's dataset"""
        self._scaler = scaler
        if hasattr(self.dataset, 'scaler'):
            self.dataset.scaler = scaler


def create_fold_datasets(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Dict,
    temp_dir: Path
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for a single fold with proper scaler handling"""

    tabular_features = config['data']['tabular_features']
    # Support both backbone config and legacy vit config
    if 'backbone' in config['model']:
        image_size = config['model']['backbone'].get('image_size', 128)
    else:
        image_size = config['model']['vit']['image_size']
    preproc = config.get('preprocessing', {})
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']

    # Save fold CSVs
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_csv = temp_dir / 'train_fold.csv'
    val_csv = temp_dir / 'val_fold.csv'
    test_csv = temp_dir / 'test_fold.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Create train dataset (fits scaler)
    train_dataset = MultiModalDataset(
        str(train_csv),
        tabular_features=tabular_features,
        target_shape=(image_size, image_size, image_size),
        augment=True,
        normalize_tabular=True,
        scaler=None,  # Fit new scaler
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    # Val/test use the same scaler
    val_dataset = MultiModalDataset(
        str(val_csv),
        tabular_features=tabular_features,
        target_shape=(image_size, image_size, image_size),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    test_dataset = MultiModalDataset(
        str(test_csv),
        tabular_features=tabular_features,
        target_shape=(image_size, image_size, image_size),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get train labels for weighted loss
    train_labels = train_df['label'].tolist()

    return train_loader, val_loader, test_loader, train_labels


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path, use_wandb: bool = False):
    """Run K-fold cross-validation with multiple seeds"""

    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'alzheimer-multimodal'),
            name=wandb_config.get('name', f'multimodal_cv_{n_folds}fold'),
            config=config
        )
        logger.info("WandB initialized")
    else:
        use_wandb = False

    # Load all data
    logger.info("Loading data...")
    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    test_df = pd.read_csv(config['data']['test_csv'])

    # Combine all data for CV
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples: {len(all_df)}")
    logger.info(f"Label distribution:\n{all_df['label'].value_counts()}")

    # Load trajectory reference for subgroup analysis (CN / AD / MCI_to_AD)
    trajectory_file = config['data'].get('trajectory_csv', '../../data/adni/adni_cn_ad_trajectory.csv')
    try:
        traj_ref = pd.read_csv(trajectory_file)[['subject_id', 'trajectory']].drop_duplicates()
        all_df = all_df.merge(traj_ref, on='subject_id', how='left')
        # Fill missing trajectory based on DX column
        if 'trajectory' in all_df.columns:
            all_df['trajectory'] = all_df['trajectory'].fillna(all_df.get('DX', 'Unknown'))
        logger.info(f"Trajectory distribution:\n{all_df['trajectory'].value_counts()}")
    except Exception as e:
        logger.warning(f"Could not load trajectory file: {e}. Subgroup analysis disabled.")
        all_df['trajectory'] = all_df.get('DX', 'Unknown')

    # Get labels for stratification
    all_labels = all_df['label'].values
    all_indices = np.arange(len(all_df))

    # Create temp directory for fold CSVs
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

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

            logger.info(f"Train: {len(train_idx_final)}, Val: {len(val_idx_local)}, Test: {len(test_idx)}")

            # Create fold dataloaders
            train_loader, val_loader, test_loader, train_labels = create_fold_datasets(
                all_df, train_idx_final, val_idx_local, test_idx, config, temp_dir
            )

            # Create trainer
            fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            trainer = MultiModalCVTrainer(config, fold, seed, fold_dir, use_wandb=use_wandb)

            # Build components
            model = trainer.build_model()
            optimizer, scheduler = trainer.build_optimizer(model)
            criterion = trainer.build_criterion(train_labels)

            # Train
            best_val_acc = trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)

            # Test
            test_metrics = trainer.validate(model, test_loader, criterion)

            # Subgroup analysis: compute accuracy per trajectory (CN / AD / MCI_to_AD)
            test_df_fold = all_df.iloc[test_idx].reset_index(drop=True)
            test_df_fold['prediction'] = test_metrics['predictions']
            test_df_fold['correct'] = (test_df_fold['prediction'] == test_df_fold['label'])

            subgroup_accs = {}
            for traj in test_df_fold['trajectory'].unique():
                subset = test_df_fold[test_df_fold['trajectory'] == traj]
                if len(subset) > 0:
                    acc = subset['correct'].mean() * 100
                    subgroup_accs[traj] = {'accuracy': acc, 'n_samples': len(subset)}

            # Log subgroup results
            logger.info("  Subgroup Analysis:")
            for traj, metrics in sorted(subgroup_accs.items()):
                logger.info(f"    {traj}: {metrics['accuracy']:.1f}% ({metrics['n_samples']} samples)")

            fold_result = {
                'seed': seed,
                'fold': fold,
                'val_accuracy': best_val_acc,
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                'test_sensitivity': test_metrics['sensitivity'],
                'test_specificity': test_metrics['specificity'],
                'test_auc': test_metrics['auc'],
                'subgroup_accuracy': subgroup_accs
            }
            fold_results.append(fold_result)
            all_results.append(fold_result)

            # Log fold test metrics to WandB
            if use_wandb:
                wandb.log({
                    f'test/seed_{seed}_fold_{fold}_accuracy': test_metrics['accuracy'],
                    f'test/seed_{seed}_fold_{fold}_balanced_accuracy': test_metrics['balanced_accuracy'],
                    f'test/seed_{seed}_fold_{fold}_sensitivity': test_metrics['sensitivity'],
                    f'test/seed_{seed}_fold_{fold}_specificity': test_metrics['specificity'],
                    f'test/seed_{seed}_fold_{fold}_auc': test_metrics['auc']
                })

            logger.info(f"Fold {fold+1} - Val: {best_val_acc:.2f}%, Test: {test_metrics['accuracy']:.2f}%, "
                       f"Balanced: {test_metrics['balanced_accuracy']:.2f}%, AUC: {test_metrics['auc']:.3f}")

        # Seed summary
        seed_accs = [r['test_accuracy'] for r in fold_results]
        logger.info(f"\nSeed {seed} Summary: {np.mean(seed_accs):.2f}% +/- {np.std(seed_accs):.2f}%")

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
        logger.info(f"Seed {seed}: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    # Overall summary
    overall_mean = results_df['test_accuracy'].mean()
    overall_std = results_df['test_accuracy'].std()
    balanced_mean = results_df['test_balanced_accuracy'].mean()
    balanced_std = results_df['test_balanced_accuracy'].std()
    sensitivity_mean = results_df['test_sensitivity'].mean()
    sensitivity_std = results_df['test_sensitivity'].std()
    specificity_mean = results_df['test_specificity'].mean()
    specificity_std = results_df['test_specificity'].std()
    auc_mean = results_df['test_auc'].mean()
    auc_std = results_df['test_auc'].std()

    logger.info(f"\n{'='*40}")
    logger.info(f"OVERALL: {overall_mean:.2f}% +/- {overall_std:.2f}%")
    logger.info(f"BALANCED: {balanced_mean:.2f}% +/- {balanced_std:.2f}%")
    logger.info(f"SENSITIVITY: {sensitivity_mean:.2f}% +/- {sensitivity_std:.2f}%")
    logger.info(f"SPECIFICITY: {specificity_mean:.2f}% +/- {specificity_std:.2f}%")
    logger.info(f"AUC: {auc_mean:.3f} +/- {auc_std:.3f}")
    logger.info(f"{'='*40}")

    # Log final summary to WandB
    if use_wandb:
        wandb.log({
            'final/accuracy_mean': overall_mean,
            'final/accuracy_std': overall_std,
            'final/balanced_accuracy_mean': balanced_mean,
            'final/balanced_accuracy_std': balanced_std,
            'final/sensitivity_mean': sensitivity_mean,
            'final/sensitivity_std': sensitivity_std,
            'final/specificity_mean': specificity_mean,
            'final/specificity_std': specificity_std,
            'final/auc_mean': auc_mean,
            'final/auc_std': auc_std
        })
        wandb.finish()

    # Subgroup summary across all folds
    logger.info("\nSUBGROUP ANALYSIS SUMMARY:")
    logger.info("="*40)
    subgroup_totals = {}
    for result in all_results:
        for traj, metrics in result.get('subgroup_accuracy', {}).items():
            if traj not in subgroup_totals:
                subgroup_totals[traj] = {'accuracies': [], 'n_samples': []}
            subgroup_totals[traj]['accuracies'].append(metrics['accuracy'])
            subgroup_totals[traj]['n_samples'].append(metrics['n_samples'])

    for traj in sorted(subgroup_totals.keys()):
        accs = subgroup_totals[traj]['accuracies']
        n_samples = subgroup_totals[traj]['n_samples']
        logger.info(f"  {traj}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}% (avg {np.mean(n_samples):.0f} samples/fold)")
    logger.info("="*40)

    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    # Compute subgroup summary stats
    subgroup_summary = {}
    for traj in sorted(subgroup_totals.keys()):
        accs = subgroup_totals[traj]['accuracies']
        subgroup_summary[traj] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'avg_samples_per_fold': float(np.mean(subgroup_totals[traj]['n_samples']))
        }

    summary = {
        'n_folds': n_folds,
        'seeds': seeds,
        'overall_accuracy_mean': float(overall_mean),
        'overall_accuracy_std': float(overall_std),
        'balanced_accuracy_mean': float(balanced_mean),
        'balanced_accuracy_std': float(balanced_std),
        'sensitivity_mean': float(sensitivity_mean),
        'sensitivity_std': float(sensitivity_std),
        'specificity_mean': float(specificity_mean),
        'specificity_std': float(specificity_std),
        'auc_mean': float(auc_mean),
        'auc_std': float(auc_std),
        'subgroup_summary': subgroup_summary,
        'per_fold_results': all_results
    }

    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup temp directory
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation Training for Multi-Modal Fusion')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine WandB usage
    use_wandb = config.get('wandb', {}).get('enabled', False) and not args.no_wandb

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run cross-validation
    summary = run_cross_validation(config, args.n_folds, args.seeds, output_dir, use_wandb=use_wandb)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
