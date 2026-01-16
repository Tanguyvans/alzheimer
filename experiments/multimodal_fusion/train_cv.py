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

        return model, self.best_val_acc


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


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path,
                         cn_ad_test_csv: Optional[str] = None, use_wandb: bool = False):
    """Run K-fold cross-validation with multiple seeds and cross-task evaluation"""

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

    # Load CN_AD data to get stable subject_ids for filtering test fold
    stable_subject_ids = None
    if cn_ad_test_csv:
        cn_ad_dir = Path(cn_ad_test_csv)
        if cn_ad_dir.is_dir():
            # Load all cn_ad CSVs to get stable subject_ids
            cn_ad_parts = []
            for csv_name in ['train.csv', 'val.csv', 'test.csv']:
                csv_path = cn_ad_dir / csv_name
                if csv_path.exists():
                    cn_ad_parts.append(pd.read_csv(csv_path))
            if cn_ad_parts:
                cn_ad_full = pd.concat(cn_ad_parts, ignore_index=True)
                stable_subject_ids = set(cn_ad_full['subject_id'].values)
                logger.info(f"Loaded {len(stable_subject_ids)} stable subject_ids for CN_AD filtering")

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
            model, best_val_acc = trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)

            # Test on trajectory test set
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

            # Cross-task evaluation: filter test fold to stable subjects only
            cn_ad_metrics = None
            if stable_subject_ids is not None:
                # Filter test fold to only stable subjects (removes MCI converters)
                cn_ad_test_fold_df = test_df_fold[test_df_fold['subject_id'].isin(stable_subject_ids)]

                if len(cn_ad_test_fold_df) > 0:
                    # Get scaler from train dataset
                    scaler = train_loader.dataset.get_scaler() if hasattr(train_loader.dataset, 'get_scaler') else None

                    # Create CN_AD test dataset from filtered test fold
                    cn_ad_csv = temp_dir / 'cn_ad_test.csv'
                    cn_ad_test_fold_df.to_csv(cn_ad_csv, index=False)

                    tabular_features = config['data']['tabular_features']
                    if 'backbone' in config['model']:
                        image_size = config['model']['backbone'].get('image_size', 128)
                    else:
                        image_size = config['model']['vit']['image_size']
                    preproc = config.get('preprocessing', {})

                    cn_ad_dataset = MultiModalDataset(
                        str(cn_ad_csv),
                        tabular_features=tabular_features,
                        target_shape=(image_size, image_size, image_size),
                        augment=False,
                        normalize_tabular=True,
                        scaler=scaler,
                        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
                        target_spacing=preproc.get('target_spacing', 1.75)
                    )

                    cn_ad_loader = DataLoader(
                        cn_ad_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=False,
                        num_workers=config['hardware']['num_workers'],
                        pin_memory=True
                    )

                    cn_ad_metrics = trainer.validate(model, cn_ad_loader, criterion)
                    logger.info(f"  CN_AD ({len(cn_ad_test_fold_df)} stable): Acc={cn_ad_metrics['accuracy']:.1f}%, "
                               f"BalAcc={cn_ad_metrics['balanced_accuracy']:.1f}%, "
                               f"Sens={cn_ad_metrics['sensitivity']:.1f}%, Spec={cn_ad_metrics['specificity']:.1f}%, "
                               f"AUC={cn_ad_metrics['auc']:.3f}")

            fold_result = {
                'seed': seed,
                'fold': fold,
                'val_accuracy': best_val_acc,
                'traj_accuracy': test_metrics['accuracy'],
                'traj_balanced_accuracy': test_metrics['balanced_accuracy'],
                'traj_sensitivity': test_metrics['sensitivity'],
                'traj_specificity': test_metrics['specificity'],
                'traj_auc': test_metrics['auc'],
                'subgroup_accuracy': subgroup_accs
            }

            # Add CN_AD metrics if available
            if cn_ad_metrics is not None:
                fold_result['cn_ad_accuracy'] = cn_ad_metrics['accuracy']
                fold_result['cn_ad_balanced_accuracy'] = cn_ad_metrics['balanced_accuracy']
                fold_result['cn_ad_sensitivity'] = cn_ad_metrics['sensitivity']
                fold_result['cn_ad_specificity'] = cn_ad_metrics['specificity']
                fold_result['cn_ad_auc'] = cn_ad_metrics['auc']

            fold_results.append(fold_result)
            all_results.append(fold_result)

            # Log fold test metrics to WandB
            if use_wandb:
                wandb.log({
                    f'test/seed_{seed}_fold_{fold}_traj_accuracy': test_metrics['accuracy'],
                    f'test/seed_{seed}_fold_{fold}_traj_balanced_accuracy': test_metrics['balanced_accuracy'],
                    f'test/seed_{seed}_fold_{fold}_traj_sensitivity': test_metrics['sensitivity'],
                    f'test/seed_{seed}_fold_{fold}_traj_specificity': test_metrics['specificity'],
                    f'test/seed_{seed}_fold_{fold}_traj_auc': test_metrics['auc']
                })
                if cn_ad_metrics is not None:
                    wandb.log({
                        f'test/seed_{seed}_fold_{fold}_cn_ad_accuracy': cn_ad_metrics['accuracy'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_balanced_accuracy': cn_ad_metrics['balanced_accuracy'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_sensitivity': cn_ad_metrics['sensitivity'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_specificity': cn_ad_metrics['specificity'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_auc': cn_ad_metrics['auc']
                    })

            logger.info(f"  Val: {best_val_acc:.1f}% | "
                       f"Traj: Acc={test_metrics['accuracy']:.1f}%, BalAcc={test_metrics['balanced_accuracy']:.1f}%, "
                       f"Sens={test_metrics['sensitivity']:.1f}%, Spec={test_metrics['specificity']:.1f}%, "
                       f"AUC={test_metrics['auc']:.3f}")

        # Seed summary
        test_accs = [r['traj_accuracy'] for r in fold_results]
        balanced_accs = [r['traj_balanced_accuracy'] for r in fold_results]
        sensitivities = [r['traj_sensitivity'] for r in fold_results]
        specificities = [r['traj_specificity'] for r in fold_results]
        aucs = [r['traj_auc'] for r in fold_results]

        logger.info(f"\nSeed {seed} Trajectory Summary:")
        logger.info(f"  Acc: {np.mean(test_accs):.1f}±{np.std(test_accs):.1f}%")
        logger.info(f"  BalAcc: {np.mean(balanced_accs):.1f}±{np.std(balanced_accs):.1f}%")
        logger.info(f"  Sens: {np.mean(sensitivities):.1f}±{np.std(sensitivities):.1f}%")
        logger.info(f"  Spec: {np.mean(specificities):.1f}±{np.std(specificities):.1f}%")
        logger.info(f"  AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f}")

        if stable_subject_ids is not None:
            cn_ad_accs = [r['cn_ad_accuracy'] for r in fold_results]
            cn_ad_bal_accs = [r['cn_ad_balanced_accuracy'] for r in fold_results]
            cn_ad_sens = [r['cn_ad_sensitivity'] for r in fold_results]
            cn_ad_spec = [r['cn_ad_specificity'] for r in fold_results]
            cn_ad_aucs = [r['cn_ad_auc'] for r in fold_results]

            logger.info(f"Seed {seed} CN_AD Summary:")
            logger.info(f"  Acc: {np.mean(cn_ad_accs):.1f}±{np.std(cn_ad_accs):.1f}%")
            logger.info(f"  BalAcc: {np.mean(cn_ad_bal_accs):.1f}±{np.std(cn_ad_bal_accs):.1f}%")
            logger.info(f"  Sens: {np.mean(cn_ad_sens):.1f}±{np.std(cn_ad_sens):.1f}%")
            logger.info(f"  Spec: {np.mean(cn_ad_spec):.1f}±{np.std(cn_ad_spec):.1f}%")
            logger.info(f"  AUC: {np.mean(cn_ad_aucs):.3f}±{np.std(cn_ad_aucs):.3f}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS (Multi-Modal)")
    logger.info(f"{'='*60}")

    results_df = pd.DataFrame(all_results)

    # Overall trajectory summary
    traj_acc_mean = results_df['traj_accuracy'].mean()
    traj_acc_std = results_df['traj_accuracy'].std()
    traj_bal_mean = results_df['traj_balanced_accuracy'].mean()
    traj_bal_std = results_df['traj_balanced_accuracy'].std()
    traj_sens_mean = results_df['traj_sensitivity'].mean()
    traj_sens_std = results_df['traj_sensitivity'].std()
    traj_spec_mean = results_df['traj_specificity'].mean()
    traj_spec_std = results_df['traj_specificity'].std()
    traj_auc_mean = results_df['traj_auc'].mean()
    traj_auc_std = results_df['traj_auc'].std()

    logger.info(f"\nCN_AD_TRAJECTORY (train+test):")
    logger.info(f"  Accuracy:     {traj_acc_mean:.1f}±{traj_acc_std:.1f}%")
    logger.info(f"  Balanced Acc: {traj_bal_mean:.1f}±{traj_bal_std:.1f}%")
    logger.info(f"  Sensitivity:  {traj_sens_mean:.1f}±{traj_sens_std:.1f}%")
    logger.info(f"  Specificity:  {traj_spec_mean:.1f}±{traj_spec_std:.1f}%")
    logger.info(f"  AUC:          {traj_auc_mean:.3f}±{traj_auc_std:.3f}")

    # CN_AD summary if available
    cn_ad_acc_mean, cn_ad_acc_std = None, None
    cn_ad_bal_mean, cn_ad_bal_std = None, None
    cn_ad_sens_mean, cn_ad_sens_std = None, None
    cn_ad_spec_mean, cn_ad_spec_std = None, None
    cn_ad_auc_mean, cn_ad_auc_std = None, None

    if 'cn_ad_accuracy' in results_df.columns:
        cn_ad_acc_mean = results_df['cn_ad_accuracy'].mean()
        cn_ad_acc_std = results_df['cn_ad_accuracy'].std()
        cn_ad_bal_mean = results_df['cn_ad_balanced_accuracy'].mean()
        cn_ad_bal_std = results_df['cn_ad_balanced_accuracy'].std()
        cn_ad_sens_mean = results_df['cn_ad_sensitivity'].mean()
        cn_ad_sens_std = results_df['cn_ad_sensitivity'].std()
        cn_ad_spec_mean = results_df['cn_ad_specificity'].mean()
        cn_ad_spec_std = results_df['cn_ad_specificity'].std()
        cn_ad_auc_mean = results_df['cn_ad_auc'].mean()
        cn_ad_auc_std = results_df['cn_ad_auc'].std()

        logger.info(f"\nCN_AD (stable AD, cross-task):")
        logger.info(f"  Accuracy:     {cn_ad_acc_mean:.1f}±{cn_ad_acc_std:.1f}%")
        logger.info(f"  Balanced Acc: {cn_ad_bal_mean:.1f}±{cn_ad_bal_std:.1f}%")
        logger.info(f"  Sensitivity:  {cn_ad_sens_mean:.1f}±{cn_ad_sens_std:.1f}%")
        logger.info(f"  Specificity:  {cn_ad_spec_mean:.1f}±{cn_ad_spec_std:.1f}%")
        logger.info(f"  AUC:          {cn_ad_auc_mean:.3f}±{cn_ad_auc_std:.3f}")

    # Log final summary to WandB
    if use_wandb:
        wandb.log({
            'final/traj_accuracy_mean': traj_acc_mean,
            'final/traj_accuracy_std': traj_acc_std,
            'final/traj_balanced_accuracy_mean': traj_bal_mean,
            'final/traj_balanced_accuracy_std': traj_bal_std,
            'final/traj_sensitivity_mean': traj_sens_mean,
            'final/traj_sensitivity_std': traj_sens_std,
            'final/traj_specificity_mean': traj_spec_mean,
            'final/traj_specificity_std': traj_spec_std,
            'final/traj_auc_mean': traj_auc_mean,
            'final/traj_auc_std': traj_auc_std
        })
        if cn_ad_acc_mean is not None:
            wandb.log({
                'final/cn_ad_accuracy_mean': cn_ad_acc_mean,
                'final/cn_ad_accuracy_std': cn_ad_acc_std,
                'final/cn_ad_balanced_accuracy_mean': cn_ad_bal_mean,
                'final/cn_ad_balanced_accuracy_std': cn_ad_bal_std,
                'final/cn_ad_sensitivity_mean': cn_ad_sens_mean,
                'final/cn_ad_sensitivity_std': cn_ad_sens_std,
                'final/cn_ad_specificity_mean': cn_ad_spec_mean,
                'final/cn_ad_specificity_std': cn_ad_spec_std,
                'final/cn_ad_auc_mean': cn_ad_auc_mean,
                'final/cn_ad_auc_std': cn_ad_auc_std
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
        'traj_accuracy': float(traj_acc_mean),
        'traj_accuracy_std': float(traj_acc_std),
        'traj_balanced_accuracy': float(traj_bal_mean),
        'traj_balanced_accuracy_std': float(traj_bal_std),
        'traj_sensitivity': float(traj_sens_mean),
        'traj_sensitivity_std': float(traj_sens_std),
        'traj_specificity': float(traj_spec_mean),
        'traj_specificity_std': float(traj_spec_std),
        'traj_auc': float(traj_auc_mean),
        'traj_auc_std': float(traj_auc_std),
        'subgroup_summary': subgroup_summary,
        'per_fold_results': all_results
    }

    # Add CN_AD metrics if available
    if cn_ad_acc_mean is not None:
        summary.update({
            'cn_ad_accuracy': float(cn_ad_acc_mean),
            'cn_ad_accuracy_std': float(cn_ad_acc_std),
            'cn_ad_balanced_accuracy': float(cn_ad_bal_mean),
            'cn_ad_balanced_accuracy_std': float(cn_ad_bal_std),
            'cn_ad_sensitivity': float(cn_ad_sens_mean),
            'cn_ad_sensitivity_std': float(cn_ad_sens_std),
            'cn_ad_specificity': float(cn_ad_spec_mean),
            'cn_ad_specificity_std': float(cn_ad_spec_std),
            'cn_ad_auc': float(cn_ad_auc_mean),
            'cn_ad_auc_std': float(cn_ad_auc_std),
        })

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
    parser.add_argument('--cn-ad-test', type=str, default='data/combined_cn_ad',
                        help='Path to CN_AD data directory for cross-task evaluation (filters test fold to stable subjects)')
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
    summary = run_cross_validation(config, args.n_folds, args.seeds, output_dir,
                                   cn_ad_test_csv=args.cn_ad_test, use_wandb=use_wandb)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
