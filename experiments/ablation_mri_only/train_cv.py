#!/usr/bin/env python3
"""
Ablation Study: MRI Only (ViT)
Cross-Validation Training using the same dataset as multimodal_fusion

Usage:
    python train_cv.py --config config.yaml --n-folds 5 --seeds 42 123 456 789 2024
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import nibabel as nib
from scipy import ndimage

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent / "mri_vit_ad"))
from model import ViT3DClassifier

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


class MRIDataset(Dataset):
    """Dataset for MRI-only classification"""

    def __init__(self, csv_path: str, transform=None, target_size: int = 128):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.target_size = target_size

        # Extract labels
        if 'label' in self.df.columns:
            self.labels = self.df['label'].values
        elif 'DX' in self.df.columns:
            dx_map = {'CN': 0, 'AD': 1, 'MCI': 1, 'Dementia': 1}
            self.labels = self.df['DX'].map(dx_map).values
        else:
            raise ValueError("No label column found")

        # Get MRI paths
        if 'scan_path' in self.df.columns:
            self.mri_paths = self.df['scan_path'].values
        elif 'mri_path' in self.df.columns:
            self.mri_paths = self.df['mri_path'].values
        elif 'npy_path' in self.df.columns:
            self.mri_paths = self.df['npy_path'].values
        else:
            raise ValueError("No MRI path column found")

        # Store subgroup info if available
        if 'subgroup' in self.df.columns:
            self.subgroups = self.df['subgroup'].values
        else:
            self.subgroups = None

    def __len__(self):
        return len(self.labels)

    def load_volume(self, path: str) -> np.ndarray:
        """Load and preprocess MRI volume"""
        if path.endswith('.npy'):
            volume = np.load(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            nii = nib.load(path)
            volume = nii.get_fdata()
        else:
            raise ValueError(f"Unknown file format: {path}")

        # Resize if needed
        if volume.shape != (self.target_size, self.target_size, self.target_size):
            zoom_factors = [self.target_size / s for s in volume.shape]
            volume = ndimage.zoom(volume, zoom_factors, order=1)

        # Normalize
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)

        return volume.astype(np.float32)

    def __getitem__(self, idx):
        mri_path = self.mri_paths[idx]
        label = self.labels[idx]

        try:
            volume = self.load_volume(mri_path)
        except Exception as e:
            logger.warning(f"Error loading {mri_path}: {e}")
            # Return zeros if loading fails
            volume = np.zeros((self.target_size, self.target_size, self.target_size), dtype=np.float32)

        # Add channel dimension
        volume = volume[np.newaxis, ...]

        if self.transform:
            volume = self.transform(volume)

        return torch.tensor(volume, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class MRICVTrainer:
    """Cross-validation trainer for MRI-only model"""

    def __init__(self, config: Dict, fold: int, seed: int, output_dir: Path, use_wandb: bool = False):
        self.config = config
        self.fold = fold
        self.seed = seed
        self.output_dir = output_dir
        self.device = self._setup_device()
        self.use_wandb = use_wandb and WANDB_AVAILABLE

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
        """Build ViT model"""
        model_cfg = self.config['model']

        model = ViT3DClassifier(
            image_size=model_cfg.get('image_size', 128),
            num_classes=model_cfg.get('num_classes', 2),
            in_channels=model_cfg.get('in_channels', 1),
            dropout=model_cfg.get('dropout', 0.1),
            classifier_dropout=model_cfg.get('classifier_dropout', 0.5),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.1)
        )

        # Load pretrained weights if available
        pretrained_path = model_cfg.get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Load with strict=False to handle classifier head mismatch
            model.load_state_dict(state_dict, strict=False)

        return model.to(self.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Dict:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for mri, labels in train_loader:
            mri = mri.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(mri)
            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.config['training']['gradient_clip'])

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module,
                 use_tta: bool = False) -> Dict:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for mri, labels in loader:
            mri = mri.to(self.device)
            labels = labels.to(self.device)

            if use_tta and self.config['training'].get('use_tta', False):
                # Simple TTA: original + flipped
                outputs = model(mri)
                outputs_flip = model(torch.flip(mri, dims=[-1]))
                outputs = (outputs + outputs_flip) / 2
            else:
                outputs = model(mri)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate balanced accuracy
        classes = np.unique(all_labels)
        recalls = []
        for c in classes:
            mask = all_labels == c
            if mask.sum() > 0:
                recalls.append((all_preds[mask] == c).mean())
        balanced_acc = np.mean(recalls) * 100

        # Sensitivity (recall for positive class = AD = 1)
        pos_mask = all_labels == 1
        sensitivity = (all_preds[pos_mask] == 1).mean() * 100 if pos_mask.sum() > 0 else 0.0

        # Specificity (recall for negative class = CN = 0)
        neg_mask = all_labels == 0
        specificity = (all_preds[neg_mask] == 0).mean() * 100 if neg_mask.sum() > 0 else 0.0

        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        return {
            'loss': total_loss / len(loader),
            'accuracy': 100.0 * correct / total,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_dataset: MRIDataset, val_dataset: MRIDataset,
              test_dataset: MRIDataset) -> Dict:
        cfg = self.config['training']

        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=self.config['hardware']['num_workers'],
                                  pin_memory=self.config['hardware'].get('pin_memory', True))
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                                shuffle=False, num_workers=self.config['hardware']['num_workers'],
                                pin_memory=self.config['hardware'].get('pin_memory', True))
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=self.config['hardware']['num_workers'],
                                 pin_memory=self.config['hardware'].get('pin_memory', True))

        model = self.build_model()

        # Optimizer with layer-wise LR decay
        optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                weight_decay=cfg['weight_decay'])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0),
            eta_min=cfg['lr_min']
        )

        # Weighted loss for class imbalance
        if cfg.get('use_weighted_loss', True):
            class_counts = np.bincount(train_dataset.labels)
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * len(class_counts)
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32).to(self.device),
                label_smoothing=cfg.get('label_smoothing', 0.0)
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))

        best_model_state = None

        for epoch in range(cfg['epochs']):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)

            scheduler.step()

            logger.info(f"Epoch {epoch+1}/{cfg['epochs']} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% - "
                       f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                       f"Val AUC: {val_metrics['auc']:.3f}")

            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    f"fold_{self.fold}/train_loss": train_metrics['loss'],
                    f"fold_{self.fold}/train_acc": train_metrics['accuracy'],
                    f"fold_{self.fold}/val_loss": val_metrics['loss'],
                    f"fold_{self.fold}/val_acc": val_metrics['accuracy'],
                    f"fold_{self.fold}/val_balanced_acc": val_metrics['balanced_accuracy'],
                    f"fold_{self.fold}/val_auc": val_metrics['auc'],
                    f"fold_{self.fold}/val_sensitivity": val_metrics['sensitivity'],
                    f"fold_{self.fold}/val_specificity": val_metrics['specificity'],
                    f"fold_{self.fold}/epoch": epoch + 1,
                    f"fold_{self.fold}/lr": optimizer.param_groups[0]['lr'],
                })

            # Early stopping
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
            else:
                self.epochs_without_improvement += 1
                min_epochs = self.config['callbacks']['early_stopping'].get('min_epochs', 0)
                if epoch >= min_epochs and self.epochs_without_improvement >= self.config['callbacks']['early_stopping']['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model and evaluate on test set
        if best_model_state:
            model.load_state_dict(best_model_state)

        test_metrics = self.evaluate(model, test_loader, criterion, use_tta=True)

        return {
            'model': model,
            'val_accuracy': self.best_val_acc,
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_auc': test_metrics['auc'],
            'test_predictions': test_metrics['predictions'],
            'test_labels': test_metrics['labels']
        }


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path,
                         cn_ad_test_csv: Optional[str] = None):
    """Run K-fold cross-validation with multiple seeds and cross-task evaluation"""

    # Initialize WandB if enabled
    use_wandb = False
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False) and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_config.get('project', 'alzheimer-ablation'),
            name=f"{wandb_config.get('name', 'mri_only_cv')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'model': config['model'],
                'training': config['training'],
                'n_folds': n_folds,
                'seeds': seeds,
                'experiment': config.get('experiment', {}),
            }
        )
        use_wandb = True
        logger.info("WandB initialized successfully")

    data_dir = Path(config['data']['data_dir'])

    # Load full dataset
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    logger.info(f"Dataset loaded: {len(full_train_df)} train+val, {len(test_df)} test samples")

    # Load CN_AD test set for cross-task evaluation if provided
    cn_ad_test_df = None
    if cn_ad_test_csv and Path(cn_ad_test_csv).exists():
        cn_ad_test_df = pd.read_csv(cn_ad_test_csv)
        logger.info(f"Loaded CN_AD test set: {len(cn_ad_test_df)} samples")

    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running with seed {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)

        # Get labels for stratification
        if 'label' in full_train_df.columns:
            labels = full_train_df['label'].values
        else:
            dx_map = {'CN': 0, 'AD': 1, 'MCI': 1, 'Dementia': 1}
            labels = full_train_df['DX'].map(dx_map).values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        seed_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, labels)):
            logger.info(f"\n--- Fold {fold+1}/{n_folds} ---")

            fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_fold_df = full_train_df.iloc[train_idx]
            val_fold_df = full_train_df.iloc[val_idx]

            train_fold_csv = fold_dir / "train.csv"
            val_fold_csv = fold_dir / "val.csv"
            test_fold_csv = fold_dir / "test.csv"

            train_fold_df.to_csv(train_fold_csv, index=False)
            val_fold_df.to_csv(val_fold_csv, index=False)
            test_df.to_csv(test_fold_csv, index=False)

            # Create datasets
            train_dataset = MRIDataset(str(train_fold_csv),
                                       target_size=config['model'].get('image_size', 128))
            val_dataset = MRIDataset(str(val_fold_csv),
                                     target_size=config['model'].get('image_size', 128))
            test_dataset = MRIDataset(str(test_fold_csv),
                                      target_size=config['model'].get('image_size', 128))

            # Train
            trainer = MRICVTrainer(config, fold, seed, fold_dir, use_wandb=use_wandb)
            results = trainer.train(train_dataset, val_dataset, test_dataset)

            logger.info(f"  Val: {results['val_accuracy']:.1f}% | "
                       f"Traj: Acc={results['test_accuracy']:.1f}%, BalAcc={results['test_balanced_accuracy']:.1f}%, "
                       f"Sens={results['test_sensitivity']:.1f}%, Spec={results['test_specificity']:.1f}%, "
                       f"AUC={results['test_auc']:.3f}")

            # Cross-task evaluation on CN_AD test set
            if cn_ad_test_df is not None:
                cn_ad_test_fold_csv = fold_dir / "cn_ad_test.csv"
                cn_ad_test_df.to_csv(cn_ad_test_fold_csv, index=False)
                cn_ad_test_dataset = MRIDataset(str(cn_ad_test_fold_csv),
                                                target_size=config['model'].get('image_size', 128))
                cn_ad_loader = DataLoader(cn_ad_test_dataset, batch_size=config['training']['batch_size'],
                                          shuffle=False, num_workers=config['hardware']['num_workers'])
                criterion = nn.CrossEntropyLoss()
                cn_ad_metrics = trainer.evaluate(results['model'], cn_ad_loader, criterion)

                results['cn_ad_accuracy'] = cn_ad_metrics['accuracy']
                results['cn_ad_balanced_accuracy'] = cn_ad_metrics['balanced_accuracy']
                results['cn_ad_sensitivity'] = cn_ad_metrics['sensitivity']
                results['cn_ad_specificity'] = cn_ad_metrics['specificity']
                results['cn_ad_auc'] = cn_ad_metrics['auc']

                logger.info(f"       CN_AD: Acc={cn_ad_metrics['accuracy']:.1f}%, "
                           f"BalAcc={cn_ad_metrics['balanced_accuracy']:.1f}%, "
                           f"Sens={cn_ad_metrics['sensitivity']:.1f}%, Spec={cn_ad_metrics['specificity']:.1f}%, "
                           f"AUC={cn_ad_metrics['auc']:.3f}")

            # Log fold test metrics to WandB
            if use_wandb:
                wandb.log({
                    f"seed_{seed}/fold_{fold}/test_acc": results['test_accuracy'],
                    f"seed_{seed}/fold_{fold}/test_balanced_acc": results['test_balanced_accuracy'],
                    f"seed_{seed}/fold_{fold}/test_sensitivity": results['test_sensitivity'],
                    f"seed_{seed}/fold_{fold}/test_specificity": results['test_specificity'],
                    f"seed_{seed}/fold_{fold}/test_auc": results['test_auc'],
                })
                if cn_ad_test_df is not None:
                    wandb.log({
                        f"seed_{seed}/fold_{fold}/cn_ad_acc": results['cn_ad_accuracy'],
                        f"seed_{seed}/fold_{fold}/cn_ad_balanced_acc": results['cn_ad_balanced_accuracy'],
                        f"seed_{seed}/fold_{fold}/cn_ad_sensitivity": results['cn_ad_sensitivity'],
                        f"seed_{seed}/fold_{fold}/cn_ad_specificity": results['cn_ad_specificity'],
                        f"seed_{seed}/fold_{fold}/cn_ad_auc": results['cn_ad_auc'],
                    })

            # Remove model from results to avoid serialization issues
            del results['model']

            seed_results.append(results)

        # Summarize seed results
        test_accs = [r['test_accuracy'] for r in seed_results]
        balanced_accs = [r['test_balanced_accuracy'] for r in seed_results]
        sensitivities = [r['test_sensitivity'] for r in seed_results]
        specificities = [r['test_specificity'] for r in seed_results]
        aucs = [r['test_auc'] for r in seed_results]

        logger.info(f"\nSeed {seed} Trajectory Summary:")
        logger.info(f"  Acc: {np.mean(test_accs):.1f}±{np.std(test_accs):.1f}%")
        logger.info(f"  BalAcc: {np.mean(balanced_accs):.1f}±{np.std(balanced_accs):.1f}%")
        logger.info(f"  Sens: {np.mean(sensitivities):.1f}±{np.std(sensitivities):.1f}%")
        logger.info(f"  Spec: {np.mean(specificities):.1f}±{np.std(specificities):.1f}%")
        logger.info(f"  AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f}")

        seed_result = {
            'seed': seed,
            'traj_accuracy_mean': np.mean(test_accs),
            'traj_accuracy_std': np.std(test_accs),
            'traj_balanced_accuracy_mean': np.mean(balanced_accs),
            'traj_balanced_accuracy_std': np.std(balanced_accs),
            'traj_sensitivity_mean': np.mean(sensitivities),
            'traj_sensitivity_std': np.std(sensitivities),
            'traj_specificity_mean': np.mean(specificities),
            'traj_specificity_std': np.std(specificities),
            'traj_auc_mean': np.mean(aucs),
            'traj_auc_std': np.std(aucs),
            'fold_results': seed_results
        }

        if cn_ad_test_df is not None:
            cn_ad_accs = [r['cn_ad_accuracy'] for r in seed_results]
            cn_ad_bal_accs = [r['cn_ad_balanced_accuracy'] for r in seed_results]
            cn_ad_sens = [r['cn_ad_sensitivity'] for r in seed_results]
            cn_ad_spec = [r['cn_ad_specificity'] for r in seed_results]
            cn_ad_aucs = [r['cn_ad_auc'] for r in seed_results]

            logger.info(f"Seed {seed} CN_AD Summary:")
            logger.info(f"  Acc: {np.mean(cn_ad_accs):.1f}±{np.std(cn_ad_accs):.1f}%")
            logger.info(f"  BalAcc: {np.mean(cn_ad_bal_accs):.1f}±{np.std(cn_ad_bal_accs):.1f}%")
            logger.info(f"  Sens: {np.mean(cn_ad_sens):.1f}±{np.std(cn_ad_sens):.1f}%")
            logger.info(f"  Spec: {np.mean(cn_ad_spec):.1f}±{np.std(cn_ad_spec):.1f}%")
            logger.info(f"  AUC: {np.mean(cn_ad_aucs):.3f}±{np.std(cn_ad_aucs):.3f}")

            seed_result.update({
                'cn_ad_accuracy_mean': np.mean(cn_ad_accs),
                'cn_ad_accuracy_std': np.std(cn_ad_accs),
                'cn_ad_balanced_accuracy_mean': np.mean(cn_ad_bal_accs),
                'cn_ad_balanced_accuracy_std': np.std(cn_ad_bal_accs),
                'cn_ad_sensitivity_mean': np.mean(cn_ad_sens),
                'cn_ad_sensitivity_std': np.std(cn_ad_sens),
                'cn_ad_specificity_mean': np.mean(cn_ad_spec),
                'cn_ad_specificity_std': np.std(cn_ad_spec),
                'cn_ad_auc_mean': np.mean(cn_ad_aucs),
                'cn_ad_auc_std': np.std(cn_ad_aucs),
            })

        all_results.append(seed_result)

    # Final summary across all seeds
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS (MRI Only)")
    logger.info(f"{'='*60}")

    # Trajectory results
    all_traj_accs = [r['traj_accuracy_mean'] for r in all_results]
    all_traj_bal_accs = [r['traj_balanced_accuracy_mean'] for r in all_results]
    all_traj_sens = [r['traj_sensitivity_mean'] for r in all_results]
    all_traj_spec = [r['traj_specificity_mean'] for r in all_results]
    all_traj_aucs = [r['traj_auc_mean'] for r in all_results]

    logger.info(f"\nCN_AD_TRAJECTORY (train+test):")
    logger.info(f"  Accuracy:     {np.mean(all_traj_accs):.1f}±{np.std(all_traj_accs):.1f}%")
    logger.info(f"  Balanced Acc: {np.mean(all_traj_bal_accs):.1f}±{np.std(all_traj_bal_accs):.1f}%")
    logger.info(f"  Sensitivity:  {np.mean(all_traj_sens):.1f}±{np.std(all_traj_sens):.1f}%")
    logger.info(f"  Specificity:  {np.mean(all_traj_spec):.1f}±{np.std(all_traj_spec):.1f}%")
    logger.info(f"  AUC:          {np.mean(all_traj_aucs):.3f}±{np.std(all_traj_aucs):.3f}")

    if cn_ad_test_df is not None:
        all_cnad_accs = [r['cn_ad_accuracy_mean'] for r in all_results]
        all_cnad_bal_accs = [r['cn_ad_balanced_accuracy_mean'] for r in all_results]
        all_cnad_sens = [r['cn_ad_sensitivity_mean'] for r in all_results]
        all_cnad_spec = [r['cn_ad_specificity_mean'] for r in all_results]
        all_cnad_aucs = [r['cn_ad_auc_mean'] for r in all_results]

        logger.info(f"\nCN_AD (stable AD, cross-task):")
        logger.info(f"  Accuracy:     {np.mean(all_cnad_accs):.1f}±{np.std(all_cnad_accs):.1f}%")
        logger.info(f"  Balanced Acc: {np.mean(all_cnad_bal_accs):.1f}±{np.std(all_cnad_bal_accs):.1f}%")
        logger.info(f"  Sensitivity:  {np.mean(all_cnad_sens):.1f}±{np.std(all_cnad_sens):.1f}%")
        logger.info(f"  Specificity:  {np.mean(all_cnad_spec):.1f}±{np.std(all_cnad_spec):.1f}%")
        logger.info(f"  AUC:          {np.mean(all_cnad_aucs):.3f}±{np.std(all_cnad_aucs):.3f}")

    # Save results
    results_file = output_dir / "cv_results.json"
    with open(results_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        summary = {
            'traj_accuracy': float(np.mean(all_traj_accs)),
            'traj_accuracy_std': float(np.std(all_traj_accs)),
            'traj_balanced_accuracy': float(np.mean(all_traj_bal_accs)),
            'traj_balanced_accuracy_std': float(np.std(all_traj_bal_accs)),
            'traj_sensitivity': float(np.mean(all_traj_sens)),
            'traj_sensitivity_std': float(np.std(all_traj_sens)),
            'traj_specificity': float(np.mean(all_traj_spec)),
            'traj_specificity_std': float(np.std(all_traj_spec)),
            'traj_auc': float(np.mean(all_traj_aucs)),
            'traj_auc_std': float(np.std(all_traj_aucs)),
        }

        if cn_ad_test_df is not None:
            summary.update({
                'cn_ad_accuracy': float(np.mean(all_cnad_accs)),
                'cn_ad_accuracy_std': float(np.std(all_cnad_accs)),
                'cn_ad_balanced_accuracy': float(np.mean(all_cnad_bal_accs)),
                'cn_ad_balanced_accuracy_std': float(np.std(all_cnad_bal_accs)),
                'cn_ad_sensitivity': float(np.mean(all_cnad_sens)),
                'cn_ad_sensitivity_std': float(np.std(all_cnad_sens)),
                'cn_ad_specificity': float(np.mean(all_cnad_spec)),
                'cn_ad_specificity_std': float(np.std(all_cnad_spec)),
                'cn_ad_auc': float(np.mean(all_cnad_aucs)),
                'cn_ad_auc_std': float(np.std(all_cnad_aucs)),
            })

        summary['per_seed_results'] = [{k: convert(v) if k != 'fold_results' else None
                                        for k, v in r.items()} for r in all_results]

        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Log final summary to WandB
    if use_wandb:
        wandb.summary.update({
            'traj_accuracy_mean': float(np.mean(all_traj_accs)),
            'traj_accuracy_std': float(np.std(all_traj_accs)),
            'traj_balanced_accuracy_mean': float(np.mean(all_traj_bal_accs)),
            'traj_balanced_accuracy_std': float(np.std(all_traj_bal_accs)),
            'traj_sensitivity_mean': float(np.mean(all_traj_sens)),
            'traj_sensitivity_std': float(np.std(all_traj_sens)),
            'traj_specificity_mean': float(np.mean(all_traj_spec)),
            'traj_specificity_std': float(np.std(all_traj_spec)),
            'traj_auc_mean': float(np.mean(all_traj_aucs)),
            'traj_auc_std': float(np.std(all_traj_aucs)),
        })
        if cn_ad_test_df is not None:
            wandb.summary.update({
                'cn_ad_accuracy_mean': float(np.mean(all_cnad_accs)),
                'cn_ad_accuracy_std': float(np.std(all_cnad_accs)),
                'cn_ad_balanced_accuracy_mean': float(np.mean(all_cnad_bal_accs)),
                'cn_ad_balanced_accuracy_std': float(np.std(all_cnad_bal_accs)),
                'cn_ad_sensitivity_mean': float(np.mean(all_cnad_sens)),
                'cn_ad_sensitivity_std': float(np.std(all_cnad_sens)),
                'cn_ad_specificity_mean': float(np.mean(all_cnad_spec)),
                'cn_ad_specificity_std': float(np.std(all_cnad_spec)),
                'cn_ad_auc_mean': float(np.mean(all_cnad_aucs)),
                'cn_ad_auc_std': float(np.std(all_cnad_aucs)),
            })
        wandb.finish()
        logger.info("WandB run finished")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MRI-only ablation study")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    parser.add_argument('--cn-ad-test', type=str, default='../multimodal_fusion/data/combined_cn_ad/test.csv',
                        help='Path to CN_AD test CSV for cross-task evaluation')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override WandB setting if --no-wandb is passed
    if args.no_wandb:
        config['wandb'] = {'enabled': False}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cross_validation(config, args.n_folds, args.seeds, output_dir, args.cn_ad_test)


if __name__ == '__main__':
    main()
