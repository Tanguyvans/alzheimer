#!/usr/bin/env python3
"""
Ablation Study: Tabular Only (FT-Transformer)
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
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent / "multimodal_fusion"))
from model import FTTransformerEncoder

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


class TabularDataset(Dataset):
    """Dataset for tabular-only classification"""

    def __init__(self, csv_path: str, features: List[str], scaler: StandardScaler = None, fit_scaler: bool = False):
        self.df = pd.read_csv(csv_path)
        self.features = features

        # Extract labels
        if 'label' in self.df.columns:
            self.labels = self.df['label'].values
        elif 'DX' in self.df.columns:
            # Map diagnosis to binary
            dx_map = {'CN': 0, 'AD': 1, 'MCI': 1, 'Dementia': 1}
            self.labels = self.df['DX'].map(dx_map).values
        else:
            raise ValueError("No label column found")

        # Extract features
        self.tabular_data = self.df[features].values.astype(np.float32)

        # Handle missing values
        for i in range(self.tabular_data.shape[1]):
            col = self.tabular_data[:, i]
            mask = np.isnan(col)
            if mask.any():
                median = np.nanmedian(col)
                self.tabular_data[mask, i] = median

        # Normalize
        if fit_scaler:
            self.scaler = StandardScaler()
            self.tabular_data = self.scaler.fit_transform(self.tabular_data)
        elif scaler is not None:
            self.scaler = scaler
            self.tabular_data = self.scaler.transform(self.tabular_data)
        else:
            self.scaler = None

        # Store subgroup info if available
        if 'subgroup' in self.df.columns:
            self.subgroups = self.df['subgroup'].values
        else:
            self.subgroups = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tabular, label


class FTTransformerClassifier(nn.Module):
    """FT-Transformer with classification head"""

    def __init__(self, num_features: int, config: Dict, num_classes: int = 2):
        super().__init__()

        tab_cfg = config.get('tabular', {})

        self.encoder = FTTransformerEncoder(
            input_dim=num_features,
            embed_dim=tab_cfg.get('embed_dim', 64),
            num_heads=tab_cfg.get('num_heads', 4),
            num_layers=tab_cfg.get('num_layers', 3),
            dropout=tab_cfg.get('dropout', 0.1),
            output_dim=tab_cfg.get('output_dim', 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class TabularCVTrainer:
    """Cross-validation trainer for Tabular-only model"""

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

    def build_model(self, num_features: int) -> nn.Module:
        model = FTTransformerClassifier(
            num_features=num_features,
            config=self.config['model'],
            num_classes=self.config['model']['num_classes']
        )
        return model.to(self.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Dict:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for tabular, labels in train_loader:
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(tabular)
            loss = criterion(outputs, labels)

            loss.backward()
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
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module = None) -> Dict:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for tabular, labels in loader:
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            outputs = model(tabular)
            if criterion is not None:
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

        # Calculate sensitivity and specificity
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity, specificity = 0, 0

        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return {
            'loss': total_loss / len(loader) if criterion else 0,
            'accuracy': 100.0 * correct / total,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_dataset: TabularDataset, val_dataset: TabularDataset,
              test_dataset: TabularDataset) -> Dict:
        cfg = self.config['training']

        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=self.config['hardware']['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                                shuffle=False, num_workers=self.config['hardware']['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=self.config['hardware']['num_workers'])

        num_features = len(self.config['data']['tabular_features'])
        model = self.build_model(num_features)

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
                if self.epochs_without_improvement >= cfg.get('patience',
                    self.config['callbacks']['early_stopping']['patience']):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model and evaluate on test set
        if best_model_state:
            model.load_state_dict(best_model_state)

        test_metrics = self.evaluate(model, test_loader, criterion)

        return {
            'model': model,
            'scaler': train_dataset.scaler if hasattr(train_dataset, 'scaler') else None,
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
            name=f"{wandb_config.get('name', 'tabular_only_cv')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    features = config['data']['tabular_features']

    # Load full dataset
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    # Combine train and val for CV
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    logger.info(f"Dataset loaded: {len(full_train_df)} train+val, {len(test_df)} test samples")
    logger.info(f"Label distribution (train+val): {full_train_df['label'].value_counts().to_dict()}")

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

            # Save temporary CSVs for this fold
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
            train_dataset = TabularDataset(str(train_fold_csv), features, fit_scaler=True)
            val_dataset = TabularDataset(str(val_fold_csv), features, scaler=train_dataset.scaler)
            test_dataset = TabularDataset(str(test_fold_csv), features, scaler=train_dataset.scaler)

            # Train
            trainer = TabularCVTrainer(config, fold, seed, fold_dir, use_wandb=use_wandb)
            results = trainer.train(train_dataset, val_dataset, test_dataset)

            # Log trajectory results
            logger.info(f"  Val: {results['val_accuracy']:.1f}% | "
                       f"Traj: Acc={results['test_accuracy']:.1f}%, BalAcc={results['test_balanced_accuracy']:.1f}%, "
                       f"Sens={results['test_sensitivity']:.1f}%, Spec={results['test_specificity']:.1f}%, "
                       f"AUC={results['test_auc']:.3f}")

            # Cross-task evaluation on CN_AD test set
            if cn_ad_test_df is not None:
                cn_ad_test_fold_csv = fold_dir / "cn_ad_test.csv"
                cn_ad_test_df.to_csv(cn_ad_test_fold_csv, index=False)
                cn_ad_test_dataset = TabularDataset(str(cn_ad_test_fold_csv), features,
                                                    scaler=train_dataset.scaler)
                cn_ad_loader = DataLoader(cn_ad_test_dataset, batch_size=32, shuffle=False)
                cn_ad_metrics = trainer.evaluate(results['model'], cn_ad_loader)

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
            if 'scaler' in results:
                del results['scaler']

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
    logger.info("FINAL CROSS-VALIDATION RESULTS (Tabular Only)")
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
    parser = argparse.ArgumentParser(description="Tabular-only ablation study")
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
