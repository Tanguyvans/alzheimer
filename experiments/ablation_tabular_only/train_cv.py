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

    def __init__(self, config: Dict, fold: int, seed: int, output_dir: Path):
        self.config = config
        self.fold = fold
        self.seed = seed
        self.output_dir = output_dir
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

        for tabular, labels in tqdm(train_loader, desc=f"Training", leave=False):
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
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for tabular, labels in loader:
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            outputs = model(tabular)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate balanced accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        classes = np.unique(all_labels)
        recalls = []
        for c in classes:
            mask = all_labels == c
            if mask.sum() > 0:
                recalls.append((all_preds[mask] == c).mean())
        balanced_acc = np.mean(recalls) * 100

        return {
            'loss': total_loss / len(loader),
            'accuracy': 100.0 * correct / total,
            'balanced_accuracy': balanced_acc,
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
                       f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")

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
            'val_accuracy': self.best_val_acc,
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_predictions': test_metrics['predictions'],
            'test_labels': test_metrics['labels']
        }


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path):
    """Run K-fold cross-validation with multiple seeds"""

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
            trainer = TabularCVTrainer(config, fold, seed, fold_dir)
            results = trainer.train(train_dataset, val_dataset, test_dataset)

            logger.info(f"Fold {fold+1} - Val: {results['val_accuracy']:.2f}%, "
                       f"Test: {results['test_accuracy']:.2f}%, "
                       f"Balanced: {results['test_balanced_accuracy']:.2f}%")

            seed_results.append(results)

        # Summarize seed results
        test_accs = [r['test_accuracy'] for r in seed_results]
        balanced_accs = [r['test_balanced_accuracy'] for r in seed_results]

        logger.info(f"\nSeed {seed} Summary: {np.mean(test_accs):.2f}% +/- {np.std(test_accs):.2f}%")

        all_results.append({
            'seed': seed,
            'test_accuracy_mean': np.mean(test_accs),
            'test_accuracy_std': np.std(test_accs),
            'balanced_accuracy_mean': np.mean(balanced_accs),
            'balanced_accuracy_std': np.std(balanced_accs),
            'fold_results': seed_results
        })

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")

    for result in all_results:
        logger.info(f"Seed {result['seed']}: {result['test_accuracy_mean']:.2f}% +/- {result['test_accuracy_std']:.2f}%")

    all_test_accs = [r['test_accuracy_mean'] for r in all_results]
    all_balanced_accs = [r['balanced_accuracy_mean'] for r in all_results]

    logger.info(f"\n{'='*40}")
    logger.info(f"OVERALL: {np.mean(all_test_accs):.2f}% +/- {np.std(all_test_accs):.2f}%")
    logger.info(f"BALANCED: {np.mean(all_balanced_accs):.2f}% +/- {np.std(all_balanced_accs):.2f}%")
    logger.info(f"{'='*40}")

    # Save results
    results_file = output_dir / "cv_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        json.dump({
            'overall_accuracy': float(np.mean(all_test_accs)),
            'overall_accuracy_std': float(np.std(all_test_accs)),
            'overall_balanced_accuracy': float(np.mean(all_balanced_accs)),
            'overall_balanced_accuracy_std': float(np.std(all_balanced_accs)),
            'per_seed_results': [{k: convert(v) if k != 'fold_results' else None
                                  for k, v in r.items()} for r in all_results]
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Tabular-only ablation study")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024],
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cross_validation(config, args.n_folds, args.seeds, output_dir)


if __name__ == '__main__':
    main()
