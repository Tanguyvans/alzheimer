#!/usr/bin/env python3
"""
ADNI-Only Training with External Validation on OASIS and NACC

Demonstrates cross-cohort generalization gap:
- Train on ADNI only (5-fold CV within ADNI)
- Test on ADNI (in-domain), OASIS (external), NACC (external)

Usage:
    python train_adni_external.py --config config_adni.yaml --n-folds 5 --seeds 42 123 456
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
from typing import Dict, List, Optional
import logging
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
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


class Trainer:
    """Simple trainer for single fold"""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device

    def build_model(self, num_tabular_features: int) -> nn.Module:
        model = build_model(self.config, num_tabular_features)
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        cfg = self.config['training']
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'], eta_min=cfg['lr_min']
        )
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

        for mri, tabular, labels in train_loader:
            mri, tabular, labels = mri.to(self.device), tabular.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(mri, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {'loss': total_loss / len(train_loader), 'accuracy': 100. * correct / total}

    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        for mri, tabular, labels in dataloader:
            mri, tabular, labels = mri.to(self.device), tabular.to(self.device), labels.to(self.device)

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

        all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

        # Metrics
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        return {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            'accuracy': 100. * correct / total if total > 0 else 0,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'n_samples': total
        }

    def train(self, model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, patience, min_epochs):
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)
            scheduler.step()

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs} - Train: {train_metrics['accuracy']:.1f}%, Val: {val_metrics['accuracy']:.1f}%")

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epoch >= min_epochs and epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, best_val_acc


def create_dataloader(csv_path: str, config: Dict, scaler=None, is_train: bool = False):
    """Create a dataloader from a CSV file"""
    tabular_features = config['data']['tabular_features']
    image_size = config['model']['vit']['image_size']
    preproc = config.get('preprocessing', {})
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']

    dataset = MultiModalDataset(
        csv_path,
        tabular_features=tabular_features,
        target_shape=(image_size, image_size, image_size),
        augment=is_train,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    return loader, dataset


def run_experiment(config: Dict, n_folds: int, seeds: List[int], output_dir: Path, use_wandb: bool = False):
    """Run ADNI-only training with external validation"""

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=config.get('wandb', {}).get('project', 'alzheimer'),
                   name=config.get('wandb', {}).get('name', 'adni_external'), config=config)

    # Setup device
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Load ADNI data
    logger.info("Loading ADNI data...")
    adni_all_path = Path(config['data']['train_csv']).parent / 'all.csv'
    if adni_all_path.exists():
        adni_df = pd.read_csv(adni_all_path)
    else:
        # Combine train/val/test
        train_df = pd.read_csv(config['data']['train_csv'])
        val_df = pd.read_csv(config['data']['val_csv'])
        test_df = pd.read_csv(config['data']['test_csv'])
        adni_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    logger.info(f"ADNI: {len(adni_df)} subjects")
    logger.info(f"  CN: {(adni_df['label']==0).sum()}, AD-traj: {(adni_df['label']==1).sum()}")

    # Load external test sets
    external_tests = config['data'].get('external_test', {})
    external_dfs = {}
    for name, path in external_tests.items():
        if Path(path).exists():
            external_dfs[name] = pd.read_csv(path)
            logger.info(f"{name.upper()}: {len(external_dfs[name])} subjects (external)")
        else:
            logger.warning(f"External test set not found: {path}")

    # Results storage
    all_results = []
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED: {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        adni_labels = adni_df['label'].values

        seed_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(adni_df)), adni_labels)):
            logger.info(f"\n--- Fold {fold+1}/{n_folds} ---")

            # Split train into train/val
            set_seed(seed + fold)
            n_train = len(train_idx)
            val_size = int(0.1 * n_train)
            perm = np.random.permutation(n_train)
            val_idx = train_idx[perm[:val_size]]
            train_idx_final = train_idx[perm[val_size:]]

            # Save fold CSVs
            train_df = adni_df.iloc[train_idx_final].reset_index(drop=True)
            val_df = adni_df.iloc[val_idx].reset_index(drop=True)
            test_df = adni_df.iloc[test_idx].reset_index(drop=True)

            train_csv = temp_dir / 'train.csv'
            val_csv = temp_dir / 'val.csv'
            test_csv = temp_dir / 'test.csv'

            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
            test_df.to_csv(test_csv, index=False)

            logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test ADNI: {len(test_df)}")

            # Create dataloaders
            train_loader, train_dataset = create_dataloader(str(train_csv), config, scaler=None, is_train=True)
            scaler = train_dataset.get_scaler()

            val_loader, _ = create_dataloader(str(val_csv), config, scaler=scaler, is_train=False)
            test_loader, _ = create_dataloader(str(test_csv), config, scaler=scaler, is_train=False)

            # Create external test loaders
            external_loaders = {}
            for name, ext_df in external_dfs.items():
                ext_csv = temp_dir / f'{name}.csv'
                ext_df.to_csv(ext_csv, index=False)
                ext_loader, _ = create_dataloader(str(ext_csv), config, scaler=scaler, is_train=False)
                external_loaders[name] = ext_loader

            # Train
            trainer = Trainer(config, device)
            num_features = len(config['data']['tabular_features'])
            model = trainer.build_model(num_features)
            optimizer, scheduler = trainer.build_optimizer(model)
            criterion = trainer.build_criterion(train_df['label'].tolist())

            cfg_cb = config['callbacks']['early_stopping']
            model, best_val_acc = trainer.train(
                model, train_loader, val_loader, optimizer, scheduler, criterion,
                num_epochs=config['training']['epochs'],
                patience=cfg_cb['patience'],
                min_epochs=cfg_cb.get('min_epochs', 0)
            )

            # Evaluate on ADNI (in-domain)
            adni_metrics = trainer.evaluate(model, test_loader, criterion)

            # Evaluate on external test sets
            external_metrics = {}
            for name, ext_loader in external_loaders.items():
                ext_metrics = trainer.evaluate(model, ext_loader, criterion)
                external_metrics[name] = ext_metrics

            # Log results (same format as train_cv.py)
            bal_acc = (adni_metrics['sensitivity'] + adni_metrics['specificity']) / 2
            logger.info(f"  Val: {best_val_acc:.1f}% | "
                       f"ADNI: Acc={adni_metrics['accuracy']:.1f}%, BalAcc={bal_acc:.1f}%, "
                       f"Sens={adni_metrics['sensitivity']:.1f}%, Spec={adni_metrics['specificity']:.1f}%, "
                       f"AUC={adni_metrics['auc']:.3f}")

            for name, metrics in external_metrics.items():
                ext_bal_acc = (metrics['sensitivity'] + metrics['specificity']) / 2
                logger.info(f"       {name.upper()} (external, {metrics['n_samples']}): Acc={metrics['accuracy']:.1f}%, "
                           f"BalAcc={ext_bal_acc:.1f}%, Sens={metrics['sensitivity']:.1f}%, "
                           f"Spec={metrics['specificity']:.1f}%, AUC={metrics['auc']:.3f}")

            # Store results
            result = {
                'seed': seed,
                'fold': fold,
                'val_accuracy': best_val_acc,
                'adni_accuracy': adni_metrics['accuracy'],
                'adni_balanced_accuracy': bal_acc,
                'adni_sensitivity': adni_metrics['sensitivity'],
                'adni_specificity': adni_metrics['specificity'],
                'adni_auc': adni_metrics['auc'],
            }
            for name, metrics in external_metrics.items():
                ext_bal_acc = (metrics['sensitivity'] + metrics['specificity']) / 2
                result[f'{name}_accuracy'] = metrics['accuracy']
                result[f'{name}_balanced_accuracy'] = ext_bal_acc
                result[f'{name}_sensitivity'] = metrics['sensitivity']
                result[f'{name}_specificity'] = metrics['specificity']
                result[f'{name}_auc'] = metrics['auc']

            seed_results.append(result)
            all_results.append(result)

            # WandB logging per fold (same structure as train_cv.py)
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'test/seed_{seed}_fold_{fold}_adni_accuracy': adni_metrics['accuracy'],
                    f'test/seed_{seed}_fold_{fold}_adni_balanced_accuracy': bal_acc,
                    f'test/seed_{seed}_fold_{fold}_adni_sensitivity': adni_metrics['sensitivity'],
                    f'test/seed_{seed}_fold_{fold}_adni_specificity': adni_metrics['specificity'],
                    f'test/seed_{seed}_fold_{fold}_adni_auc': adni_metrics['auc'],
                })
                for name, metrics in external_metrics.items():
                    ext_bal_acc = (metrics['sensitivity'] + metrics['specificity']) / 2
                    wandb.log({
                        f'test/seed_{seed}_fold_{fold}_{name}_accuracy': metrics['accuracy'],
                        f'test/seed_{seed}_fold_{fold}_{name}_balanced_accuracy': ext_bal_acc,
                        f'test/seed_{seed}_fold_{fold}_{name}_sensitivity': metrics['sensitivity'],
                        f'test/seed_{seed}_fold_{fold}_{name}_specificity': metrics['specificity'],
                        f'test/seed_{seed}_fold_{fold}_{name}_auc': metrics['auc'],
                    })

        # Seed summary (same format as train_cv.py)
        adni_accs = [r['adni_accuracy'] for r in seed_results]
        adni_bal_accs = [r['adni_balanced_accuracy'] for r in seed_results]
        adni_sens = [r['adni_sensitivity'] for r in seed_results]
        adni_spec = [r['adni_specificity'] for r in seed_results]
        adni_aucs = [r['adni_auc'] for r in seed_results]

        logger.info(f"\nSeed {seed} ADNI (in-domain) Summary:")
        logger.info(f"  Acc: {np.mean(adni_accs):.1f}±{np.std(adni_accs):.1f}%")
        logger.info(f"  BalAcc: {np.mean(adni_bal_accs):.1f}±{np.std(adni_bal_accs):.1f}%")
        logger.info(f"  Sens: {np.mean(adni_sens):.1f}±{np.std(adni_sens):.1f}%")
        logger.info(f"  Spec: {np.mean(adni_spec):.1f}±{np.std(adni_spec):.1f}%")
        logger.info(f"  AUC: {np.mean(adni_aucs):.3f}±{np.std(adni_aucs):.3f}")

        for name in external_dfs.keys():
            ext_accs = [r[f'{name}_accuracy'] for r in seed_results]
            ext_bal_accs = [r[f'{name}_balanced_accuracy'] for r in seed_results]
            ext_sens = [r[f'{name}_sensitivity'] for r in seed_results]
            ext_spec = [r[f'{name}_specificity'] for r in seed_results]
            ext_aucs = [r[f'{name}_auc'] for r in seed_results]

            logger.info(f"Seed {seed} {name.upper()} (external) Summary:")
            logger.info(f"  Acc: {np.mean(ext_accs):.1f}±{np.std(ext_accs):.1f}%")
            logger.info(f"  BalAcc: {np.mean(ext_bal_accs):.1f}±{np.std(ext_bal_accs):.1f}%")
            logger.info(f"  Sens: {np.mean(ext_sens):.1f}±{np.std(ext_sens):.1f}%")
            logger.info(f"  Spec: {np.mean(ext_spec):.1f}±{np.std(ext_spec):.1f}%")
            logger.info(f"  AUC: {np.mean(ext_aucs):.3f}±{np.std(ext_aucs):.3f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS - ADNI-Only Training")
    logger.info(f"{'='*60}")

    results_df = pd.DataFrame(all_results)

    # ADNI (in-domain)
    adni_acc = results_df['adni_accuracy'].mean()
    adni_acc_std = results_df['adni_accuracy'].std()
    adni_bal_acc = results_df['adni_balanced_accuracy'].mean()
    adni_bal_acc_std = results_df['adni_balanced_accuracy'].std()
    adni_sens = results_df['adni_sensitivity'].mean()
    adni_sens_std = results_df['adni_sensitivity'].std()
    adni_spec = results_df['adni_specificity'].mean()
    adni_spec_std = results_df['adni_specificity'].std()
    adni_auc = results_df['adni_auc'].mean()
    adni_auc_std = results_df['adni_auc'].std()

    logger.info(f"\nADNI (in-domain):")
    logger.info(f"  Accuracy:     {adni_acc:.1f}±{adni_acc_std:.1f}%")
    logger.info(f"  Balanced Acc: {adni_bal_acc:.1f}±{adni_bal_acc_std:.1f}%")
    logger.info(f"  Sensitivity:  {adni_sens:.1f}±{adni_sens_std:.1f}%")
    logger.info(f"  Specificity:  {adni_spec:.1f}±{adni_spec_std:.1f}%")
    logger.info(f"  AUC:          {adni_auc:.3f}±{adni_auc_std:.3f}")

    # External test sets
    external_summaries = {}
    for name in external_dfs.keys():
        ext_acc = results_df[f'{name}_accuracy'].mean()
        ext_acc_std = results_df[f'{name}_accuracy'].std()
        ext_bal_acc = results_df[f'{name}_balanced_accuracy'].mean()
        ext_bal_acc_std = results_df[f'{name}_balanced_accuracy'].std()
        ext_sens = results_df[f'{name}_sensitivity'].mean()
        ext_sens_std = results_df[f'{name}_sensitivity'].std()
        ext_spec = results_df[f'{name}_specificity'].mean()
        ext_spec_std = results_df[f'{name}_specificity'].std()
        ext_auc = results_df[f'{name}_auc'].mean()
        ext_auc_std = results_df[f'{name}_auc'].std()

        logger.info(f"\n{name.upper()} (external):")
        logger.info(f"  Accuracy:     {ext_acc:.1f}±{ext_acc_std:.1f}%")
        logger.info(f"  Balanced Acc: {ext_bal_acc:.1f}±{ext_bal_acc_std:.1f}%")
        logger.info(f"  Sensitivity:  {ext_sens:.1f}±{ext_sens_std:.1f}%")
        logger.info(f"  Specificity:  {ext_spec:.1f}±{ext_spec_std:.1f}%")
        logger.info(f"  AUC:          {ext_auc:.3f}±{ext_auc_std:.3f}")

        external_summaries[name] = {
            'accuracy': float(ext_acc),
            'accuracy_std': float(ext_acc_std),
            'balanced_accuracy': float(ext_bal_acc),
            'balanced_accuracy_std': float(ext_bal_acc_std),
            'sensitivity': float(ext_sens),
            'sensitivity_std': float(ext_sens_std),
            'specificity': float(ext_spec),
            'specificity_std': float(ext_spec_std),
            'auc': float(ext_auc),
            'auc_std': float(ext_auc_std),
        }

    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    summary = {
        'n_folds': n_folds,
        'seeds': seeds,
        'adni_accuracy': float(adni_acc),
        'adni_accuracy_std': float(adni_acc_std),
        'adni_balanced_accuracy': float(adni_bal_acc),
        'adni_balanced_accuracy_std': float(adni_bal_acc_std),
        'adni_sensitivity': float(adni_sens),
        'adni_sensitivity_std': float(adni_sens_std),
        'adni_specificity': float(adni_spec),
        'adni_specificity_std': float(adni_spec_std),
        'adni_auc': float(adni_auc),
        'adni_auc_std': float(adni_auc_std),
        'external': external_summaries,
        'per_fold_results': all_results
    }

    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # WandB final summary (same structure as train_cv.py)
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'final/adni_accuracy_mean': adni_acc,
            'final/adni_accuracy_std': adni_acc_std,
            'final/adni_balanced_accuracy_mean': adni_bal_acc,
            'final/adni_balanced_accuracy_std': adni_bal_acc_std,
            'final/adni_sensitivity_mean': adni_sens,
            'final/adni_sensitivity_std': adni_sens_std,
            'final/adni_specificity_mean': adni_spec,
            'final/adni_specificity_std': adni_spec_std,
            'final/adni_auc_mean': adni_auc,
            'final/adni_auc_std': adni_auc_std,
        })
        for name, ext_summary in external_summaries.items():
            wandb.log({
                f'final/{name}_accuracy_mean': ext_summary['accuracy'],
                f'final/{name}_accuracy_std': ext_summary['accuracy_std'],
                f'final/{name}_balanced_accuracy_mean': ext_summary['balanced_accuracy'],
                f'final/{name}_balanced_accuracy_std': ext_summary['balanced_accuracy_std'],
                f'final/{name}_sensitivity_mean': ext_summary['sensitivity'],
                f'final/{name}_sensitivity_std': ext_summary['sensitivity_std'],
                f'final/{name}_specificity_mean': ext_summary['specificity'],
                f'final/{name}_specificity_std': ext_summary['specificity_std'],
                f'final/{name}_auc_mean': ext_summary['auc'],
                f'final/{name}_auc_std': ext_summary['auc_std'],
            })
        wandb.finish()

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description='ADNI-Only Training with External Validation')
    parser.add_argument('--config', type=str, default='config_adni.yaml')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--output-dir', type=str, default='results_adni_external')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_wandb = config.get('wandb', {}).get('enabled', False) and not args.no_wandb

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_experiment(config, args.n_folds, args.seeds, output_dir, use_wandb=use_wandb)
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
