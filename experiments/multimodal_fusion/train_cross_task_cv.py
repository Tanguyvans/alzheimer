#!/usr/bin/env python3
"""
Cross-Task Evaluation with Cross-Validation and Multiple Seeds

Train on cn_ad_trajectory, test on both cn_ad_trajectory and cn_ad test sets.
Uses 5-fold CV with 3 seeds for robust evaluation.

Usage:
    python train_cross_task_cv.py --config config.yaml --cn-ad-test data/combined_cn_ad/test.csv
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import wandb

from model import build_model
from dataset import MultiModalDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mri, tabular, labels in loader:
        mri, tabular, labels = mri.to(device), tabular.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mri, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return {'loss': total_loss / len(loader), 'accuracy': 100. * correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for mri, tabular, labels in loader:
        mri, tabular = mri.to(device), tabular.to(device)
        outputs = model(mri, tabular)
        probs = torch.softmax(outputs, dim=1)
        _, pred = outputs.max(1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    accuracy = 100. * (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds)

    class_accs = []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        class_accs.append((all_preds[mask] == all_labels[mask]).mean())
    balanced_acc = 100. * np.mean(class_accs)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity, specificity = 0, 0

    # Compute AUC
    all_probs = np.array(all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def create_dataset(csv_path, config, scaler=None, augment=False):
    tabular_features = config['data']['tabular_features']
    if 'backbone' in config['model']:
        image_size = config['model']['backbone'].get('image_size', 128)
    elif 'vit' in config['model']:
        image_size = config['model']['vit']['image_size']
    else:
        image_size = 128
    preproc = config.get('preprocessing', {})

    return MultiModalDataset(
        csv_path,
        tabular_features=tabular_features,
        target_shape=(image_size, image_size, image_size),
        augment=augment,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )


def train_fold(model, train_loader, val_loader, config, device, weights, fold_id=0, use_wandb=False):
    cfg = config['training']
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0), eta_min=cfg['lr_min']
    )
    warmup = cfg.get('warmup_epochs', 0)
    if warmup > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, main_scheduler], [warmup])
    else:
        scheduler = main_scheduler

    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state = None
    early_stop_cfg = config['callbacks']['early_stopping']
    early_stop_enabled = early_stop_cfg.get('enabled', True)
    patience = early_stop_cfg['patience']
    min_epochs = early_stop_cfg.get('min_epochs', 0)
    no_improve = 0

    pbar = tqdm(range(cfg['epochs']), desc="Training", leave=False)
    for epoch in pbar:
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        pbar.set_postfix({'train': f"{train_m['accuracy']:.1f}%", 'val': f"{val_m['accuracy']:.1f}%"})

        # Log to wandb
        if use_wandb:
            wandb.log({
                f'fold{fold_id}/train_acc': train_m['accuracy'],
                f'fold{fold_id}/val_acc': val_m['accuracy'],
                f'fold{fold_id}/val_bal_acc': val_m['balanced_accuracy'],
                f'fold{fold_id}/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })

        if val_m['accuracy'] > best_val_acc:
            best_val_acc = val_m['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Only allow early stopping if enabled and after min_epochs
        if early_stop_enabled and epoch >= min_epochs and no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--cn-ad-test', default='data/combined_cn_ad/test.csv')
    parser.add_argument('--output-dir', default='cross_task_cv_results')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize wandb if enabled
    use_wandb = config.get('wandb', {}).get('enabled', False)
    if use_wandb:
        wandb.init(
            project=config['wandb'].get('project', 'alzheimer-multimodal'),
            name=config['wandb'].get('name', 'cross_task_cv'),
            config=config
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # Load combined train+val for CV
    train_csv = config['data']['train_csv']
    val_csv = config['data']['val_csv']

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    combined_csv = output_dir / 'combined_train_val.csv'
    combined_df.to_csv(combined_csv, index=False)
    logger.info(f"Combined train+val: {len(combined_df)} samples")

    # Test sets (fixed)
    test_traj_csv = config['data']['test_csv']
    test_cnad_csv = args.cn_ad_test

    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    num_features = len(config['data']['tabular_features'])

    all_results = []

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)

        # Create full dataset for CV splits
        full_ds = create_dataset(combined_csv, config, augment=False)
        labels = np.array([full_ds[i][2] if isinstance(full_ds[i][2], int) else full_ds[i][2].item()
                         for i in range(len(full_ds))])
        scaler = full_ds.get_scaler()

        # Test datasets
        test_traj_ds = create_dataset(test_traj_csv, config, scaler=scaler)
        test_cnad_ds = create_dataset(test_cnad_csv, config, scaler=scaler)

        test_traj_loader = DataLoader(test_traj_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)
        test_cnad_loader = DataLoader(test_cnad_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)

        seed_traj_results = []
        seed_cnad_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_ds)), labels)):
            logger.info(f"\n--- Fold {fold+1}/{args.n_folds} ---")

            # Create train dataset with augmentation
            train_ds_aug = create_dataset(combined_csv, config, scaler=scaler, augment=True)
            train_subset = Subset(train_ds_aug, train_idx)
            val_subset = Subset(full_ds, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

            # Class weights
            train_labels = labels[train_idx]
            class_counts = np.bincount(train_labels)
            weights = torch.FloatTensor(1.0 / class_counts).to(device)
            weights = weights / weights.sum() * len(class_counts)

            # Fresh model
            model = build_model(config, num_features).to(device)
            model, best_val = train_fold(model, train_loader, val_loader, config, device, weights,
                                         fold_id=fold+1, use_wandb=use_wandb)

            # Evaluate on both test sets
            traj_m = evaluate(model, test_traj_loader, device)
            cnad_m = evaluate(model, test_cnad_loader, device)

            logger.info(f"  Val: {best_val:.1f}% | Traj: {traj_m['balanced_accuracy']:.1f}% (AUC: {traj_m['auc']:.3f}) | CN_AD: {cnad_m['balanced_accuracy']:.1f}% (AUC: {cnad_m['auc']:.3f})")

            seed_traj_results.append({
                'accuracy': traj_m['accuracy'],
                'balanced_accuracy': traj_m['balanced_accuracy'],
                'sensitivity': traj_m['sensitivity'],
                'specificity': traj_m['specificity'],
                'auc': traj_m['auc']
            })

            seed_cnad_results.append({
                'accuracy': cnad_m['accuracy'],
                'balanced_accuracy': cnad_m['balanced_accuracy'],
                'sensitivity': cnad_m['sensitivity'],
                'specificity': cnad_m['specificity'],
                'auc': cnad_m['auc']
            })

            # Save fold model
            torch.save(model.state_dict(), output_dir / f'model_seed{seed}_fold{fold}.pth')

        # Aggregate seed results
        all_results.append({
            'seed': seed,
            'cn_ad_trajectory': seed_traj_results,
            'cn_ad': seed_cnad_results
        })

    # Compute overall statistics
    def aggregate_metrics(results_list):
        metrics = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'auc']
        agg = {}
        for m in metrics:
            vals = [r[m] for r in results_list]
            agg[m] = {'mean': np.mean(vals), 'std': np.std(vals)}
        return agg

    # Flatten all folds across all seeds
    all_traj = [r for seed_res in all_results for r in seed_res['cn_ad_trajectory']]
    all_cnad = [r for seed_res in all_results for r in seed_res['cn_ad']]

    traj_agg = aggregate_metrics(all_traj)
    cnad_agg = aggregate_metrics(all_cnad)

    # Summary
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS (5-fold CV x 3 seeds = 15 runs)")
    logger.info("="*100)
    logger.info(f"{'Test Set':<25} {'Accuracy':<18} {'Balanced':<18} {'Sensitivity':<18} {'Specificity':<18} {'AUC':<12}")
    logger.info("-"*100)
    logger.info(f"{'cn_ad_trajectory':<25} "
                f"{traj_agg['accuracy']['mean']:.1f}±{traj_agg['accuracy']['std']:.1f}%{'':<5} "
                f"{traj_agg['balanced_accuracy']['mean']:.1f}±{traj_agg['balanced_accuracy']['std']:.1f}%{'':<5} "
                f"{traj_agg['sensitivity']['mean']:.1f}±{traj_agg['sensitivity']['std']:.1f}%{'':<5} "
                f"{traj_agg['specificity']['mean']:.1f}±{traj_agg['specificity']['std']:.1f}%{'':<5} "
                f"{traj_agg['auc']['mean']:.3f}±{traj_agg['auc']['std']:.3f}")
    logger.info(f"{'cn_ad (stable AD)':<25} "
                f"{cnad_agg['accuracy']['mean']:.1f}±{cnad_agg['accuracy']['std']:.1f}%{'':<5} "
                f"{cnad_agg['balanced_accuracy']['mean']:.1f}±{cnad_agg['balanced_accuracy']['std']:.1f}%{'':<5} "
                f"{cnad_agg['sensitivity']['mean']:.1f}±{cnad_agg['sensitivity']['std']:.1f}%{'':<5} "
                f"{cnad_agg['specificity']['mean']:.1f}±{cnad_agg['specificity']['std']:.1f}%{'':<5} "
                f"{cnad_agg['auc']['mean']:.3f}±{cnad_agg['auc']['std']:.3f}")
    logger.info("="*100)

    # Save results
    final_results = {
        'n_folds': args.n_folds,
        'seeds': args.seeds,
        'total_runs': len(all_traj),
        'cn_ad_trajectory': {
            'accuracy': {'mean': traj_agg['accuracy']['mean'], 'std': traj_agg['accuracy']['std']},
            'balanced_accuracy': {'mean': traj_agg['balanced_accuracy']['mean'], 'std': traj_agg['balanced_accuracy']['std']},
            'sensitivity': {'mean': traj_agg['sensitivity']['mean'], 'std': traj_agg['sensitivity']['std']},
            'specificity': {'mean': traj_agg['specificity']['mean'], 'std': traj_agg['specificity']['std']},
            'auc': {'mean': traj_agg['auc']['mean'], 'std': traj_agg['auc']['std']}
        },
        'cn_ad': {
            'accuracy': {'mean': cnad_agg['accuracy']['mean'], 'std': cnad_agg['accuracy']['std']},
            'balanced_accuracy': {'mean': cnad_agg['balanced_accuracy']['mean'], 'std': cnad_agg['balanced_accuracy']['std']},
            'sensitivity': {'mean': cnad_agg['sensitivity']['mean'], 'std': cnad_agg['sensitivity']['std']},
            'specificity': {'mean': cnad_agg['specificity']['mean'], 'std': cnad_agg['specificity']['std']},
            'auc': {'mean': cnad_agg['auc']['mean'], 'std': cnad_agg['auc']['std']}
        },
        'per_seed': all_results
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Log final summary to wandb
    if use_wandb:
        wandb.log({
            'final/traj_acc': traj_agg['accuracy']['mean'],
            'final/traj_bal_acc': traj_agg['balanced_accuracy']['mean'],
            'final/traj_auc': traj_agg['auc']['mean'],
            'final/cnad_acc': cnad_agg['accuracy']['mean'],
            'final/cnad_bal_acc': cnad_agg['balanced_accuracy']['mean'],
            'final/cnad_auc': cnad_agg['auc']['mean'],
        })
        wandb.finish()

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
