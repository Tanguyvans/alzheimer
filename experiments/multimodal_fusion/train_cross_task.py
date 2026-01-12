#!/usr/bin/env python3
"""
Cross-Task Evaluation: Train on cn_ad_trajectory, Test on both tasks

Train on cn_ad_trajectory (CN vs AD + MCI converters), then evaluate on:
1. cn_ad_trajectory test set (in-domain)
2. cn_ad test set (cross-task - stable AD only)

Usage:
    python train_cross_task.py --config config.yaml --cn-ad-test data/combined_cn_ad/test.csv
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import confusion_matrix

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

    # Balanced accuracy
    class_accs = []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        class_accs.append((all_preds[mask] == all_labels[mask]).mean())
    balanced_acc = 100. * np.mean(class_accs)

    # Sensitivity/Specificity
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity, specificity = 0, 0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--cn-ad-test', default='data/combined_cn_ad/test.csv',
                        help='Path to cn_ad test CSV')
    parser.add_argument('--output-dir', default='cross_task_results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # Datasets
    logger.info("Loading cn_ad_trajectory datasets (training task)...")
    train_ds = create_dataset(config['data']['train_csv'], config, augment=True)
    scaler = train_ds.get_scaler()
    val_ds = create_dataset(config['data']['val_csv'], config, scaler=scaler)
    test_traj_ds = create_dataset(config['data']['test_csv'], config, scaler=scaler)

    logger.info(f"Loading cn_ad test set from {args.cn_ad_test}...")
    test_cnad_ds = create_dataset(args.cn_ad_test, config, scaler=scaler)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    logger.info(f"Test trajectory: {len(test_traj_ds)}, Test cn_ad: {len(test_cnad_ds)}")

    # Dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_traj_loader = DataLoader(test_traj_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    test_cnad_loader = DataLoader(test_cnad_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    # Model
    num_features = len(config['data']['tabular_features'])
    model = build_model(config, num_features).to(device)

    # Optimizer & Scheduler
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

    # Loss with class weights
    train_labels = [train_ds[i][2].item() for i in range(len(train_ds))]
    class_counts = np.bincount(train_labels)
    weights = torch.FloatTensor(1.0 / class_counts).to(device)
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training
    best_val_acc = 0.0
    best_state = None
    patience = config['callbacks']['early_stopping']['patience']
    no_improve = 0

    logger.info("\n" + "="*60)
    logger.info("TRAINING ON cn_ad_trajectory")
    logger.info("="*60)

    pbar = tqdm(range(cfg['epochs']), desc="Training")
    for epoch in pbar:
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, device)
        scheduler.step()

        pbar.set_postfix({'train': f"{train_m['accuracy']:.1f}%", 'val': f"{val_m['accuracy']:.1f}%"})

        if val_m['accuracy'] > best_val_acc:
            best_val_acc = val_m['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / 'best_model.pth')

    # Evaluation
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)

    # Test on cn_ad_trajectory
    logger.info("\n--- cn_ad_trajectory (in-domain) ---")
    traj_m = evaluate(model, test_traj_loader, device)
    logger.info(f"Accuracy: {traj_m['accuracy']:.2f}%")
    logger.info(f"Balanced: {traj_m['balanced_accuracy']:.2f}%")
    logger.info(f"Sensitivity: {traj_m['sensitivity']:.2f}%")
    logger.info(f"Specificity: {traj_m['specificity']:.2f}%")

    # Subgroup analysis
    traj_df = pd.read_csv(config['data']['test_csv'])
    try:
        clinical = config['data'].get('clinical_csv', '../../data/adni/adni_cn_ad_trajectory.csv')
        traj_ref = pd.read_csv(clinical)[['subject_id', 'trajectory']].drop_duplicates()
        traj_df = traj_df.merge(traj_ref, on='subject_id', how='left')
        traj_df['trajectory'] = traj_df['trajectory'].fillna(traj_df.get('DX', 'Unknown'))
    except:
        traj_df['trajectory'] = traj_df.get('DX', 'Unknown')

    traj_df['prediction'] = traj_m['predictions']
    traj_df['correct'] = traj_df['prediction'] == traj_df['label']

    logger.info("\nSubgroups:")
    subgroups = {}
    for grp in sorted(traj_df['trajectory'].unique()):
        sub = traj_df[traj_df['trajectory'] == grp]
        acc = 100. * sub['correct'].mean()
        subgroups[grp] = {'accuracy': acc, 'n': len(sub)}
        logger.info(f"  {grp}: {acc:.1f}% ({len(sub)} samples)")

    # Test on cn_ad
    logger.info("\n--- cn_ad (cross-task: stable AD only) ---")
    cnad_m = evaluate(model, test_cnad_loader, device)
    logger.info(f"Accuracy: {cnad_m['accuracy']:.2f}%")
    logger.info(f"Balanced: {cnad_m['balanced_accuracy']:.2f}%")
    logger.info(f"Sensitivity (AD): {cnad_m['sensitivity']:.2f}%")
    logger.info(f"Specificity (CN): {cnad_m['specificity']:.2f}%")

    # Save results
    results = {
        'training_task': 'cn_ad_trajectory',
        'seed': args.seed,
        'best_val_accuracy': best_val_acc,
        'test_cn_ad_trajectory': {
            'accuracy': traj_m['accuracy'],
            'balanced_accuracy': traj_m['balanced_accuracy'],
            'sensitivity': traj_m['sensitivity'],
            'specificity': traj_m['specificity'],
            'subgroups': subgroups
        },
        'test_cn_ad': {
            'accuracy': cnad_m['accuracy'],
            'balanced_accuracy': cnad_m['balanced_accuracy'],
            'sensitivity': cnad_m['sensitivity'],
            'specificity': cnad_m['specificity']
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Test Set':<25} {'Acc':<10} {'Balanced':<10} {'Sens':<10} {'Spec':<10}")
    logger.info("-"*60)
    logger.info(f"{'cn_ad_trajectory':<25} {traj_m['accuracy']:.1f}%{'':<4} {traj_m['balanced_accuracy']:.1f}%{'':<4} {traj_m['sensitivity']:.1f}%{'':<4} {traj_m['specificity']:.1f}%")
    logger.info(f"{'cn_ad (stable AD)':<25} {cnad_m['accuracy']:.1f}%{'':<4} {cnad_m['balanced_accuracy']:.1f}%{'':<4} {cnad_m['sensitivity']:.1f}%{'':<4} {cnad_m['specificity']:.1f}%")
    logger.info("="*60)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
