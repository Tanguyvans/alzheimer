#!/usr/bin/env python3
"""
ResNet3D + MLP Early Fusion — single train/val/test run with fine-tuning.

Early fusion: ResNet3D backbone + Tabular MLP → concat → classifier (end-to-end).
Differential LR (backbone 10x lower) + backbone freezing for first N epochs.

Usage:
    python train.py --config config.yaml
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import logging
import argparse
import json
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
import importlib.util

from model import EarlyFusionModel, build_early_fusion_model

# Import MultiModalDataset from multimodal_fusion
_mm_dataset_path = Path(__file__).parent.parent / "multimodal_fusion" / "dataset.py"
_spec = importlib.util.spec_from_file_location("multimodal_fusion_dataset", _mm_dataset_path)
_mm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm_module)
MultiModalDataset = _mm_module.MultiModalDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(config: Dict) -> torch.device:
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_metrics(y_true, y_pred, y_proba) -> Dict:
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5
    return {
        'accuracy': float(acc), 'balanced_accuracy': float(bal_acc),
        'sensitivity': float(sensitivity), 'specificity': float(specificity),
        'auc': float(auc), 'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def log_metrics(name, metrics):
    logger.info(f"\n{name}:")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.1f}%")
    logger.info(f"  Balanced Acc: {metrics['balanced_accuracy']:.1f}%")
    logger.info(f"  Sensitivity:  {metrics['sensitivity']:.1f}%")
    logger.info(f"  Specificity:  {metrics['specificity']:.1f}%")
    logger.info(f"  AUC:          {metrics['auc']:.3f}")
    logger.info(f"  Confusion:    {metrics['confusion_matrix']}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for mri, tabular, labels in loader:
        mri, tabular, labels = mri.to(device), tabular.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(mri, tabular)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs.float(), dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true, y_pred, y_proba = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='ResNet3D + MLP Early Fusion')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(config)
    train_cfg = config['training']
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']
    batch_size = train_cfg['batch_size']
    num_workers = config['hardware']['num_workers']

    # ── Datasets ──
    logger.info("Loading datasets...")
    train_dataset = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=True, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    val_dataset = MultiModalDataset(
        config['data']['val_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    test_dataset = MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model ──
    model = build_early_fusion_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"EarlyFusionModel params: {total_params:,}")

    # Class weights
    train_labels = pd.read_csv(config['data']['train_csv'])['label'].tolist()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    # Differential LR
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    backbone_lr = train_cfg['learning_rate'] * 0.1

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': train_cfg['learning_rate']},
    ], weight_decay=train_cfg['weight_decay'])

    # Scheduler
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'] - warmup_epochs, eta_min=train_cfg['lr_min']
    )
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine_scheduler

    # Freeze backbone initially
    freeze_backbone_epochs = train_cfg.get('freeze_backbone_epochs', 0)
    if freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs")

    # AMP + gradient accumulation
    use_amp = train_cfg.get('mixed_precision', False) and device.type == 'cuda'
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
    accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
    if use_amp:
        logger.info("Mixed precision (AMP) enabled")
    if accum_steps > 1:
        logger.info(f"Gradient accumulation: {accum_steps} steps (effective batch={batch_size * accum_steps})")

    # Training loop — monitor balanced accuracy to avoid majority-class collapse
    best_val_bal_acc = 0.0
    epochs_no_improve = 0
    patience = config['callbacks']['early_stopping']['patience']
    min_epochs = config['callbacks']['early_stopping'].get('min_epochs', 30)
    best_state = None
    grad_clip = train_cfg.get('gradient_clip', 1.0)

    logger.info(f"\nTraining for up to {train_cfg['epochs']} epochs (patience={patience}, min_epochs={min_epochs})")
    logger.info("=" * 60)

    for epoch in range(train_cfg['epochs']):
        # Unfreeze backbone
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            logger.info(f"Backbone unfrozen at epoch {epoch+1}")

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()
        for step, (mri, tabular, labels) in enumerate(train_loader):
            mri, tabular, labels = mri.to(device), tabular.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(mri, tabular)
                loss = criterion(outputs, labels) / accum_steps

            scaler_amp.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        frozen_str = " [frozen]" if epoch < freeze_backbone_epochs else ""
        logger.info(
            f"Epoch {epoch+1}/{train_cfg['epochs']}: "
            f"train_acc={train_acc:.1f}%  "
            f"val_acc={val_metrics['accuracy']:.1f}% "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.1f}% "
            f"val_auc={val_metrics['auc']:.3f}"
            f"{frozen_str}"
        )

        # Track best by balanced accuracy
        if val_metrics['balanced_accuracy'] > best_val_bal_acc:
            best_val_bal_acc = val_metrics['balanced_accuracy']
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch >= min_epochs and epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    logger.info(f"\nBest val balanced accuracy: {best_val_bal_acc:.1f}%")

    # Evaluate
    for split_name, loader in [('VAL', val_loader), ('TEST', test_loader)]:
        metrics = evaluate(model, loader, criterion, device, use_amp=use_amp)
        log_metrics(f"{split_name} Results", metrics)

    # Save model
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
    torch.save({'model_state_dict': best_state, 'config': config}, output_dir / 'best_model.pth')
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions for analysis (DeLong, confusion matrices)
    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for mri, tabular, labels in loader:
                mri, tabular = mri.to(device), tabular.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(mri, tabular)
                probs = torch.softmax(outputs.float(), dim=1)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        np.save(output_dir / f'y_true_{split_name}.npy', np.array(all_labels))
        np.save(output_dir / f'y_proba_{split_name}.npy', np.array(all_probs))

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
