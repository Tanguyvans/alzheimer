#!/usr/bin/env python3
"""
Quick single-run MRI training to test improvements
No CV - just train/val/test split

Usage:
    python train_single.py --config config_improved.yaml
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import argparse
import nibabel as nib
from scipy import ndimage
import math

sys.path.append(str(Path(__file__).parent.parent / "mri_vit_ad"))
from model import ViT3DClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                   reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [lr * self.last_epoch / self.warmup_epochs for lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return [self.min_lr + (lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for lr in self.base_lrs]


def get_layer_lr_groups(model, base_lr, decay=0.75):
    """Layer-wise LR decay for ViT"""
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'blocks.' in name:
            block_num = int(name.split('blocks.')[1].split('.')[0])
            lr = base_lr * (decay ** (12 - block_num))
        elif 'patch_embed' in name or 'pos_embed' in name:
            lr = base_lr * (decay ** 12)
        else:
            lr = base_lr
        groups.append({'params': [param], 'lr': lr})
    return groups


class MRIDataset(Dataset):
    def __init__(self, csv_path, target_size=128, augment=False):
        self.df = pd.read_csv(csv_path)
        self.target_size = target_size
        self.augment = augment

        if 'label' in self.df.columns:
            self.labels = self.df['label'].values
        else:
            dx_map = {'CN': 0, 'AD': 1, 'MCI': 1, 'Dementia': 1}
            self.labels = self.df['DX'].map(dx_map).values

        for col in ['scan_path', 'mri_path', 'npy_path']:
            if col in self.df.columns:
                self.mri_paths = self.df[col].values
                break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.mri_paths[idx]
        label = self.labels[idx]

        try:
            if path.endswith('.npy'):
                vol = np.load(path)
            else:
                vol = nib.load(path).get_fdata()

            if vol.shape != (self.target_size,) * 3:
                zoom = [self.target_size / s for s in vol.shape]
                vol = ndimage.zoom(vol, zoom, order=1)

            vol = (vol - vol.mean()) / (vol.std() + 1e-8)

            if self.augment:
                if np.random.rand() > 0.5:
                    vol = np.flip(vol, axis=np.random.randint(0, 3)).copy()
                vol = vol * (1 + np.random.uniform(-0.1, 0.1))

        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            vol = np.zeros((self.target_size,) * 3, dtype=np.float32)

        return torch.tensor(vol[np.newaxis], dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train(config):
    cfg = config['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('seed', 42))

    data_dir = Path(config['data']['data_dir'])
    train_ds = MRIDataset(data_dir / "train.csv", augment=True)
    val_ds = MRIDataset(data_dir / "val.csv")
    test_ds = MRIDataset(data_dir / "test.csv")

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Model
    model_cfg = config['model']
    model = ViT3DClassifier(
        image_size=model_cfg.get('image_size', 128),
        num_classes=2,
        dropout=model_cfg.get('dropout', 0.2),
        classifier_dropout=model_cfg.get('classifier_dropout', 0.6),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.2)
    )

    # Load pretrained
    pretrained = model_cfg.get('pretrained_path')
    if pretrained and os.path.exists(pretrained):
        logger.info(f"Loading pretrained: {pretrained}")
        state = torch.load(pretrained, map_location='cpu')
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=False)

    model = model.to(device)

    # Optimizer with layer-wise LR
    if cfg.get('layer_wise_lr_decay', 0) > 0:
        params = get_layer_lr_groups(model, cfg['learning_rate'], cfg['layer_wise_lr_decay'])
        optimizer = optim.AdamW(params, weight_decay=cfg['weight_decay'])
        logger.info(f"Layer-wise LR decay: {cfg['layer_wise_lr_decay']}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    # Scheduler
    scheduler = CosineWarmupScheduler(optimizer, cfg.get('warmup_epochs', 10), cfg['epochs'], cfg['lr_min'])

    # Loss
    class_counts = np.bincount(train_ds.labels)
    weights = torch.tensor(len(class_counts) / class_counts, dtype=torch.float32).to(device)

    if cfg.get('use_focal_loss', False):
        criterion = FocalLoss(weights, cfg.get('focal_gamma', 2.0), cfg.get('label_smoothing', 0.1))
        logger.info("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get('label_smoothing', 0.1))

    best_val = 0
    best_state = None
    patience_counter = 0
    patience = config['callbacks']['early_stopping']['patience']

    use_mixup = cfg.get('use_mixup', False)
    accum = cfg.get('gradient_accumulation', 1)

    for epoch in range(cfg['epochs']):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        optimizer.zero_grad()

        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            x, y = x.to(device), y.to(device)

            if use_mixup and np.random.rand() > 0.5:
                x, y_a, y_b, lam = mixup_data(x, y, 0.2)
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                out = model(x)
                loss = criterion(out, y)

            (loss / accum).backward()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('gradient_clip', 1.0))
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_correct += out.argmax(1).eq(y).sum().item()
            train_total += y.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = 100 * val_correct / val_total

        # Balanced accuracy
        val_preds, val_labels = np.array(val_preds), np.array(val_labels)
        bal_acc = np.mean([((val_preds[val_labels == c] == c).mean() if (val_labels == c).sum() > 0 else 0)
                          for c in [0, 1]]) * 100

        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} | LR: {lr:.2e} | "
                   f"Train: {100*train_correct/train_total:.1f}% | Val: {val_acc:.1f}% | Bal: {bal_acc:.1f}%")

        # Early stopping on balanced accuracy
        if bal_acc > best_val:
            best_val = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    test_correct, test_total = 0, 0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # TTA
            out = (model(x) + model(torch.flip(x, [-1])) + model(torch.flip(x, [-2]))) / 3
            pred = out.argmax(1)
            test_correct += pred.eq(y).sum().item()
            test_total += y.size(0)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = 100 * test_correct / test_total
    test_preds, test_labels = np.array(test_preds), np.array(test_labels)
    test_bal = np.mean([((test_preds[test_labels == c] == c).mean() if (test_labels == c).sum() > 0 else 0)
                        for c in [0, 1]]) * 100

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: Test Acc: {test_acc:.2f}% | Balanced: {test_bal:.2f}%")
    logger.info(f"{'='*50}")

    return test_acc, test_bal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_improved.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
