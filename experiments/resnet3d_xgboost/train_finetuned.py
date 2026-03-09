#!/usr/bin/env python3
"""
ResNet3D (fine-tuned) + XGBoost — single train/val/test run.

Phase 1: Fine-tune ResNet3D with a simple linear head (to adapt features to AD task)
Phase 2: Extract features from fine-tuned backbone + XGBoost

Usage:
    python train_finetuned.py --config config.yaml
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import importlib.util

import xgboost as xgb

# Import ResNet3DBackbone from resnet3d_mlp
_resnet_model_path = Path(__file__).parent.parent / "resnet3d_mlp" / "model.py"
_spec_resnet = importlib.util.spec_from_file_location("resnet3d_mlp_model", _resnet_model_path)
_resnet_module = importlib.util.module_from_spec(_spec_resnet)
_spec_resnet.loader.exec_module(_resnet_module)
ResNet3DBackbone = _resnet_module.ResNet3DBackbone

# Import MultiModalDataset from multimodal_fusion
_mm_dataset_path = Path(__file__).parent.parent / "multimodal_fusion" / "dataset.py"
_spec_mm = importlib.util.spec_from_file_location("multimodal_fusion_dataset", _mm_dataset_path)
_mm_module = importlib.util.module_from_spec(_spec_mm)
_spec_mm.loader.exec_module(_mm_module)
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


class ResNet3DClassifier(nn.Module):
    """Simple ResNet3D + linear head for fine-tuning."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNet3DBackbone(pretrained=pretrained)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def extract_features(self, x):
        self.backbone.eval()
        return self.backbone(x)


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


def extract_features(backbone, loader, device):
    """Extract CNN features from backbone."""
    all_cnn, all_tab, all_labels = [], [], []
    backbone.eval()
    with torch.no_grad():
        for mri, tabular, labels in loader:
            mri = mri.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                cnn_feat = backbone(mri)
            all_cnn.append(cnn_feat.float().cpu().numpy())
            all_tab.append(tabular.numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_cnn), np.concatenate(all_tab), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description='ResNet3D (fine-tuned) + XGBoost')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='results_finetuned')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--freeze-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(config)
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']
    batch_size = config['hardware'].get('batch_size', 4)
    num_workers = config['hardware']['num_workers']
    xgb_cfg = config['model']['xgboost']

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ═══════════════════════════════════════════════
    # PHASE 1: Fine-tune ResNet3D with simple head
    # ═══════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1: Fine-tuning ResNet3D backbone")
    logger.info("=" * 60)

    model = ResNet3DClassifier(pretrained=True, num_classes=2).to(device)
    logger.info(f"ResNet3D params: {sum(p.numel() for p in model.parameters()):,}")

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

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-7)

    # Freeze backbone initially
    if args.freeze_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info(f"Backbone frozen for first {args.freeze_epochs} epochs")

    # AMP
    scaler_amp = torch.amp.GradScaler('cuda', enabled=True)

    best_val_bal_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    patience = 15

    for epoch in range(args.finetune_epochs):
        # Unfreeze
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            logger.info(f"Backbone unfrozen at epoch {epoch+1}")

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for mri, tabular, labels in train_loader:
            mri, labels = mri.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(mri)
                loss = criterion(outputs, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for mri, tabular, labels in val_loader:
                mri, labels = mri.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(mri)
                    loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                probs = torch.softmax(outputs.float(), dim=1)
                _, pred = outputs.max(1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_preds, val_labels_arr = np.array(val_preds), np.array(val_labels)
        val_acc = accuracy_score(val_labels_arr, val_preds) * 100
        val_bal_acc = balanced_accuracy_score(val_labels_arr, val_preds) * 100
        try:
            val_auc = roc_auc_score(val_labels_arr, np.array(val_probs))
        except:
            val_auc = 0.5

        frozen_str = " [frozen]" if epoch < args.freeze_epochs else ""
        logger.info(
            f"Epoch {epoch+1}/{args.finetune_epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}%  "
            f"val_acc={val_acc:.1f}% val_bal_acc={val_bal_acc:.1f}% val_auc={val_auc:.3f}"
            f"{frozen_str}"
        )

        # Track best by balanced accuracy (not accuracy, to avoid majority-class trap)
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch >= 10:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info(f"Best val balanced accuracy: {best_val_bal_acc:.1f}%")

    # Save fine-tuned backbone
    torch.save(best_state, output_dir / 'finetuned_resnet3d.pth')

    # ═══════════════════════════════════════════════
    # PHASE 2: Extract features + XGBoost
    # ═══════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2: Feature extraction + XGBoost")
    logger.info("=" * 60)

    # Recreate loaders without augmentation for feature extraction
    train_dataset_noaug = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=train_dataset.get_scaler(),
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info("Extracting features from fine-tuned backbone...")
    X_cnn_train, X_tab_train, y_train = extract_features(model.backbone, train_loader_noaug, device)
    X_cnn_val, X_tab_val, y_val = extract_features(model.backbone, val_loader, device)
    X_cnn_test, X_tab_test, y_test = extract_features(model.backbone, test_loader, device)

    X_train = np.hstack([X_cnn_train, X_tab_train])
    X_val = np.hstack([X_cnn_val, X_tab_val])
    X_test = np.hstack([X_cnn_test, X_tab_test])
    logger.info(f"Feature dims: CNN={X_cnn_train.shape[1]}, Tab={X_tab_train.shape[1]}, Total={X_train.shape[1]}")

    # Scale
    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_val = feat_scaler.transform(X_val)
    X_test = feat_scaler.transform(X_test)

    # XGBoost
    class_counts = np.bincount(y_train)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': xgb_cfg['max_depth'],
        'learning_rate': xgb_cfg['learning_rate'],
        'subsample': xgb_cfg['subsample'],
        'colsample_bytree': xgb_cfg['colsample_bytree'],
        'random_state': args.seed,
        'tree_method': 'hist',
        'scale_pos_weight': class_counts[0] / max(class_counts[1], 1),
    }

    feature_names = [f"cnn_{i}" for i in range(X_cnn_train.shape[1])] + list(tabular_features)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    logger.info("Training XGBoost...")
    xgb_model = xgb.train(
        params, dtrain,
        num_boost_round=xgb_cfg['num_boost_round'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=xgb_cfg['early_stopping_rounds'],
        verbose_eval=50,
    )

    # Evaluate
    for split_name, dmatrix, y_true in [('VAL', dval, y_val), ('TEST', dtest, y_test)]:
        y_proba = xgb_model.predict(dmatrix)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_proba)

        logger.info(f"\n{split_name} Results:")
        logger.info(f"  Accuracy:     {metrics['accuracy']:.1f}%")
        logger.info(f"  Balanced Acc: {metrics['balanced_accuracy']:.1f}%")
        logger.info(f"  Sensitivity:  {metrics['sensitivity']:.1f}%")
        logger.info(f"  Specificity:  {metrics['specificity']:.1f}%")
        logger.info(f"  AUC:          {metrics['auc']:.3f}")
        logger.info(f"  Confusion:    {metrics['confusion_matrix']}")

    # Save
    test_proba = xgb_model.predict(dtest)
    test_pred = (test_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)

    xgb_model.save_model(str(output_dir / 'xgboost_model.json'))
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
