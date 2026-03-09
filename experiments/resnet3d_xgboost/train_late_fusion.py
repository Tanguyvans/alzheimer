#!/usr/bin/env python3
"""
True Late Fusion: ResNet3D (MRI) + XGBoost (Tabular) — separate predictions, combined.

Pipeline:
  1. Fine-tune ResNet3D + Linear head on MRI → proba_mri
  2. Train XGBoost on tabular features only → proba_tab
  3. Late fusion: combine probabilities (average, weighted, learned stacking)

Usage:
    python train_late_fusion.py --config config.yaml
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
from sklearn.linear_model import LogisticRegression
import importlib.util

import xgboost as xgb

# Import ResNet3DBackbone
_resnet_model_path = Path(__file__).parent.parent / "resnet3d_mlp" / "model.py"
_spec_resnet = importlib.util.spec_from_file_location("resnet3d_mlp_model", _resnet_model_path)
_resnet_module = importlib.util.module_from_spec(_spec_resnet)
_spec_resnet.loader.exec_module(_resnet_module)
ResNet3DBackbone = _resnet_module.ResNet3DBackbone

# Import MultiModalDataset
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
    """ResNet3D + linear head for fine-tuning."""
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNet3DBackbone(pretrained=pretrained)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


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


def main():
    parser = argparse.ArgumentParser(description='True Late Fusion: ResNet3D + XGBoost')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='results_late_fusion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--freeze-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1)
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Use frozen pretrained backbone (no fine-tuning)')
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
    tab_scaler = train_dataset.get_scaler()

    val_dataset = MultiModalDataset(
        config['data']['val_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    test_dataset = MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Class weights
    train_labels = pd.read_csv(config['data']['train_csv'])['label'].tolist()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    # ═══════════════════════════════════════════════════════
    # BRANCH 1: ResNet3D on MRI
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("BRANCH 1: ResNet3D on MRI")
    logger.info("=" * 60)

    model = ResNet3DClassifier(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    scaler_amp = torch.amp.GradScaler('cuda', enabled=True)

    if args.skip_finetune:
        logger.info("Using frozen pretrained backbone (--skip-finetune)")
    else:
        # Differential LR
        backbone_params = list(model.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-7)

        if args.freeze_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = False
            logger.info(f"Backbone frozen for first {args.freeze_epochs} epochs")

        best_val_bal_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(args.finetune_epochs):
            if epoch == args.freeze_epochs and args.freeze_epochs > 0:
                for p in model.backbone.parameters():
                    p.requires_grad = True
                logger.info(f"Backbone unfrozen at epoch {epoch+1}")

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
            scheduler.step()

            # Validate
            model.eval()
            val_preds, val_labels_list, val_probs = [], [], []
            with torch.no_grad():
                for mri, tabular, labels in val_loader:
                    mri, labels = mri.to(device), labels.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model(mri)
                    probs = torch.softmax(outputs.float(), dim=1)
                    _, pred = outputs.max(1)
                    val_preds.extend(pred.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())
                    val_probs.extend(probs[:, 1].cpu().numpy())

            val_bal_acc = balanced_accuracy_score(val_labels_list, val_preds) * 100
            val_acc = accuracy_score(val_labels_list, val_preds) * 100
            try:
                val_auc = roc_auc_score(val_labels_list, val_probs)
            except:
                val_auc = 0.5

            frozen_str = " [frozen]" if epoch < args.freeze_epochs else ""
            logger.info(
                f"Epoch {epoch+1}/{args.finetune_epochs}: "
                f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}% "
                f"val_bal_acc={val_bal_acc:.1f}% val_auc={val_auc:.3f}{frozen_str}"
            )

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 15 and epoch >= 10:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"Best val balanced accuracy: {best_val_bal_acc:.1f}%")

    # Get MRI probabilities for all splits
    def get_mri_probas(loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for mri, tabular, labels in loader:
                mri = mri.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(mri)
                probs = torch.softmax(outputs.float(), dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_probs), np.array(all_labels)

    # No-aug train loader for inference
    train_dataset_noaug = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    logger.info("Getting MRI probabilities...")
    proba_mri_train, y_train = get_mri_probas(train_loader_noaug)
    proba_mri_val, y_val = get_mri_probas(val_loader)
    proba_mri_test, y_test = get_mri_probas(test_loader)

    # MRI-only results
    mri_pred = (proba_mri_test >= 0.5).astype(int)
    mri_metrics = compute_metrics(y_test, mri_pred, proba_mri_test)
    log_metrics("BRANCH 1 — MRI only (TEST)", mri_metrics)

    # ═══════════════════════════════════════════════════════
    # BRANCH 2: XGBoost on tabular only
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH 2: XGBoost on Tabular features only")
    logger.info("=" * 60)

    # Extract tabular features
    def get_tabular(loader):
        all_tab, all_labels = [], []
        for mri, tabular, labels in loader:
            all_tab.append(tabular.numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_tab), np.concatenate(all_labels)

    X_tab_train, _ = get_tabular(train_loader_noaug)
    X_tab_val, _ = get_tabular(val_loader)
    X_tab_test, _ = get_tabular(test_loader)

    tab_class_counts = np.bincount(y_train)
    tab_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': xgb_cfg['max_depth'],
        'learning_rate': xgb_cfg['learning_rate'],
        'subsample': xgb_cfg['subsample'],
        'colsample_bytree': xgb_cfg['colsample_bytree'],
        'random_state': args.seed,
        'tree_method': 'hist',
        'scale_pos_weight': tab_class_counts[0] / max(tab_class_counts[1], 1),
    }

    dtrain_tab = xgb.DMatrix(X_tab_train, label=y_train, feature_names=tabular_features)
    dval_tab = xgb.DMatrix(X_tab_val, label=y_val, feature_names=tabular_features)
    dtest_tab = xgb.DMatrix(X_tab_test, label=y_test, feature_names=tabular_features)

    logger.info("Training XGBoost (tabular only)...")
    xgb_tab = xgb.train(
        tab_params, dtrain_tab,
        num_boost_round=xgb_cfg['num_boost_round'],
        evals=[(dtrain_tab, 'train'), (dval_tab, 'val')],
        early_stopping_rounds=xgb_cfg['early_stopping_rounds'],
        verbose_eval=50,
    )

    proba_tab_train = xgb_tab.predict(dtrain_tab)
    proba_tab_val = xgb_tab.predict(dval_tab)
    proba_tab_test = xgb_tab.predict(dtest_tab)

    # Tabular-only results
    tab_pred = (proba_tab_test >= 0.5).astype(int)
    tab_metrics = compute_metrics(y_test, tab_pred, proba_tab_test)
    log_metrics("BRANCH 2 — Tabular only (TEST)", tab_metrics)

    # ═══════════════════════════════════════════════════════
    # LATE FUSION: Combine probabilities
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("LATE FUSION: Combining predictions")
    logger.info("=" * 60)

    all_fusion_results = {}

    # Method 1: Simple average
    proba_avg = (proba_mri_test + proba_tab_test) / 2
    pred_avg = (proba_avg >= 0.5).astype(int)
    avg_metrics = compute_metrics(y_test, pred_avg, proba_avg)
    log_metrics("LATE FUSION — Simple Average (TEST)", avg_metrics)
    all_fusion_results['simple_average'] = avg_metrics

    # Method 2: Weighted average (sweep weights on val set)
    best_w, best_val_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        proba_w = w * proba_mri_val + (1 - w) * proba_tab_val
        try:
            auc_w = roc_auc_score(y_val, proba_w)
        except:
            auc_w = 0.5
        if auc_w > best_val_auc:
            best_val_auc = auc_w
            best_w = w

    logger.info(f"\nBest weight (MRI): {best_w:.2f} (val AUC: {best_val_auc:.3f})")
    proba_weighted = best_w * proba_mri_test + (1 - best_w) * proba_tab_test
    pred_weighted = (proba_weighted >= 0.5).astype(int)
    weighted_metrics = compute_metrics(y_test, pred_weighted, proba_weighted)
    log_metrics(f"LATE FUSION — Weighted Average w_mri={best_w:.2f} (TEST)", weighted_metrics)
    all_fusion_results['weighted_average'] = {**weighted_metrics, 'weight_mri': float(best_w)}

    # Method 3: Learned stacking (logistic regression on val probas)
    X_stack_train = np.column_stack([proba_mri_train, proba_tab_train])
    X_stack_val = np.column_stack([proba_mri_val, proba_tab_val])
    X_stack_test = np.column_stack([proba_mri_test, proba_tab_test])

    # Fit on train, pick best C on val
    best_C, best_stack_val_auc = 1.0, 0.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, random_state=args.seed, max_iter=1000)
        lr.fit(X_stack_train, y_train)
        proba_lr_val = lr.predict_proba(X_stack_val)[:, 1]
        try:
            auc_lr = roc_auc_score(y_val, proba_lr_val)
        except:
            auc_lr = 0.5
        if auc_lr > best_stack_val_auc:
            best_stack_val_auc = auc_lr
            best_C = C

    stacker = LogisticRegression(C=best_C, random_state=args.seed, max_iter=1000)
    stacker.fit(X_stack_train, y_train)
    proba_stack = stacker.predict_proba(X_stack_test)[:, 1]
    pred_stack = (proba_stack >= 0.5).astype(int)
    stack_metrics = compute_metrics(y_test, pred_stack, proba_stack)
    logger.info(f"\nStacking weights: MRI={stacker.coef_[0][0]:.3f}, Tab={stacker.coef_[0][1]:.3f}, bias={stacker.intercept_[0]:.3f}")
    log_metrics(f"LATE FUSION — Learned Stacking C={best_C} (TEST)", stack_metrics)
    all_fusion_results['learned_stacking'] = {
        **stack_metrics,
        'C': best_C,
        'weight_mri': float(stacker.coef_[0][0]),
        'weight_tab': float(stacker.coef_[0][1]),
    }

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Method':<35} {'Acc':>6} {'BalAcc':>7} {'Sens':>6} {'Spec':>6} {'AUC':>6}")
    logger.info("-" * 70)
    for name, m in [
        ('MRI only (ResNet3D)', mri_metrics),
        ('Tabular only (XGBoost)', tab_metrics),
        ('Late Fusion: Average', avg_metrics),
        ('Late Fusion: Weighted', weighted_metrics),
        ('Late Fusion: Stacking', stack_metrics),
    ]:
        logger.info(f"{name:<35} {m['accuracy']:5.1f}% {m['balanced_accuracy']:5.1f}% "
                     f"{m['sensitivity']:5.1f}% {m['specificity']:5.1f}% {m['auc']:5.3f}")

    # Save
    results = {
        'mri_only': mri_metrics,
        'tabular_only': tab_metrics,
        'late_fusion': all_fusion_results,
        'config': {
            'finetune_epochs': args.finetune_epochs,
            'freeze_epochs': args.freeze_epochs,
            'lr': args.lr,
            'skip_finetune': args.skip_finetune,
            'seed': args.seed,
        }
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    xgb_tab.save_model(str(output_dir / 'xgboost_tabular.json'))
    if not args.skip_finetune:
        torch.save(model.state_dict(), output_dir / 'resnet3d_finetuned.pth')

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
