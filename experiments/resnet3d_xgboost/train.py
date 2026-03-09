#!/usr/bin/env python3
"""
ResNet3D (frozen) + XGBoost — single train/val/test run.

Usage:
    python train.py --config config.yaml
"""

import yaml
import torch
import torch.nn as nn
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


def setup_device(config: Dict) -> torch.device:
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device):
    """Extract CNN features from frozen backbone + raw tabular features."""
    all_cnn, all_tab, all_labels = [], [], []
    backbone.eval()
    with torch.no_grad():
        for mri, tabular, labels in tqdm(loader, desc="  Extracting features", leave=False):
            mri = mri.to(device)
            cnn_feat = backbone(mri)
            all_cnn.append(cnn_feat.cpu().numpy())
            all_tab.append(tabular.numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_cnn),
        np.concatenate(all_tab),
        np.concatenate(all_labels),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
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
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'auc': float(auc),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='ResNet3D (frozen) + XGBoost — single run')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(config)
    xgb_cfg = config['model']['xgboost']
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']
    batch_size = config['hardware'].get('batch_size', 4)
    num_workers = config['hardware']['num_workers']

    # Load frozen backbone
    logger.info("Loading frozen ResNet50 3D backbone (MedicalNet pretrained)...")
    backbone = ResNet3DBackbone(pretrained=config['model']['resnet'].get('pretrained', True)).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    logger.info(f"ResNet3D params: {sum(p.numel() for p in backbone.parameters()):,} (all frozen)")

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MultiModalDataset(
        config['data']['train_csv'],
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=None,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    val_dataset = MultiModalDataset(
        config['data']['val_csv'],
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    test_dataset = MultiModalDataset(
        config['data']['test_csv'],
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Extract features
    logger.info("Extracting CNN features...")
    X_cnn_train, X_tab_train, y_train = extract_features(backbone, train_loader, device)
    X_cnn_val, X_tab_val, y_val = extract_features(backbone, val_loader, device)
    X_cnn_test, X_tab_test, y_test = extract_features(backbone, test_loader, device)

    # Concat CNN + tabular
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
    for split_name, dmatrix, y_true in [('val', dval, y_val), ('test', dtest, y_test)]:
        y_proba = xgb_model.predict(dmatrix)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_proba)

        logger.info(f"\n{split_name.upper()} Results:")
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
