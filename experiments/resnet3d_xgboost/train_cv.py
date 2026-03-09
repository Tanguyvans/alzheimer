#!/usr/bin/env python3
"""
Cross-Validation Training for ResNet3D (frozen) + XGBoost Late Fusion

Extracts CNN features from frozen MedicalNet pretrained ResNet50, concatenates
with raw tabular features, and trains XGBoost classifier.
Same CV methodology as multimodal_fusion/train_cv.py.

Usage:
    python train_cv.py --config config.yaml --n-folds 5 --seeds 42 123 456
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import importlib.util

import xgboost as xgb

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
    """Set random seeds for reproducibility"""
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
    """Compute accuracy, balanced_accuracy, sensitivity, specificity, AUC."""
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
        'predictions': y_pred,
        'labels': y_true,
        'probs': y_proba
    }


def create_fold_datasets(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Dict,
    temp_dir: Path
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """Create dataloaders for a single fold with proper scaler handling"""

    tabular_features = config['data']['tabular_features']
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    batch_size = config['hardware'].get('batch_size', 4)
    num_workers = config['hardware']['num_workers']

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_csv = temp_dir / 'train_fold.csv'
    val_csv = temp_dir / 'val_fold.csv'
    test_csv = temp_dir / 'test_fold.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # No augmentation for feature extraction
    train_dataset = MultiModalDataset(
        str(train_csv),
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
        str(val_csv),
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    test_dataset = MultiModalDataset(
        str(test_csv),
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    train_labels = train_df['label'].tolist()
    return train_loader, val_loader, test_loader, train_labels


def run_cross_validation(config: Dict, n_folds: int, seeds: List[int], output_dir: Path,
                         cn_ad_test_csv: Optional[str] = None, use_wandb: bool = False,
                         max_folds: Optional[int] = None):
    """Run K-fold cross-validation with multiple seeds"""

    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'alzheimer-resnet3d-xgboost'),
            name=wandb_config.get('name', f'resnet3d_xgboost_cv_{n_folds}fold'),
            config=config
        )
        logger.info("WandB initialized")
    else:
        use_wandb = False

    device = setup_device(config)
    xgb_cfg = config['model']['xgboost']

    # Load all data
    logger.info("Loading data...")
    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    test_df = pd.read_csv(config['data']['test_csv'])

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples: {len(all_df)}")
    logger.info(f"Label distribution:\n{all_df['label'].value_counts()}")

    # Load trajectory reference for subgroup analysis
    trajectory_file = config['data'].get('trajectory_csv', '../../data/adni/adni_cn_ad_trajectory.csv')
    try:
        traj_ref = pd.read_csv(trajectory_file)[['subject_id', 'trajectory']].drop_duplicates()
        all_df = all_df.merge(traj_ref, on='subject_id', how='left')
        if 'trajectory' in all_df.columns:
            all_df['trajectory'] = all_df['trajectory'].fillna(all_df.get('DX', 'Unknown'))
        logger.info(f"Trajectory distribution:\n{all_df['trajectory'].value_counts()}")
    except Exception as e:
        logger.warning(f"Could not load trajectory file: {e}. Subgroup analysis disabled.")
        all_df['trajectory'] = all_df.get('DX', 'Unknown')

    # Load CN_AD stable subject_ids for cross-task evaluation
    stable_subject_ids = None
    if cn_ad_test_csv:
        cn_ad_dir = Path(cn_ad_test_csv)
        if cn_ad_dir.is_dir():
            cn_ad_parts = []
            for csv_name in ['train.csv', 'val.csv', 'test.csv']:
                csv_path = cn_ad_dir / csv_name
                if csv_path.exists():
                    cn_ad_parts.append(pd.read_csv(csv_path))
            if cn_ad_parts:
                cn_ad_full = pd.concat(cn_ad_parts, ignore_index=True)
                stable_subject_ids = set(cn_ad_full['subject_id'].values)
                logger.info(f"Loaded {len(stable_subject_ids)} stable subject_ids for CN_AD filtering")

    all_labels = all_df['label'].values
    all_indices = np.arange(len(all_df))

    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen backbone once
    logger.info("Loading frozen ResNet50 3D backbone (MedicalNet pretrained)...")
    resnet_cfg = config['model']['resnet']
    backbone = ResNet3DBackbone(pretrained=resnet_cfg.get('pretrained', True)).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    total_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"ResNet3D backbone params: {total_params:,} (all frozen)")

    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED: {seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_results = []

        effective_folds = max_folds if max_folds else n_folds
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_indices, all_labels)):
            if fold >= effective_folds:
                break
            logger.info(f"\n--- Fold {fold+1}/{effective_folds} ---")

            # Split train into train/val (90/10)
            set_seed(seed + fold)
            n_train = len(train_idx)
            val_size = int(0.1 * n_train)
            perm = np.random.permutation(n_train)
            val_idx_local = train_idx[perm[:val_size]]
            train_idx_final = train_idx[perm[val_size:]]

            logger.info(f"Train: {len(train_idx_final)}, Val: {len(val_idx_local)}, Test: {len(test_idx)}")

            # Create fold dataloaders
            train_loader, val_loader, test_loader, train_labels = create_fold_datasets(
                all_df, train_idx_final, val_idx_local, test_idx, config, temp_dir
            )

            # Extract CNN features
            logger.info("  Extracting CNN features...")
            X_cnn_train, X_tab_train, y_train = extract_features(backbone, train_loader, device)
            X_cnn_val, X_tab_val, y_val = extract_features(backbone, val_loader, device)
            X_cnn_test, X_tab_test, y_test = extract_features(backbone, test_loader, device)

            # Concat CNN + raw tabular
            X_train = np.hstack([X_cnn_train, X_tab_train])
            X_val = np.hstack([X_cnn_val, X_tab_val])
            X_test = np.hstack([X_cnn_test, X_tab_test])

            logger.info(f"  Feature dims: CNN={X_cnn_train.shape[1]}, Tab={X_tab_train.shape[1]}, Total={X_train.shape[1]}")

            # StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # XGBoost
            class_counts = np.bincount(y_train)
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': xgb_cfg['max_depth'],
                'learning_rate': xgb_cfg['learning_rate'],
                'subsample': xgb_cfg['subsample'],
                'colsample_bytree': xgb_cfg['colsample_bytree'],
                'random_state': seed,
                'tree_method': 'hist',
                'scale_pos_weight': class_counts[0] / max(class_counts[1], 1),
            }

            tabular_features = config['data']['tabular_features']
            feature_names = [f"cnn_{i}" for i in range(X_cnn_train.shape[1])] + list(tabular_features)
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

            xgb_model = xgb.train(
                params, dtrain,
                num_boost_round=xgb_cfg['num_boost_round'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=xgb_cfg['early_stopping_rounds'],
                verbose_eval=50,
            )

            # Predict
            y_proba = xgb_model.predict(dtest)
            y_pred = (y_proba >= 0.5).astype(int)

            test_metrics = compute_metrics(y_test, y_pred, y_proba)

            logger.info(
                f"  Test: Acc={test_metrics['accuracy']:.1f}%, "
                f"BalAcc={test_metrics['balanced_accuracy']:.1f}%, "
                f"Sens={test_metrics['sensitivity']:.1f}%, "
                f"Spec={test_metrics['specificity']:.1f}%, "
                f"AUC={test_metrics['auc']:.3f}"
            )

            # Subgroup analysis
            test_df_fold = all_df.iloc[test_idx].reset_index(drop=True)
            test_df_fold['prediction'] = y_pred
            test_df_fold['correct'] = (test_df_fold['prediction'] == test_df_fold['label'])

            subgroup_accs = {}
            for traj in test_df_fold['trajectory'].unique():
                subset = test_df_fold[test_df_fold['trajectory'] == traj]
                if len(subset) > 0:
                    acc = subset['correct'].mean() * 100
                    subgroup_accs[traj] = {'accuracy': acc, 'n_samples': len(subset)}

            logger.info("  Subgroup Analysis:")
            for traj, metrics in sorted(subgroup_accs.items()):
                logger.info(f"    {traj}: {metrics['accuracy']:.1f}% ({metrics['n_samples']} samples)")

            # Cross-task CN_AD evaluation
            cn_ad_metrics = None
            if stable_subject_ids is not None:
                cn_ad_mask = test_df_fold['subject_id'].isin(stable_subject_ids)
                cn_ad_test_fold_df = test_df_fold[cn_ad_mask]

                if len(cn_ad_test_fold_df) > 0:
                    cn_ad_indices = cn_ad_mask.values.nonzero()[0]
                    cn_ad_y_true = y_test[cn_ad_indices]
                    cn_ad_y_pred = y_pred[cn_ad_indices]
                    cn_ad_y_proba = y_proba[cn_ad_indices]

                    cn_ad_metrics = compute_metrics(cn_ad_y_true, cn_ad_y_pred, cn_ad_y_proba)
                    logger.info(
                        f"  CN_AD ({len(cn_ad_test_fold_df)} stable): "
                        f"Acc={cn_ad_metrics['accuracy']:.1f}%, "
                        f"BalAcc={cn_ad_metrics['balanced_accuracy']:.1f}%, "
                        f"Sens={cn_ad_metrics['sensitivity']:.1f}%, "
                        f"Spec={cn_ad_metrics['specificity']:.1f}%, "
                        f"AUC={cn_ad_metrics['auc']:.3f}"
                    )

            fold_result = {
                'seed': seed,
                'fold': fold,
                'traj_accuracy': test_metrics['accuracy'],
                'traj_balanced_accuracy': test_metrics['balanced_accuracy'],
                'traj_sensitivity': test_metrics['sensitivity'],
                'traj_specificity': test_metrics['specificity'],
                'traj_auc': test_metrics['auc'],
                'subgroup_accuracy': subgroup_accs
            }

            if cn_ad_metrics is not None:
                fold_result['cn_ad_accuracy'] = cn_ad_metrics['accuracy']
                fold_result['cn_ad_balanced_accuracy'] = cn_ad_metrics['balanced_accuracy']
                fold_result['cn_ad_sensitivity'] = cn_ad_metrics['sensitivity']
                fold_result['cn_ad_specificity'] = cn_ad_metrics['specificity']
                fold_result['cn_ad_auc'] = cn_ad_metrics['auc']

            fold_results.append(fold_result)
            all_results.append(fold_result)

            # WandB logging
            if use_wandb:
                wandb.log({
                    f'test/seed_{seed}_fold_{fold}_traj_accuracy': test_metrics['accuracy'],
                    f'test/seed_{seed}_fold_{fold}_traj_balanced_accuracy': test_metrics['balanced_accuracy'],
                    f'test/seed_{seed}_fold_{fold}_traj_sensitivity': test_metrics['sensitivity'],
                    f'test/seed_{seed}_fold_{fold}_traj_specificity': test_metrics['specificity'],
                    f'test/seed_{seed}_fold_{fold}_traj_auc': test_metrics['auc']
                })
                if cn_ad_metrics is not None:
                    wandb.log({
                        f'test/seed_{seed}_fold_{fold}_cn_ad_accuracy': cn_ad_metrics['accuracy'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_balanced_accuracy': cn_ad_metrics['balanced_accuracy'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_sensitivity': cn_ad_metrics['sensitivity'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_specificity': cn_ad_metrics['specificity'],
                        f'test/seed_{seed}_fold_{fold}_cn_ad_auc': cn_ad_metrics['auc']
                    })

            # Save XGBoost model
            fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            xgb_model.save_model(str(fold_dir / 'xgboost_model.json'))

        # Seed summary
        test_accs = [r['traj_accuracy'] for r in fold_results]
        balanced_accs = [r['traj_balanced_accuracy'] for r in fold_results]
        sensitivities = [r['traj_sensitivity'] for r in fold_results]
        specificities = [r['traj_specificity'] for r in fold_results]
        aucs = [r['traj_auc'] for r in fold_results]

        logger.info(f"\nSeed {seed} Trajectory Summary:")
        logger.info(f"  Acc: {np.mean(test_accs):.1f}+/-{np.std(test_accs):.1f}%")
        logger.info(f"  BalAcc: {np.mean(balanced_accs):.1f}+/-{np.std(balanced_accs):.1f}%")
        logger.info(f"  Sens: {np.mean(sensitivities):.1f}+/-{np.std(sensitivities):.1f}%")
        logger.info(f"  Spec: {np.mean(specificities):.1f}+/-{np.std(specificities):.1f}%")
        logger.info(f"  AUC: {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}")

        if stable_subject_ids is not None:
            cn_ad_accs = [r['cn_ad_accuracy'] for r in fold_results if 'cn_ad_accuracy' in r]
            if cn_ad_accs:
                cn_ad_bal_accs = [r['cn_ad_balanced_accuracy'] for r in fold_results if 'cn_ad_balanced_accuracy' in r]
                cn_ad_sens = [r['cn_ad_sensitivity'] for r in fold_results if 'cn_ad_sensitivity' in r]
                cn_ad_spec = [r['cn_ad_specificity'] for r in fold_results if 'cn_ad_specificity' in r]
                cn_ad_aucs = [r['cn_ad_auc'] for r in fold_results if 'cn_ad_auc' in r]

                logger.info(f"Seed {seed} CN_AD Summary:")
                logger.info(f"  Acc: {np.mean(cn_ad_accs):.1f}+/-{np.std(cn_ad_accs):.1f}%")
                logger.info(f"  BalAcc: {np.mean(cn_ad_bal_accs):.1f}+/-{np.std(cn_ad_bal_accs):.1f}%")
                logger.info(f"  Sens: {np.mean(cn_ad_sens):.1f}+/-{np.std(cn_ad_sens):.1f}%")
                logger.info(f"  Spec: {np.mean(cn_ad_spec):.1f}+/-{np.std(cn_ad_spec):.1f}%")
                logger.info(f"  AUC: {np.mean(cn_ad_aucs):.3f}+/-{np.std(cn_ad_aucs):.3f}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS (ResNet3D + XGBoost)")
    logger.info(f"{'='*60}")

    results_df = pd.DataFrame(all_results)

    traj_acc_mean = results_df['traj_accuracy'].mean()
    traj_acc_std = results_df['traj_accuracy'].std()
    traj_bal_mean = results_df['traj_balanced_accuracy'].mean()
    traj_bal_std = results_df['traj_balanced_accuracy'].std()
    traj_sens_mean = results_df['traj_sensitivity'].mean()
    traj_sens_std = results_df['traj_sensitivity'].std()
    traj_spec_mean = results_df['traj_specificity'].mean()
    traj_spec_std = results_df['traj_specificity'].std()
    traj_auc_mean = results_df['traj_auc'].mean()
    traj_auc_std = results_df['traj_auc'].std()

    logger.info(f"\nCN_AD_TRAJECTORY (train+test):")
    logger.info(f"  Accuracy:     {traj_acc_mean:.1f}+/-{traj_acc_std:.1f}%")
    logger.info(f"  Balanced Acc: {traj_bal_mean:.1f}+/-{traj_bal_std:.1f}%")
    logger.info(f"  Sensitivity:  {traj_sens_mean:.1f}+/-{traj_sens_std:.1f}%")
    logger.info(f"  Specificity:  {traj_spec_mean:.1f}+/-{traj_spec_std:.1f}%")
    logger.info(f"  AUC:          {traj_auc_mean:.3f}+/-{traj_auc_std:.3f}")

    cn_ad_acc_mean, cn_ad_acc_std = None, None
    cn_ad_bal_mean, cn_ad_bal_std = None, None
    cn_ad_sens_mean, cn_ad_sens_std = None, None
    cn_ad_spec_mean, cn_ad_spec_std = None, None
    cn_ad_auc_mean, cn_ad_auc_std = None, None

    if 'cn_ad_accuracy' in results_df.columns:
        cn_ad_acc_mean = results_df['cn_ad_accuracy'].mean()
        cn_ad_acc_std = results_df['cn_ad_accuracy'].std()
        cn_ad_bal_mean = results_df['cn_ad_balanced_accuracy'].mean()
        cn_ad_bal_std = results_df['cn_ad_balanced_accuracy'].std()
        cn_ad_sens_mean = results_df['cn_ad_sensitivity'].mean()
        cn_ad_sens_std = results_df['cn_ad_sensitivity'].std()
        cn_ad_spec_mean = results_df['cn_ad_specificity'].mean()
        cn_ad_spec_std = results_df['cn_ad_specificity'].std()
        cn_ad_auc_mean = results_df['cn_ad_auc'].mean()
        cn_ad_auc_std = results_df['cn_ad_auc'].std()

        logger.info(f"\nCN_AD (stable AD, cross-task):")
        logger.info(f"  Accuracy:     {cn_ad_acc_mean:.1f}+/-{cn_ad_acc_std:.1f}%")
        logger.info(f"  Balanced Acc: {cn_ad_bal_mean:.1f}+/-{cn_ad_bal_std:.1f}%")
        logger.info(f"  Sensitivity:  {cn_ad_sens_mean:.1f}+/-{cn_ad_sens_std:.1f}%")
        logger.info(f"  Specificity:  {cn_ad_spec_mean:.1f}+/-{cn_ad_spec_std:.1f}%")
        logger.info(f"  AUC:          {cn_ad_auc_mean:.3f}+/-{cn_ad_auc_std:.3f}")

    # WandB final summary
    if use_wandb:
        wandb.log({
            'final/traj_accuracy_mean': traj_acc_mean,
            'final/traj_accuracy_std': traj_acc_std,
            'final/traj_balanced_accuracy_mean': traj_bal_mean,
            'final/traj_balanced_accuracy_std': traj_bal_std,
            'final/traj_sensitivity_mean': traj_sens_mean,
            'final/traj_sensitivity_std': traj_sens_std,
            'final/traj_specificity_mean': traj_spec_mean,
            'final/traj_specificity_std': traj_spec_std,
            'final/traj_auc_mean': traj_auc_mean,
            'final/traj_auc_std': traj_auc_std
        })
        if cn_ad_acc_mean is not None:
            wandb.log({
                'final/cn_ad_accuracy_mean': cn_ad_acc_mean,
                'final/cn_ad_accuracy_std': cn_ad_acc_std,
                'final/cn_ad_balanced_accuracy_mean': cn_ad_bal_mean,
                'final/cn_ad_balanced_accuracy_std': cn_ad_bal_std,
                'final/cn_ad_sensitivity_mean': cn_ad_sens_mean,
                'final/cn_ad_sensitivity_std': cn_ad_sens_std,
                'final/cn_ad_specificity_mean': cn_ad_spec_mean,
                'final/cn_ad_specificity_std': cn_ad_spec_std,
                'final/cn_ad_auc_mean': cn_ad_auc_mean,
                'final/cn_ad_auc_std': cn_ad_auc_std
            })
        wandb.finish()

    # Subgroup summary
    logger.info("\nSUBGROUP ANALYSIS SUMMARY:")
    logger.info("=" * 40)
    subgroup_totals = {}
    for result in all_results:
        for traj, metrics in result.get('subgroup_accuracy', {}).items():
            if traj not in subgroup_totals:
                subgroup_totals[traj] = {'accuracies': [], 'n_samples': []}
            subgroup_totals[traj]['accuracies'].append(metrics['accuracy'])
            subgroup_totals[traj]['n_samples'].append(metrics['n_samples'])

    for traj in sorted(subgroup_totals.keys()):
        accs = subgroup_totals[traj]['accuracies']
        n_samples = subgroup_totals[traj]['n_samples']
        logger.info(f"  {traj}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}% (avg {np.mean(n_samples):.0f} samples/fold)")
    logger.info("=" * 40)

    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    subgroup_summary = {}
    for traj in sorted(subgroup_totals.keys()):
        accs = subgroup_totals[traj]['accuracies']
        subgroup_summary[traj] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'avg_samples_per_fold': float(np.mean(subgroup_totals[traj]['n_samples']))
        }

    summary = {
        'experiment': 'resnet3d_xgboost_late_fusion',
        'n_folds': n_folds,
        'seeds': seeds,
        'traj_accuracy': float(traj_acc_mean),
        'traj_accuracy_std': float(traj_acc_std),
        'traj_balanced_accuracy': float(traj_bal_mean),
        'traj_balanced_accuracy_std': float(traj_bal_std),
        'traj_sensitivity': float(traj_sens_mean),
        'traj_sensitivity_std': float(traj_sens_std),
        'traj_specificity': float(traj_spec_mean),
        'traj_specificity_std': float(traj_spec_std),
        'traj_auc': float(traj_auc_mean),
        'traj_auc_std': float(traj_auc_std),
        'subgroup_summary': subgroup_summary,
        'per_fold_results': all_results
    }

    if cn_ad_acc_mean is not None:
        summary.update({
            'cn_ad_accuracy': float(cn_ad_acc_mean),
            'cn_ad_accuracy_std': float(cn_ad_acc_std),
            'cn_ad_balanced_accuracy': float(cn_ad_bal_mean),
            'cn_ad_balanced_accuracy_std': float(cn_ad_bal_std),
            'cn_ad_sensitivity': float(cn_ad_sens_mean),
            'cn_ad_sensitivity_std': float(cn_ad_sens_std),
            'cn_ad_specificity': float(cn_ad_spec_mean),
            'cn_ad_specificity_std': float(cn_ad_spec_std),
            'cn_ad_auc': float(cn_ad_auc_mean),
            'cn_ad_auc_std': float(cn_ad_auc_std),
        })

    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup temp directory
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation for ResNet3D (frozen) + XGBoost Late Fusion')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='cv_results', help='Output directory')
    parser.add_argument('--cn-ad-test', type=str, default='../multimodal_fusion/data/combined_cn_ad',
                        help='Path to CN_AD data directory for cross-task evaluation')
    parser.add_argument('--max-folds', type=int, default=None, help='Max folds to run (default: all)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_wandb = config.get('wandb', {}).get('enabled', False) and not args.no_wandb

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_cross_validation(
        config, args.n_folds, args.seeds, output_dir,
        cn_ad_test_csv=args.cn_ad_test, use_wandb=use_wandb,
        max_folds=args.max_folds
    )

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
