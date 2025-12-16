#!/usr/bin/env python3
"""
Unified XGBoost trainer for tabular Alzheimer's classification.

Supports multiple tasks and datasets via YAML config files.

Usage:
    python train.py --config configs/cn_ad_adni.yaml
    python train.py --config configs/cn_mci_ad_oasis.yaml
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, label_binarize
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task definitions
TASKS = {
    'cn_ad': {
        'classes': ['CN', 'AD'],
        'num_classes': 2,
        'objective': 'binary:logistic',
    },
    'cn_mci_ad': {
        'classes': ['CN', 'MCI', 'AD'],
        'num_classes': 3,
        'objective': 'multi:softprob',
    },
    'cn_mcis_mcic_ad': {
        'classes': ['CN', 'MCI_stable', 'MCI_converter', 'AD'],
        'num_classes': 4,
        'objective': 'multi:softprob',
    },
}

# Default features (can be overridden in config)
DEFAULT_FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    'VSWEIGHT', 'BMI',
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    'BCDEPRES',
]


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('seed', 42)
    config.setdefault('train_ratio', 0.7)
    config.setdefault('val_ratio', 0.15)
    config.setdefault('test_ratio', 0.15)
    config.setdefault('first_visit_only', True)
    config.setdefault('features', DEFAULT_FEATURES)
    config.setdefault('xgboost', {})
    config['xgboost'].setdefault('max_depth', 6)
    config['xgboost'].setdefault('learning_rate', 0.1)
    config['xgboost'].setdefault('subsample', 0.8)
    config['xgboost'].setdefault('colsample_bytree', 0.8)
    config['xgboost'].setdefault('num_boost_round', 200)
    config['xgboost'].setdefault('early_stopping_rounds', 20)

    return config


def prepare_data_adni(df: pd.DataFrame, task: str, features: list, first_visit_only: bool = True):
    """Prepare ADNI data"""
    df_prep = df.copy()
    task_info = TASKS[task]
    class_names = task_info['classes']
    num_classes = task_info['num_classes']

    # Take only first visit (baseline) per subject
    if first_visit_only and 'VISCODE' in df_prep.columns:
        before = len(df_prep)
        # Prefer baseline (bl), then screening (sc)
        df_prep = df_prep[df_prep['VISCODE'] == 'bl']
        if len(df_prep) == 0:
            df_prep = df.copy()
            df_prep = df_prep.sort_values('VISCODE').groupby('Subject').first().reset_index()
        logger.info(f"ADNI first visit only: {before:,} -> {len(df_prep):,} samples")

    # Calculate AGE
    if 'AGE' not in df_prep.columns:
        if 'EXAMDATE' in df_prep.columns:
            df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
            df_prep['AGE'] = df_prep['EXAMDATE'].dt.year - df_prep['PTDOBYY']
        elif 'Acq Date' in df_prep.columns:
            df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
            df_prep['AGE'] = df_prep['Acq Date'].dt.year - df_prep['PTDOBYY']
        else:
            df_prep['AGE'] = 2010 - df_prep['PTDOBYY']

    # Calculate BMI
    if 'BMI' not in df_prep.columns and 'VSWEIGHT' in df_prep.columns and 'VSHEIGHT' in df_prep.columns:
        df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Assign labels based on task
    if task == 'cn_ad':
        if 'CLASS_4' in df_prep.columns:
            df_prep = df_prep[df_prep['CLASS_4'].isin(['CN', 'AD'])].copy()
            df_prep['label'] = (df_prep['CLASS_4'] == 'AD').astype(int)
        elif 'Group' in df_prep.columns:
            df_prep = df_prep[df_prep['Group'].isin(['CN', 'AD'])].copy()
            df_prep['label'] = (df_prep['Group'] == 'AD').astype(int)
        elif 'DX' in df_prep.columns:
            df_prep = df_prep[df_prep['DX'].isin(['CN', 'AD'])].copy()
            df_prep['label'] = (df_prep['DX'] == 'AD').astype(int)

    elif task == 'cn_mci_ad':
        mapping = {'CN': 0, 'MCI': 1, 'AD': 2}
        if 'CLASS_4' in df_prep.columns:
            def map_label(c4):
                if c4 == 'CN': return 0
                elif c4 in ['MCI_stable', 'MCI_to_AD']: return 1
                elif c4 == 'AD': return 2
                return -1
            df_prep['label'] = df_prep['CLASS_4'].apply(map_label)
        elif 'Group' in df_prep.columns:
            df_prep['label'] = df_prep['Group'].map(mapping)
        elif 'DX' in df_prep.columns:
            df_prep['label'] = df_prep['DX'].map(mapping)
        df_prep = df_prep[df_prep['label'] >= 0].copy()

    elif task == 'cn_mcis_mcic_ad':
        if 'CLASS_4' in df_prep.columns:
            mapping = {'CN': 0, 'MCI_stable': 1, 'MCI_to_AD': 2, 'AD': 3}
            df_prep['label'] = df_prep['CLASS_4'].map(mapping)
        df_prep = df_prep[df_prep['label'].notna()].copy()
        df_prep['label'] = df_prep['label'].astype(int)

    return _finalize_data(df_prep, features, class_names)


def prepare_data_oasis(df: pd.DataFrame, task: str, features: list, first_visit_only: bool = True):
    """Prepare OASIS data"""
    df_prep = df.copy()
    task_info = TASKS[task]
    class_names = task_info['classes']

    # Take only first visit per subject
    if first_visit_only and 'Subject' in df_prep.columns:
        before = len(df_prep)
        if 'days_to_visit' in df_prep.columns:
            # Sort by days_to_visit and take first per subject
            df_prep = df_prep.sort_values(['Subject', 'days_to_visit']).groupby('Subject').first().reset_index()
        else:
            # Take first occurrence per subject
            df_prep = df_prep.groupby('Subject').first().reset_index()
        logger.info(f"OASIS first visit only: {before:,} -> {len(df_prep):,} samples")

    # OASIS already has AGE and DX columns from build_oasis_dataset.py

    # Calculate BMI if not present
    if 'BMI' not in df_prep.columns and 'VSWEIGHT' in df_prep.columns and 'VSHEIGHT' in df_prep.columns:
        df_prep['BMI'] = (df_prep['VSWEIGHT'] / (df_prep['VSHEIGHT'] ** 2)) * 703  # lbs/in^2 formula

    # Assign labels based on task
    if task == 'cn_ad':
        df_prep = df_prep[df_prep['DX'].isin(['CN', 'AD'])].copy()
        df_prep['label'] = (df_prep['DX'] == 'AD').astype(int)

    elif task == 'cn_mci_ad':
        mapping = {'CN': 0, 'MCI': 1, 'AD': 2}
        df_prep = df_prep[df_prep['DX'].isin(['CN', 'MCI', 'AD'])].copy()
        df_prep['label'] = df_prep['DX'].map(mapping)

    elif task == 'cn_mcis_mcic_ad':
        logger.warning("OASIS doesn't have MCI stable/converter distinction")
        return None, None

    return _finalize_data(df_prep, features, class_names)


def prepare_data_nacc(df: pd.DataFrame, task: str, features: list):
    """Prepare NACC data (from build_nacc_dataset.py output)"""
    df_prep = df.copy()
    task_info = TASKS[task]
    class_names = task_info['classes']

    # NACC data from build_nacc_dataset.py already has:
    # - AGE (mapped from NACCAGE)
    # - DX (mapped from NACCUDSD: CN, MCI, AD)
    # - All features mapped to ADNI names

    # Assign labels based on task
    if task == 'cn_ad':
        df_prep = df_prep[df_prep['DX'].isin(['CN', 'AD'])].copy()
        df_prep['label'] = (df_prep['DX'] == 'AD').astype(int)

    elif task == 'cn_mci_ad':
        mapping = {'CN': 0, 'MCI': 1, 'AD': 2}
        df_prep = df_prep[df_prep['DX'].isin(['CN', 'MCI', 'AD'])].copy()
        df_prep['label'] = df_prep['DX'].map(mapping)

    elif task == 'cn_mcis_mcic_ad':
        logger.warning("NACC doesn't have MCI stable/converter distinction")
        return None, None

    return _finalize_data(df_prep, features, class_names)


def prepare_data_combined(adni_csv: str, oasis_csv: str, task: str, features: list, nacc_csv: str = None, first_visit_only: bool = True):
    """Prepare combined ADNI + OASIS + NACC data"""
    task_info = TASKS[task]
    class_names = task_info['classes']

    all_dfs = []
    all_features = []

    # Prepare ADNI data
    if adni_csv and Path(adni_csv).exists():
        df_adni = pd.read_csv(adni_csv)
        logger.info(f"Loaded {len(df_adni)} ADNI samples")
        df_adni_prep, adni_features = prepare_data_adni(df_adni, task, features, first_visit_only)
        if df_adni_prep is not None:
            df_adni_prep['source'] = 'ADNI'
            df_adni_prep['Subject'] = 'ADNI_' + df_adni_prep['Subject'].astype(str)
            all_dfs.append((df_adni_prep, adni_features))
            all_features.append(set(adni_features))

    # Prepare OASIS data
    if oasis_csv and Path(oasis_csv).exists():
        df_oasis = pd.read_csv(oasis_csv)
        logger.info(f"Loaded {len(df_oasis)} OASIS samples")
        df_oasis_prep, oasis_features = prepare_data_oasis(df_oasis, task, features, first_visit_only)
        if df_oasis_prep is not None:
            df_oasis_prep['source'] = 'OASIS'
            df_oasis_prep['Subject'] = 'OASIS_' + df_oasis_prep['Subject'].astype(str)
            all_dfs.append((df_oasis_prep, oasis_features))
            all_features.append(set(oasis_features))

    # Prepare NACC data (already first_visit_only from build_nacc_dataset.py)
    if nacc_csv and Path(nacc_csv).exists():
        df_nacc = pd.read_csv(nacc_csv)
        logger.info(f"Loaded {len(df_nacc)} NACC samples")
        df_nacc_prep, nacc_features = prepare_data_nacc(df_nacc, task, features)
        if df_nacc_prep is not None:
            df_nacc_prep['source'] = 'NACC'
            df_nacc_prep['Subject'] = 'NACC_' + df_nacc_prep['Subject'].astype(str)
            all_dfs.append((df_nacc_prep, nacc_features))
            all_features.append(set(nacc_features))

    if not all_dfs:
        return None, None

    # Find common features across all datasets
    common_features = list(set.intersection(*all_features)) if len(all_features) > 1 else list(all_features[0])
    logger.info(f"Common features across {len(all_dfs)} datasets: {len(common_features)}")

    # Combine datasets
    combined_dfs = []
    for df_prep, _ in all_dfs:
        combined_dfs.append(df_prep[common_features + ['label', 'Subject', 'source']])

    df_combined = pd.concat(combined_dfs, ignore_index=True)

    logger.info(f"Combined dataset: {len(df_combined)} samples")
    for source in df_combined['source'].unique():
        count = (df_combined['source'] == source).sum()
        logger.info(f"  {source}: {count} samples")
    for i, name in enumerate(class_names):
        count = sum(df_combined['label'] == i)
        logger.info(f"  Class {i} ({name}): {count} samples")

    return df_combined, common_features


def _finalize_data(df_prep, features, class_names):
    """Finalize data preparation"""
    # Get available features
    available_features = [f for f in features if f in df_prep.columns]
    missing_features = [f for f in features if f not in df_prep.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    # Get subject column
    subject_col = None
    for col in ['Subject', 'PTID', 'OASISID', 'NACCID']:
        if col in df_prep.columns:
            subject_col = col
            break

    if subject_col is None:
        df_prep['Subject'] = range(len(df_prep))
        subject_col = 'Subject'

    df_features = df_prep[available_features + ['label', subject_col]].copy()
    df_features = df_features.rename(columns={subject_col: 'Subject'})

    # Impute missing values
    for col in available_features:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)

    logger.info(f"Prepared {len(df_features)} samples with {len(available_features)} features")
    for i, name in enumerate(class_names):
        count = sum(df_features['label'] == i)
        logger.info(f"  Class {i} ({name}): {count} samples")

    return df_features, available_features


def patient_level_split(df, num_classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data at patient level"""
    patient_classes = {}
    for label in range(num_classes):
        patient_classes[label] = df[df['label'] == label]['Subject'].unique()
        logger.info(f"Unique patients - Class {label}: {len(patient_classes[label])}")

    def split_patients(patients):
        if len(patients) < 3:
            return patients, np.array([]), np.array([])
        train_p, temp_p = train_test_split(patients, test_size=(val_ratio + test_ratio), random_state=seed)
        if len(temp_p) < 2:
            return train_p, temp_p, np.array([])
        val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        return train_p, val_p, test_p

    train_patients, val_patients, test_patients = [], [], []
    for label in range(num_classes):
        train_p, val_p, test_p = split_patients(patient_classes[label])
        train_patients.extend(train_p)
        val_patients.extend(val_p)
        test_patients.extend(test_p)

    train_df = df[df['Subject'].isin(train_patients)]
    val_df = df[df['Subject'].isin(val_patients)]
    test_df = df[df['Subject'].isin(test_patients)]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def train_xgboost(X_train, y_train, X_val, y_val, config, task_info, feature_names):
    """Train XGBoost model"""
    xgb_config = config['xgboost']
    num_classes = task_info['num_classes']
    objective = task_info['objective']

    params = {
        'objective': objective,
        'eval_metric': 'logloss' if num_classes == 2 else 'mlogloss',
        'max_depth': xgb_config['max_depth'],
        'learning_rate': xgb_config['learning_rate'],
        'subsample': xgb_config['subsample'],
        'colsample_bytree': xgb_config['colsample_bytree'],
        'random_state': config['seed'],
        'tree_method': 'hist',
    }

    if num_classes > 2:
        params['num_class'] = num_classes

    # Class weights for imbalanced data
    if num_classes == 2:
        class_counts = np.bincount(y_train)
        params['scale_pos_weight'] = class_counts[0] / class_counts[1]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params, dtrain,
        num_boost_round=xgb_config['num_boost_round'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=xgb_config['early_stopping_rounds'],
        verbose_eval=10
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X, y, task_info, feature_names, dataset_name='Test'):
    """Evaluate model"""
    num_classes = task_info['num_classes']
    class_names = task_info['classes']

    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    y_proba = model.predict(dmatrix)

    if num_classes == 2:
        y_pred = (y_proba >= 0.5).astype(int)
        y_proba_full = np.column_stack([1 - y_proba, y_proba])
    else:
        y_pred = np.argmax(y_proba, axis=1)
        y_proba_full = y_proba

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'classification_report': classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    }

    # AUC-ROC
    try:
        if num_classes == 2:
            metrics['auc_roc'] = roc_auc_score(y, y_proba)
        else:
            y_bin = label_binarize(y, classes=list(range(num_classes)))
            metrics['auc_roc'] = roc_auc_score(y_bin, y_proba_full, average='macro', multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        metrics['auc_roc'] = None

    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name.upper()} RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    if metrics['auc_roc']:
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=class_names, zero_division=0)}")

    return metrics, y_pred, y_proba_full


def plot_results(y_true, y_pred, y_proba, metrics, task_info, output_dir, model=None):
    """Generate plots"""
    output_dir = Path(output_dir)
    class_names = task_info['classes']
    num_classes = task_info['num_classes']

    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {" | ".join(class_names)}')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Normalized Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC Curve
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Feature Importance
    if model is not None:
        importance = model.get_score(importance_type='gain')
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_n = min(15, len(sorted_imp))
            features = [x[0] for x in sorted_imp[:top_n]][::-1]
            values = [x[1] for x in sorted_imp[:top_n]][::-1]

            plt.figure(figsize=(10, 8))
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
            plt.barh(features, values, color=colors)
            plt.xlabel('Importance (Gain)')
            plt.title('Top Feature Importance')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Unified XGBoost trainer')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    task = config['task']
    dataset = config['dataset']

    # Resolve project root (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent

    logger.info(f"Task: {task}, Dataset: {dataset}")

    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASKS.keys())}")

    task_info = TASKS[task]

    # Create output directory
    output_dir = project_root / config.get('output_dir', f'results/{task}_{dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load and prepare data based on dataset
    features = config.get('features', DEFAULT_FEATURES)
    first_visit_only = config.get('first_visit_only', True)
    if dataset == 'combined':
        adni_csv = project_root / config.get('adni_csv', 'data/adni/ALL_4class_clinical.csv')
        oasis_csv = project_root / config.get('oasis_csv', 'data/oasis/oasis_all.csv')
        nacc_csv = config.get('nacc_csv')
        if nacc_csv:
            nacc_csv = project_root / nacc_csv
        df_features, feature_names = prepare_data_combined(str(adni_csv), str(oasis_csv), task, features, str(nacc_csv) if nacc_csv else None, first_visit_only)
    else:
        input_csv = project_root / config['input_csv']
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} samples")
        if dataset == 'adni':
            df_features, feature_names = prepare_data_adni(df, task, features, first_visit_only)
        elif dataset == 'oasis':
            df_features, feature_names = prepare_data_oasis(df, task, features, first_visit_only)
        elif dataset == 'nacc':
            df_features, feature_names = prepare_data_nacc(df, task, features)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    if df_features is None:
        logger.error("Data preparation failed")
        return

    # Split data
    train_df, val_df, test_df = patient_level_split(
        df_features, task_info['num_classes'],
        config['train_ratio'], config['val_ratio'], config['test_ratio'],
        config['seed']
    )

    # Prepare arrays
    X_train = train_df[feature_names].values
    y_train = train_df['label'].values
    X_val = val_df[feature_names].values
    y_val = val_df['label'].values
    X_test = test_df[feature_names].values
    y_test = test_df['label'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, config, task_info, feature_names)

    # Evaluate
    test_metrics, y_pred, y_proba = evaluate_model(model, X_test_scaled, y_test, task_info, feature_names, 'Test')

    # Save artifacts
    model.save_model(str(output_dir / 'xgboost_model.json'))
    joblib.dump(scaler, output_dir / 'scaler.pkl')

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    with open(output_dir / 'task_info.json', 'w') as f:
        json.dump(task_info, f, indent=2)

    # Save predictions
    pred_dict = {'Subject': test_df['Subject'].values, 'y_true': y_test, 'y_pred': y_pred}
    for i, name in enumerate(task_info['classes']):
        pred_dict[f'prob_{name}'] = y_proba[:, i]
    pd.DataFrame(pred_dict).to_csv(output_dir / 'predictions.csv', index=False)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    if importance:
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Plot
    plot_results(y_test, y_pred, y_proba, test_metrics, task_info, output_dir, model)

    logger.info(f"\nAll results saved to {output_dir}")
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Final Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
