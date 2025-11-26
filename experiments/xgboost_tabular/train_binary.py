#!/usr/bin/env python3
"""
Train XGBoost for binary classification: CN vs (AD + MCI→AD)

Usage:
    python train_binary.py \
        --input-csv /path/to/ALL_classes_clinical.csv \
        --output-dir results/binary \
        --seed 42
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


CLINICAL_FEATURES = [
    # Demographics
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY', 'PTTLANG',
    # Physical measurements
    'VSWEIGHT', 'VSHEIGHT', 'BMI',
    # Medical history
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    # Cognitive scores
    'MMSCORE', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    # Clinical assessments
    'CDGLOBAL', 'BCFAQ', 'BCDEPRES',
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features and create binary labels"""
    df_prep = df.copy()

    # Calculate AGE
    if 'Acq Date' in df_prep.columns:
        df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
        df_prep['acq_year'] = df_prep['Acq Date'].dt.year
        df_prep['AGE'] = df_prep['acq_year'] - df_prep['PTDOBYY']
    else:
        df_prep['AGE'] = 2010 - df_prep['PTDOBYY']

    # Calculate BMI
    df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Binary label: CN=0, AD (including MCI→AD)=1
    # DX column contains the actual diagnosis (MCI patients who converted to AD have DX='AD')
    df_prep['label'] = (df_prep['DX'] == 'AD').astype(int)

    # Filter to only CN and AD (excluding MCI stable)
    df_prep = df_prep[df_prep['DX'].isin(['CN', 'AD'])].copy()

    # Get available features
    available_features = [f for f in CLINICAL_FEATURES if f in df_prep.columns]
    missing_features = [f for f in CLINICAL_FEATURES if f not in df_prep.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    # Select features
    df_features = df_prep[available_features + ['label', 'Subject']].copy()

    # Impute missing values with median
    for col in available_features:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)

    logger.info(f"Prepared {len(df_features)} samples with {len(available_features)} features")
    logger.info(f"Class distribution: CN={sum(df_features['label']==0)}, AD={sum(df_features['label']==1)}")

    return df_features, available_features


def patient_level_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data at patient level to prevent data leakage"""
    # Get unique patients per class
    cn_patients = df[df['label'] == 0]['Subject'].unique()
    ad_patients = df[df['label'] == 1]['Subject'].unique()

    logger.info(f"Unique patients - CN: {len(cn_patients)}, AD: {len(ad_patients)}")

    def split_patients(patients):
        train_p, temp_p = train_test_split(patients, test_size=(val_ratio + test_ratio), random_state=seed)
        val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        return train_p, val_p, test_p

    cn_train, cn_val, cn_test = split_patients(cn_patients)
    ad_train, ad_val, ad_test = split_patients(ad_patients)

    train_df = df[df['Subject'].isin(np.concatenate([cn_train, ad_train]))]
    val_df = df[df['Subject'].isin(np.concatenate([cn_val, ad_val]))]
    test_df = df[df['Subject'].isin(np.concatenate([cn_test, ad_test]))]

    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

    return train_df, val_df, test_df


def train_xgboost(X_train, y_train, X_val, y_val, use_class_weights=True):
    """Train XGBoost with early stopping"""
    # Class weights
    if use_class_weights:
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1]
        logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    logger.info(f"Best iteration: {model.best_iteration}")

    return model


def evaluate_model(model, X, y, feature_names, dataset_name='Test'):
    """Evaluate model and return metrics"""
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'auc_roc': roc_auc_score(y, y_proba),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'classification_report': classification_report(y, y_pred, target_names=['CN', 'AD'], output_dict=True)
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name.upper()} RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
    logger.info(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['CN', 'AD'])}")

    return metrics, y_pred, y_proba


def plot_results(y_true, y_proba, metrics, output_dir):
    """Generate and save plots"""
    output_dir = Path(output_dir)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification (CN vs AD)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Binary Classification')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train binary XGBoost classifier')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to ALL_classes_clinical.csv')
    parser.add_argument('--output-dir', type=str, default='results/binary', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} samples")

    df_features, feature_names = prepare_data(df)

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(df_features, seed=args.seed)

    # Prepare arrays
    X_train = train_df[feature_names].values
    y_train = train_df['label'].values
    X_val = val_df[feature_names].values
    y_val = val_df['label'].values
    X_test = test_df[feature_names].values
    y_test = test_df['label'].values

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate
    test_metrics, y_pred, y_proba = evaluate_model(model, X_test_scaled, y_test, feature_names, 'Test')

    # Save model and artifacts
    model.save_model(str(output_dir / 'xgboost_model.json'))
    joblib.dump(scaler, output_dir / 'scaler.pkl')

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Plot results
    plot_results(y_test, y_proba, test_metrics, output_dir)

    logger.info(f"\nAll results saved to {output_dir}")
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Final Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"Final Test AUC-ROC: {test_metrics['auc_roc']:.4f}")


if __name__ == '__main__':
    main()
