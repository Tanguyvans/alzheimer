#!/usr/bin/env python3
"""
Step 2: Train XGBoost with longitudinal cognitive change features.

Compares:
1. Baseline features only (cross-sectional)
2. Baseline + longitudinal change rates

Usage:
    python 02_train.py
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
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "results"

# Baseline clinical features (cross-sectional)
BASELINE_FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY',
    'VSWEIGHT', 'VSHEIGHT', 'BMI',
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    'CDGLOBAL', 'BCFAQ', 'BCDEPRES', 'MMSCORE'
]

# Longitudinal change features
LONGITUDINAL_FEATURES = [
    'years_between', 'n_visits',
    'trail_a_change_rate', 'trail_b_change_rate',
    'category_change_rate', 'clock_change_rate',
    'bnt_change_rate', 'digit_fwd_change_rate', 'digit_bwd_change_rate',
    'trail_a_pct_change', 'trail_b_pct_change',
    'category_pct_change', 'clock_pct_change'
]


def prepare_data(df: pd.DataFrame, features: list, label_col: str = 'label'):
    """Prepare features and labels"""

    df = df.copy()

    # Calculate derived features
    if 'AGE' not in df.columns and 'PTDOBYY' in df.columns:
        df['AGE'] = 2024 - df['PTDOBYY']

    if 'BMI' not in df.columns and 'VSWEIGHT' in df.columns and 'VSHEIGHT' in df.columns:
        df['BMI'] = df['VSWEIGHT'] / ((df['VSHEIGHT'] / 100) ** 2)

    # Get available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        logger.warning(f"Missing features: {missing[:10]}...")

    # Get labels
    if label_col not in df.columns:
        # Create labels from DX or Group
        if 'DX' in df.columns:
            label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
            df[label_col] = df['DX'].map(label_map)
        elif 'Group' in df.columns:
            label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
            df[label_col] = df['Group'].map(label_map)

    # Filter valid rows
    df_valid = df.dropna(subset=[label_col])

    X = df_valid[available].copy()
    y = df_valid[label_col].values

    # Impute missing values
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y, available, df_valid


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, model_name="Model"):
    """Train XGBoost and evaluate"""

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
    }

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)

    model = xgb.train(
        params, dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Predict
    y_proba = model.predict(dtest)
    y_pred = np.argmax(y_proba, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Results")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {acc*100:.1f}%")
    logger.info(f"Balanced Accuracy: {bal_acc*100:.1f}%")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD'])}")

    return {
        'model': model,
        'scaler': scaler,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    input_path = args.input or OUTPUT_DIR / 'longitudinal_features_with_clinical.csv'
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df)} samples")

    # Add trajectory info for later analysis
    df['trajectory'] = 'unknown'
    df.loc[(df['BL_DX'] == 1) & (df['LAST_DX'] == 1), 'trajectory'] = 'CN_stable'
    df.loc[(df['BL_DX'] == 2) & (df['LAST_DX'] == 2), 'trajectory'] = 'MCI_stable'
    df.loc[(df['BL_DX'] == 2) & (df['LAST_DX'] == 3), 'trajectory'] = 'MCI_to_AD'
    df.loc[(df['BL_DX'] == 3) & (df['LAST_DX'] == 3), 'trajectory'] = 'AD_stable'

    # Patient-level split
    patients = df['PTID'].unique()
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)

    train_df = df[df['PTID'].isin(train_patients)]
    test_df = df[df['PTID'].isin(test_patients)]

    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # ============================================================
    # Model 1: Baseline features only (cross-sectional)
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("MODEL 1: BASELINE FEATURES ONLY (Cross-sectional)")
    logger.info("="*70)

    X_train_bl, y_train, bl_features, _ = prepare_data(train_df, BASELINE_FEATURES)
    X_test_bl, y_test, _, test_df_valid = prepare_data(test_df, BASELINE_FEATURES)

    logger.info(f"Features used: {len(bl_features)}")
    results_baseline = train_and_evaluate(
        X_train_bl, y_train, X_test_bl, y_test, bl_features, "Baseline Only"
    )

    # ============================================================
    # Model 2: Baseline + Longitudinal features
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("MODEL 2: BASELINE + LONGITUDINAL FEATURES")
    logger.info("="*70)

    all_features = BASELINE_FEATURES + LONGITUDINAL_FEATURES

    X_train_all, y_train, all_feat_names, _ = prepare_data(train_df, all_features)
    X_test_all, y_test, _, test_df_valid = prepare_data(test_df, all_features)

    logger.info(f"Features used: {len(all_feat_names)}")
    results_longitudinal = train_and_evaluate(
        X_train_all, y_train, X_test_all, y_test, all_feat_names, "Baseline + Longitudinal"
    )

    # Save model
    results_longitudinal['model'].save_model(str(OUTPUT_DIR / 'xgboost_longitudinal.json'))
    joblib.dump(results_longitudinal['scaler'], OUTPUT_DIR / 'scaler.pkl')
    with open(OUTPUT_DIR / 'feature_names.json', 'w') as f:
        json.dump(all_feat_names, f)

    # Save test predictions with trajectory info
    test_df_valid = test_df_valid.copy()
    test_df_valid['y_pred'] = results_longitudinal['y_pred']
    test_df_valid['y_true'] = y_test
    test_df_valid['correct'] = (results_longitudinal['y_pred'] == y_test)
    test_df_valid.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)

    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Model':<30} {'Accuracy':>12} {'Balanced Acc':>15}")
    print("-" * 60)
    print(f"{'Baseline Only':<30} {results_baseline['accuracy']*100:>11.1f}% {results_baseline['balanced_accuracy']*100:>14.1f}%")
    print(f"{'Baseline + Longitudinal':<30} {results_longitudinal['accuracy']*100:>11.1f}% {results_longitudinal['balanced_accuracy']*100:>14.1f}%")

    improvement = results_longitudinal['balanced_accuracy'] - results_baseline['balanced_accuracy']
    print(f"\nImprovement: {improvement*100:+.1f}% balanced accuracy")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrices
    for ax, results, title in [
        (axes[0], results_baseline, 'Baseline Only'),
        (axes[1], results_longitudinal, 'Baseline + Longitudinal')
    ]:
        cm = results['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['CN', 'MCI', 'AD'],
                   yticklabels=['CN', 'MCI', 'AD'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{title}\nBal. Acc: {results["balanced_accuracy"]*100:.1f}%')

    plt.suptitle('XGBoost: Impact of Longitudinal Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_confusion_matrix.png', dpi=150, bbox_inches='tight')
    logger.info(f"\nSaved comparison plot to {OUTPUT_DIR / 'comparison_confusion_matrix.png'}")

    # Feature importance for longitudinal model
    importance = results_longitudinal['model'].get_score(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)

    # Highlight longitudinal features
    imp_df['is_longitudinal'] = imp_df['feature'].isin(LONGITUDINAL_FEATURES)

    print("\n" + "="*70)
    print("TOP 15 FEATURES (Baseline + Longitudinal Model)")
    print("="*70)
    for i, row in imp_df.head(15).iterrows():
        marker = " [LONG]" if row['is_longitudinal'] else ""
        print(f"  {row['feature']:25s}: {row['importance']:.2f}{marker}")

    # Save importance
    imp_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    # Save summary
    summary = {
        'baseline_accuracy': float(results_baseline['accuracy']),
        'baseline_balanced_accuracy': float(results_baseline['balanced_accuracy']),
        'longitudinal_accuracy': float(results_longitudinal['accuracy']),
        'longitudinal_balanced_accuracy': float(results_longitudinal['balanced_accuracy']),
        'improvement': float(improvement),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_baseline_features': len(bl_features),
        'n_longitudinal_features': len(all_feat_names)
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
