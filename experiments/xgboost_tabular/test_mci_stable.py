#!/usr/bin/env python3
"""
Test trained XGBoost model on MCI stable patients.

Usage:
    python test_mci_stable.py \
        --model-dir results/cn_ad_mci_ad \
        --mci-csv /path/to/mci_stable_patients.csv \
        --output-dir results/mci_stable_analysis
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_features(df, feature_names):
    """Prepare features for prediction"""
    df_prep = df.copy()

    # Calculate AGE
    if 'EXAMDATE' in df_prep.columns:
        df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
        df_prep['exam_year'] = df_prep['EXAMDATE'].dt.year
        df_prep['AGE'] = df_prep['exam_year'] - df_prep['PTDOBYY']
    elif 'Acq Date' in df_prep.columns:
        df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
        df_prep['acq_year'] = df_prep['Acq Date'].dt.year
        df_prep['AGE'] = df_prep['acq_year'] - df_prep['PTDOBYY']
    else:
        df_prep['AGE'] = 2010 - df_prep['PTDOBYY']

    # Calculate BMI
    if 'VSWEIGHT' in df_prep.columns and 'VSHEIGHT' in df_prep.columns:
        df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Get available features
    available = [f for f in feature_names if f in df_prep.columns]
    missing = [f for f in feature_names if f not in df_prep.columns]

    if missing:
        logger.warning(f"Missing features (will be filled with 0): {missing}")
        for f in missing:
            df_prep[f] = 0

    # Extract features
    X = df_prep[feature_names].copy()

    # Fill NaN with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return X, df_prep


def main():
    parser = argparse.ArgumentParser(description='Test XGBoost on MCI stable patients')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--mci-csv', type=str, required=True, help='Path to MCI stable CSV')
    parser.add_argument('--output-dir', type=str, default='results/mci_stable_analysis', help='Output directory')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and scaler
    logger.info(f"Loading model from {model_dir}")
    model = xgb.Booster()
    model.load_model(str(model_dir / 'xgboost_model.json'))

    scaler = joblib.load(model_dir / 'scaler.pkl')

    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    logger.info(f"Loaded model with {len(feature_names)} features")

    # Load MCI stable data
    logger.info(f"Loading MCI stable data from {args.mci_csv}")
    mci_df = pd.read_csv(args.mci_csv)
    logger.info(f"Loaded {len(mci_df)} MCI stable samples")

    # Prepare features
    X, mci_df_prep = prepare_features(mci_df, feature_names)
    X_scaled = scaler.transform(X)

    # Predict
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    # Results
    n_cn_like = sum(y_pred == 0)
    n_ad_like = sum(y_pred == 1)

    logger.info(f"\n{'='*60}")
    logger.info("MCI STABLE CLASSIFICATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total MCI stable samples: {len(y_pred)}")
    logger.info(f"Classified as CN-like: {n_cn_like} ({n_cn_like/len(y_pred)*100:.1f}%)")
    logger.info(f"Classified as AD-like: {n_ad_like} ({n_ad_like/len(y_pred)*100:.1f}%)")
    logger.info(f"\nMean AD probability: {y_proba.mean():.3f}")
    logger.info(f"Median AD probability: {np.median(y_proba):.3f}")

    # Probability distribution
    logger.info(f"\nProbability distribution:")
    bins = [(0, 0.3, 'CN-like (<30%)'),
            (0.3, 0.5, 'Mild (30-50%)'),
            (0.5, 0.7, 'Moderate (50-70%)'),
            (0.7, 1.0, 'AD-like (>70%)')]

    for low, high, label in bins:
        count = sum((y_proba >= low) & (y_proba < high))
        logger.info(f"  {label}: {count} ({count/len(y_proba)*100:.1f}%)")

    # Save predictions
    results_df = mci_df_prep.copy()
    results_df['AD_probability'] = y_proba
    results_df['predicted_class'] = ['AD-like' if p == 1 else 'CN-like' for p in y_pred]

    # Get subject ID column
    subject_col = 'Subject' if 'Subject' in results_df.columns else 'PTID'

    results_df.to_csv(output_dir / 'mci_stable_predictions.csv', index=False)
    logger.info(f"\nPredictions saved to {output_dir / 'mci_stable_predictions.csv'}")

    # Plot probability distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(y_proba, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0.5, color='red', linestyle='--', label='Decision boundary')
    axes[0].set_xlabel('AD Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MCI Stable: AD Probability Distribution')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(y_proba, vert=True)
    axes[1].axhline(0.5, color='red', linestyle='--', label='Decision boundary')
    axes[1].set_ylabel('AD Probability')
    axes[1].set_title('MCI Stable: AD Probability Box Plot')
    axes[1].set_xticklabels(['MCI Stable'])

    plt.tight_layout()
    plt.savefig(output_dir / 'mci_stable_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([n_cn_like, n_ad_like],
           labels=['CN-like', 'AD-like'],
           autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'],
           explode=(0.02, 0.02))
    ax.set_title('MCI Stable Classification')
    plt.savefig(output_dir / 'mci_stable_pie.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary
    summary = {
        'total_samples': len(y_pred),
        'cn_like_count': int(n_cn_like),
        'ad_like_count': int(n_ad_like),
        'cn_like_percent': round(n_cn_like/len(y_pred)*100, 2),
        'ad_like_percent': round(n_ad_like/len(y_pred)*100, 2),
        'mean_ad_probability': round(float(y_proba.mean()), 4),
        'median_ad_probability': round(float(np.median(y_proba)), 4),
        'std_ad_probability': round(float(y_proba.std()), 4)
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
