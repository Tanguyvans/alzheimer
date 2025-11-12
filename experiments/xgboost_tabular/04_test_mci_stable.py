#!/usr/bin/env python3
"""
Test trained XGBoost model on MCI-stable patients

This script answers the question: "Where do MCI-stable patients get classified?"
- Do they look more like CN (healthy)?
- Do they look more like AD (diseased)?
- Are they in between?

Usage:
    python 04_test_mci_stable.py \
        --model-dir models \
        --all-groups-csv ../../data/clinical_data_all_groups.csv \
        --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
        --output-csv mci_stable_predictions.csv
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def identify_mci_stable(all_groups_df, converters_df):
    """
    Identify MCI-stable patients (those who didn't convert to AD)

    Args:
        all_groups_df: DataFrame with all patients
        converters_df: DataFrame with MCI-to-AD converters

    Returns:
        DataFrame with MCI-stable patients only
    """
    logger.info("Identifying MCI-stable patients...")

    # Get all MCI patients
    mci_all = all_groups_df[all_groups_df['Group'] == 'MCI'].copy()

    # Get MCI converters (those in the training data)
    mci_converter_ids = converters_df[converters_df['Group'] == 'MCI']['PTID'].unique()

    # MCI-stable = MCI patients who are NOT in the converter list
    mci_stable = mci_all[~mci_all['PTID'].isin(mci_converter_ids)].copy()

    logger.info(f"\nMCI breakdown:")
    logger.info(f"  Total MCI: {len(mci_all)}")
    logger.info(f"  MCI-converters (in training as AD): {len(mci_all[mci_all['PTID'].isin(mci_converter_ids)])}")
    logger.info(f"  MCI-stable (not in training): {len(mci_stable)}")
    logger.info(f"  Unique MCI-stable patients: {mci_stable['PTID'].nunique()}")

    return mci_stable


def prepare_features_for_prediction(df, feature_names):
    """
    Prepare features matching the training format

    Args:
        df: DataFrame with MCI-stable patients
        feature_names: List of feature names from training

    Returns:
        DataFrame with only the features used in training
    """
    logger.info("Preparing features for prediction...")

    # Calculate AGE and BMI if not present
    if 'AGE' not in df.columns and 'PTDOBYY' in df.columns:
        current_year = 2010  # Approximate
        df['AGE'] = current_year - df['PTDOBYY']

    if 'BMI' not in df.columns:
        df['BMI'] = df['VSWEIGHT'] / ((df['VSHEIGHT'] / 100) ** 2)

    # Select only the features used in training
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        logger.warning("These will be filled with 0")

    # Create feature dataframe with all required features
    X = pd.DataFrame()
    for feat in feature_names:
        if feat in df.columns:
            X[feat] = df[feat]
        else:
            X[feat] = 0  # Fill missing features with 0

    # Handle missing values (same as training)
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col].fillna(median_val, inplace=True)

    logger.info(f"Prepared {len(X)} samples with {len(feature_names)} features")

    return X


def main():
    parser = argparse.ArgumentParser(description='Test XGBoost on MCI-stable patients')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--all-groups-csv', type=str, required=True,
                        help='CSV with all patient groups')
    parser.add_argument('--converters-csv', type=str, required=True,
                        help='CSV with MCI-to-AD converters (AD_CN_MCI_to_AD.csv)')
    parser.add_argument('--output-csv', type=str, default='mci_stable_predictions.csv',
                        help='Output CSV with predictions')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load model
    logger.info(f"Loading model from {model_dir}...")
    model_path = model_dir / 'xgboost_model.json'
    model = xgb.Booster()
    model.load_model(str(model_path))

    # Load scaler
    scaler_path = model_dir / 'scaler.pkl'
    scaler = joblib.load(scaler_path)

    # Load feature names
    feature_names_path = model_dir / 'feature_names.json'
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)

    logger.info("✓ Model loaded successfully")

    # Load data
    logger.info(f"\nLoading data...")
    all_groups_df = pd.read_csv(args.all_groups_csv)
    converters_df = pd.read_csv(args.converters_csv)

    # Identify MCI-stable patients
    mci_stable_df = identify_mci_stable(all_groups_df, converters_df)

    # Prepare features
    X_mci = prepare_features_for_prediction(mci_stable_df, feature_names)

    # Scale features
    X_mci_scaled = scaler.transform(X_mci)

    # Predict
    logger.info("\nMaking predictions...")
    dmatrix = xgb.DMatrix(X_mci_scaled, feature_names=feature_names)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    # Analyze predictions
    logger.info("\n" + "="*80)
    logger.info("MCI-STABLE CLASSIFICATION RESULTS")
    logger.info("="*80)

    n_classified_cn = (y_pred == 0).sum()
    n_classified_ad = (y_pred == 1).sum()

    logger.info(f"\nTotal MCI-stable patients: {len(y_pred)}")
    logger.info(f"  Classified as CN (0): {n_classified_cn} ({n_classified_cn/len(y_pred)*100:.1f}%)")
    logger.info(f"  Classified as AD (1): {n_classified_ad} ({n_classified_ad/len(y_pred)*100:.1f}%)")

    # Probability distribution
    logger.info(f"\nProbability Statistics:")
    logger.info(f"  Mean CN probability: {(1-y_proba).mean():.3f} ± {(1-y_proba).std():.3f}")
    logger.info(f"  Mean AD probability: {y_proba.mean():.3f} ± {y_proba.std():.3f}")
    logger.info(f"  Median CN probability: {np.median(1-y_proba):.3f}")
    logger.info(f"  Median AD probability: {np.median(y_proba):.3f}")

    # Classification into bins
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    bin_labels = ['CN-like (<30% AD prob)', 'Mild (30-50%)', 'Moderate (50-70%)', 'AD-like (>70%)']
    ad_prob_binned = pd.cut(y_proba, bins=bins, labels=bin_labels)

    logger.info(f"\nDetailed probability distribution:")
    for label in bin_labels:
        count = (ad_prob_binned == label).sum()
        logger.info(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")

    # Save predictions
    logger.info(f"\nSaving predictions to {args.output_csv}...")
    results_df = mci_stable_df.copy()
    results_df['predicted_label'] = y_pred
    results_df['predicted_CN_proba'] = 1 - y_proba
    results_df['predicted_AD_proba'] = y_proba
    results_df['predicted_class'] = results_df['predicted_label'].map({0: 'CN', 1: 'AD'})
    results_df['probability_bin'] = ad_prob_binned

    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"✓ Predictions saved!")

    # Plot probability distribution
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(y_proba, bins=20, edgecolor='black', alpha=0.7, color='#2E86AB')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('AD Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Patients', fontsize=12, fontweight='bold')
    plt.title('MCI-Stable: AD Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([1-y_proba, y_proba], labels=['CN Probability', 'AD Probability'])
    plt.ylabel('Probability', fontsize=12, fontweight='bold')
    plt.title('MCI-Stable: Probability Box Plot', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = model_dir / 'mci_stable_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Distribution plot saved to {plot_path}")
    plt.close()

    # Interpretation
    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION")
    logger.info("="*80)
    logger.info("\nWhat does this tell us about MCI-stable patients?")

    if n_classified_cn > n_classified_ad:
        logger.info(f"✓ Most MCI-stable patients ({n_classified_cn/len(y_pred)*100:.1f}%) look more like CN")
        logger.info("  → These patients have mild cognitive impairment but stable brain health")
    else:
        logger.info(f"! Most MCI-stable patients ({n_classified_ad/len(y_pred)*100:.1f}%) look more like AD")
        logger.info("  → These patients show AD-like patterns despite not converting yet")

    uncertain_count = ((y_proba > 0.3) & (y_proba < 0.7)).sum()
    logger.info(f"\n{uncertain_count} patients ({uncertain_count/len(y_pred)*100:.1f}%) are in the uncertain range")
    logger.info("  → These are truly intermediate cases between CN and AD")

    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
