#!/usr/bin/env python3
"""
Predict using the trained longitudinal XGBoost model.

Requires tabular data with at least 2 visits per patient to compute change rates.

Usage:
    # Predict on a CSV with multiple visits per patient
    python predict.py --input patient_data.csv --output predictions.csv

    # The input CSV must have:
    # - PTID: patient identifier
    # - VISCODE: visit code (bl, m06, m12, etc.)
    # - VISDATE: visit date
    # - Cognitive scores: TRAASCOR, TRABSCOR, CATANIMSC, CLOCKSCOR, etc.
    # - Clinical features: AGE (or PTDOBYY), PTGENDER, PTEDUCAT, etc.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "results"

# Features expected by the model
BASELINE_FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY',
    'VSWEIGHT', 'VSHEIGHT', 'BMI',
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    'CDGLOBAL', 'BCFAQ', 'BCDEPRES', 'MMSCORE'
]

LONGITUDINAL_FEATURES = [
    'years_between', 'n_visits',
    'trail_a_change_rate', 'trail_b_change_rate',
    'category_change_rate', 'clock_change_rate',
    'bnt_change_rate', 'digit_fwd_change_rate', 'digit_bwd_change_rate',
    'trail_a_pct_change', 'trail_b_pct_change',
    'category_pct_change', 'clock_pct_change'
]

COG_SCORES = {
    'TRAASCOR': 'trail_a',
    'TRABSCOR': 'trail_b',
    'CATANIMSC': 'category',
    'CLOCKSCOR': 'clock',
    'BNTTOTAL': 'bnt',
    'MMSCORE': 'mmse',
    'DSPANFOR': 'digit_fwd',
    'DSPANBAC': 'digit_bwd',
}


def compute_patient_features(patient_data: pd.DataFrame) -> dict:
    """
    Compute all features for a single patient from their visit data.

    Args:
        patient_data: DataFrame with all visits for one patient

    Returns:
        dict with baseline + longitudinal features
    """
    # Parse dates
    if 'VISDATE' in patient_data.columns:
        patient_data = patient_data.copy()
        patient_data['VISDATE'] = pd.to_datetime(patient_data['VISDATE'], errors='coerce')
        patient_data = patient_data.dropna(subset=['VISDATE'])

    if len(patient_data) < 2:
        logger.warning(f"Patient {patient_data['PTID'].iloc[0]} has < 2 visits, cannot compute longitudinal features")
        return None

    patient_data = patient_data.sort_values('VISDATE')

    # Get baseline and last visit
    bl_data = patient_data[patient_data['VISCODE'] == 'bl']
    baseline = bl_data.iloc[0] if len(bl_data) > 0 else patient_data.iloc[0]
    last_visit = patient_data.iloc[-1]

    # Time between visits
    days_between = (last_visit['VISDATE'] - baseline['VISDATE']).days
    years_between = days_between / 365.25

    if years_between < 0.25:
        logger.warning(f"Patient {baseline['PTID']} has < 3 months follow-up")
        return None

    # Start with baseline features
    features = {'PTID': baseline['PTID']}

    # Add baseline clinical features
    for feat in BASELINE_FEATURES:
        if feat in baseline.index:
            features[feat] = baseline[feat]

    # Compute derived features
    if 'AGE' not in features and 'PTDOBYY' in baseline.index:
        features['AGE'] = 2024 - baseline['PTDOBYY']

    if 'BMI' not in features and 'VSWEIGHT' in features and 'VSHEIGHT' in features:
        if features['VSHEIGHT'] > 0:
            features['BMI'] = features['VSWEIGHT'] / ((features['VSHEIGHT'] / 100) ** 2)

    # Add longitudinal features
    features['years_between'] = years_between
    features['n_visits'] = len(patient_data)

    # Compute change rates for cognitive scores
    for col, name in COG_SCORES.items():
        if col in patient_data.columns:
            bl_val = baseline.get(col)
            last_val = last_visit.get(col)

            if pd.notna(bl_val) and pd.notna(last_val) and years_between > 0:
                change = last_val - bl_val
                features[f'{name}_change_rate'] = change / years_between
                if bl_val != 0:
                    features[f'{name}_pct_change'] = (change / bl_val) * 100

    return features


def load_model(model_dir: Path = None):
    """Load the trained model, scaler, and feature names."""
    model_dir = model_dir or MODEL_DIR

    model = xgb.Booster()
    model.load_model(str(model_dir / 'xgboost_longitudinal.json'))

    scaler = joblib.load(model_dir / 'scaler.pkl')

    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def predict(input_df: pd.DataFrame, model_dir: Path = None) -> pd.DataFrame:
    """
    Make predictions on input data.

    Args:
        input_df: DataFrame with patient visits (must have PTID, VISCODE, VISDATE, and features)
        model_dir: Directory containing the trained model

    Returns:
        DataFrame with predictions per patient
    """
    model, scaler, feature_names = load_model(model_dir)

    # Compute features for each patient
    patients = input_df['PTID'].unique()
    logger.info(f"Processing {len(patients)} patients...")

    all_features = []
    for ptid in patients:
        patient_data = input_df[input_df['PTID'] == ptid]
        features = compute_patient_features(patient_data)
        if features:
            all_features.append(features)

    if not all_features:
        raise ValueError("No patients with valid longitudinal data (need 2+ visits)")

    features_df = pd.DataFrame(all_features)
    logger.info(f"Computed features for {len(features_df)} patients")

    # Prepare feature matrix
    X = pd.DataFrame(index=features_df.index, columns=feature_names)
    for col in feature_names:
        if col in features_df.columns:
            X[col] = features_df[col]
        else:
            X[col] = 0  # Missing features filled with 0

    # Fill NaN with median (or 0 if all NaN)
    for col in X.columns:
        if X[col].isnull().all():
            X[col] = 0
        elif X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Scale and predict
    X_scaled = scaler.transform(X.values)
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)

    y_proba = model.predict(dmatrix)
    y_pred = np.argmax(y_proba, axis=1)

    # Build results
    class_names = ['CN', 'MCI', 'AD']
    results = pd.DataFrame({
        'PTID': features_df['PTID'],
        'prediction': [class_names[p] for p in y_pred],
        'prob_CN': y_proba[:, 0],
        'prob_MCI': y_proba[:, 1],
        'prob_AD': y_proba[:, 2],
        'confidence': np.max(y_proba, axis=1),
        'n_visits': features_df['n_visits'],
        'years_between': features_df['years_between']
    })

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict using longitudinal XGBoost model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with patient visits')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions CSV')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading data from {args.input}")
    input_df = pd.read_csv(args.input, low_memory=False)

    # Check required columns
    required = ['PTID', 'VISCODE', 'VISDATE']
    missing = [c for c in required if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(input_df)} records for {input_df['PTID'].nunique()} patients")

    # Make predictions
    model_dir = Path(args.model_dir) if args.model_dir else None
    results = predict(input_df, model_dir)

    # Save results
    results.to_csv(args.output, index=False)
    logger.info(f"Saved predictions to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\nPatients processed: {len(results)}")
    print(f"\nPrediction distribution:")
    print(results['prediction'].value_counts())
    print(f"\nMean confidence: {results['confidence'].mean():.1%}")
    print(f"\nSample predictions:")
    print(results.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
