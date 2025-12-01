#!/usr/bin/env python3
"""
Step 1: Compute longitudinal cognitive change features.

Computes change rates in cognitive scores between baseline and follow-up visits.

Features computed:
- Trail Making A/B change rate
- Category fluency change rate
- Clock drawing change rate
- Boston Naming Test change rate
- Digit span change rate
- Time between assessments

Usage:
    python 01_compute_features.py
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TABULAR_DIR = DATA_DIR / "tabular"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_cognitive_data():
    """Load cognitive test data from multiple sources"""

    # Load NEUROBAT (main cognitive scores)
    neuro = pd.read_csv(TABULAR_DIR / "3D_MPRAGE_Imaging_Cohort_NEUROBAT_10Oct2025.csv",
                        low_memory=False)
    neuro['VISDATE'] = pd.to_datetime(neuro['VISDATE'], errors='coerce')

    # Load BLSCHECK for MMSE if available
    try:
        blscheck = pd.read_csv(TABULAR_DIR / "3D_MPRAGE_Imaging_Cohort_BLSCHECK_10Oct2025.csv",
                              low_memory=False)
        if 'MMSCORE' in blscheck.columns:
            neuro = neuro.merge(
                blscheck[['PTID', 'VISCODE', 'MMSCORE']],
                on=['PTID', 'VISCODE'],
                how='left'
            )
    except:
        pass

    # Load clinical data which has MMSCORE
    clinical = pd.read_csv(DATA_DIR / "ALL_4class_clinical.csv", low_memory=False)
    if 'MMSCORE' in clinical.columns and 'MMSCORE' not in neuro.columns:
        clinical_mms = clinical[['PTID', 'VISCODE', 'MMSCORE']].dropna()
        neuro = neuro.merge(clinical_mms, on=['PTID', 'VISCODE'], how='left')

    logger.info(f"Loaded cognitive data: {len(neuro)} records, {neuro['PTID'].nunique()} patients")

    return neuro


def compute_change_rate(baseline_val, followup_val, years_between):
    """Compute annualized change rate"""
    if pd.isna(baseline_val) or pd.isna(followup_val) or years_between <= 0:
        return np.nan
    return (followup_val - baseline_val) / years_between


def compute_patient_longitudinal_features(patient_data: pd.DataFrame) -> dict:
    """
    Compute longitudinal features for a single patient.

    Args:
        patient_data: DataFrame with all visits for one patient

    Returns:
        dict with longitudinal features
    """
    # Drop rows without dates and sort
    patient_data = patient_data.dropna(subset=['VISDATE'])
    if len(patient_data) < 2:
        return None

    patient_data = patient_data.sort_values('VISDATE')

    # Get baseline (bl visit or first visit) and last visit
    bl_data = patient_data[patient_data['VISCODE'] == 'bl']
    if len(bl_data) > 0:
        baseline = bl_data.iloc[0]
    else:
        baseline = patient_data.iloc[0]

    last_visit = patient_data.iloc[-1]

    # Time between visits
    if pd.notna(baseline['VISDATE']) and pd.notna(last_visit['VISDATE']):
        days_between = (last_visit['VISDATE'] - baseline['VISDATE']).days
        years_between = days_between / 365.25
    else:
        return None

    if years_between < 0.25:  # Less than 3 months - skip
        return None

    features = {
        'PTID': baseline['PTID'],
        'n_visits': len(patient_data),
        'years_between': years_between,
        'baseline_date': baseline['VISDATE'],
        'last_date': last_visit['VISDATE'],
    }

    # Cognitive scores to track
    cog_scores = {
        'TRAASCOR': 'trail_a',      # Trail Making A (higher = worse)
        'TRABSCOR': 'trail_b',      # Trail Making B (higher = worse)
        'CATANIMSC': 'category',    # Category fluency (lower = worse)
        'CLOCKSCOR': 'clock',       # Clock drawing (lower = worse)
        'BNTTOTAL': 'bnt',          # Boston Naming (lower = worse)
        'MMSCORE': 'mmse',          # MMSE (lower = worse)
        'DSPANFOR': 'digit_fwd',    # Digit span forward
        'DSPANBAC': 'digit_bwd',    # Digit span backward
    }

    for col, name in cog_scores.items():
        if col in patient_data.columns:
            bl_val = baseline.get(col)
            last_val = last_visit.get(col)

            # Baseline value
            features[f'{name}_baseline'] = bl_val
            features[f'{name}_last'] = last_val

            # Change (last - baseline)
            if pd.notna(bl_val) and pd.notna(last_val):
                features[f'{name}_change'] = last_val - bl_val
                features[f'{name}_change_rate'] = compute_change_rate(bl_val, last_val, years_between)
                features[f'{name}_pct_change'] = ((last_val - bl_val) / bl_val * 100) if bl_val != 0 else np.nan

    return features


def compute_all_longitudinal_features(neuro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute longitudinal features for all patients"""

    # Filter to patients with 2+ visits
    visit_counts = neuro_df.groupby('PTID').size()
    multi_visit_patients = visit_counts[visit_counts >= 2].index

    logger.info(f"Patients with 2+ visits: {len(multi_visit_patients)}")

    results = []

    for ptid in multi_visit_patients:
        patient_data = neuro_df[neuro_df['PTID'] == ptid].copy()
        features = compute_patient_longitudinal_features(patient_data)
        if features:
            results.append(features)

    results_df = pd.DataFrame(results)
    logger.info(f"Computed longitudinal features for {len(results_df)} patients")

    return results_df


def merge_with_clinical(long_features: pd.DataFrame, clinical_path: Path) -> pd.DataFrame:
    """Merge longitudinal features with clinical data"""

    clinical = pd.read_csv(clinical_path, low_memory=False)

    # Get one record per patient (baseline)
    clinical_baseline = clinical.groupby('PTID').first().reset_index()

    # Merge
    merged = clinical_baseline.merge(long_features, on='PTID', how='inner')

    logger.info(f"Merged: {len(merged)} patients with both clinical and longitudinal data")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Compute longitudinal cognitive features')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / 'longitudinal_features.csv'

    # Load cognitive data
    neuro_df = load_cognitive_data()

    # Compute longitudinal features
    long_features = compute_all_longitudinal_features(neuro_df)

    # Save raw longitudinal features
    long_features.to_csv(output_path, index=False)
    logger.info(f"Saved longitudinal features to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("LONGITUDINAL FEATURE SUMMARY")
    print("=" * 70)

    print(f"\nPatients with longitudinal data: {len(long_features)}")
    print(f"Average follow-up time: {long_features['years_between'].mean():.1f} years")
    print(f"Average visits: {long_features['n_visits'].mean():.1f}")

    # Change rate statistics for available scores
    change_cols = [c for c in long_features.columns if '_change_rate' in c]

    print("\nCognitive Change Rates (per year):")
    print("-" * 50)
    for col in change_cols:
        if long_features[col].notna().sum() > 10:
            mean_change = long_features[col].mean()
            std_change = long_features[col].std()
            n_valid = long_features[col].notna().sum()
            score_name = col.replace('_change_rate', '')
            print(f"  {score_name:15s}: {mean_change:+.2f} +/- {std_change:.2f} (n={n_valid})")

    # Merge with clinical for final dataset
    clinical_path = DATA_DIR / "ALL_4class_clinical.csv"
    if clinical_path.exists():
        merged = merge_with_clinical(long_features, clinical_path)
        merged_path = OUTPUT_DIR / 'longitudinal_features_with_clinical.csv'
        merged.to_csv(merged_path, index=False)
        logger.info(f"Saved merged dataset to {merged_path}")

        # Show class distribution
        if 'DX' in merged.columns or 'Group' in merged.columns:
            class_col = 'DX' if 'DX' in merged.columns else 'Group'
            print(f"\nClass distribution (patients with longitudinal data):")
            print(merged[class_col].value_counts())


if __name__ == '__main__':
    main()
