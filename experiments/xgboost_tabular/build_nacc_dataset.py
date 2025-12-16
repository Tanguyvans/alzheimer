#!/usr/bin/env python3
"""
Build NACC tabular dataset for XGBoost classification.

Maps NACC features to ADNI-equivalent names for unified training.

Usage:
    python build_nacc_dataset.py --output data/nacc/nacc_tabular.csv
    python build_nacc_dataset.py --mri-only --output data/nacc/nacc_tabular_mri.csv
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NACC to ADNI feature mapping
NACC_TO_ADNI_MAPPING = {
    # Demographics
    'NACCAGE': 'AGE',
    'SEX': 'PTGENDER',  # NACC: 1=M, 2=F; ADNI: 1=M, 2=F (same)
    'EDUC': 'PTEDUCAT',
    'MARISTAT': 'PTMARRY',

    # Vitals
    'WEIGHT': 'VSWEIGHT',
    'NACCBMI': 'BMI',

    # Medical history
    'ALCOHOL': 'MH14ALCH',  # Alcohol abuse
    'TOBAC30': 'MH16SMOK',  # Smoking
    'CVHATT': 'MH4CARD',    # Cardiovascular/heart attack
    'NACCADEP': 'MHPSYCH',  # Depression/psychiatric (approximation)
    'STROKE': 'MH2NEURL',   # Stroke/neurological

    # Neuropsych tests
    'TRAILA': 'TRAASCOR',   # Trail Making A
    'TRAILB': 'TRABSCOR',   # Trail Making B
    'TRAILBRR': 'TRABERRCOM',  # Trail B errors
    'ANIMALS': 'CATANIMSC',    # Category fluency (animals)
    'MOCACLOC': 'CLOCKSCOR',   # MoCA clock drawing (0-3) as clock score proxy
    'BOSTON': 'BNTTOTAL',      # Boston Naming Test
    'DIGIF': 'DSPANFOR',       # Digit span forward
    'DIGIB': 'DSPANBAC',       # Digit span backward
}

# Columns to load from NACC
NACC_COLUMNS = [
    'NACCID', 'NACCVNUM', 'NACCUDSD', 'VISITYR',
    'NACCAGE', 'SEX', 'EDUC', 'MARISTAT',
    'WEIGHT', 'NACCBMI',
    'ALCOHOL', 'TOBAC30', 'CVHATT', 'NACCADEP', 'STROKE',
    'TRAILA', 'TRAILB', 'TRAILBRR', 'ANIMALS', 'MOCACLOC', 'BOSTON', 'DIGIF', 'DIGIB',
    'NACCMMSE', 'NACCMOCA',  # Keep for reference (but exclude from features)
]


def load_nacc_data(uds_csv: str, mri_csv: str = None, mri_only: bool = False):
    """Load NACC UDS data, optionally filtering for MRI subjects."""

    logger.info(f"Loading NACC UDS data from {uds_csv}")
    df = pd.read_csv(uds_csv, usecols=NACC_COLUMNS, low_memory=False)
    logger.info(f"Loaded {len(df):,} visits from {df['NACCID'].nunique():,} subjects")

    if mri_only and mri_csv:
        logger.info(f"Loading MRI metadata from {mri_csv}")
        mri = pd.read_csv(mri_csv, usecols=['NACCID', 'NACCVNUM', 'MRIT1'])

        # Filter for T1 scans
        mri_t1 = mri[mri['MRIT1'] == 1][['NACCID', 'NACCVNUM']]
        logger.info(f"Found {len(mri_t1):,} T1 MRI scans")

        # Merge to keep only visits with T1 MRI
        df = df.merge(mri_t1, on=['NACCID', 'NACCVNUM'], how='inner')
        logger.info(f"After MRI filter: {len(df):,} visits from {df['NACCID'].nunique():,} subjects")

    return df


def map_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Map NACC diagnosis to CN/MCI/AD labels."""

    # NACCUDSD: 1=Normal, 2=Impaired not MCI, 3=MCI, 4=Dementia
    diagnosis_map = {
        1: 'CN',
        2: 'Impaired',  # Will be excluded for CN vs AD
        3: 'MCI',
        4: 'AD'  # Dementia mapped to AD (includes AD + other dementias)
    }

    df['DX'] = df['NACCUDSD'].map(diagnosis_map)

    logger.info("Diagnosis distribution:")
    for dx, count in df['DX'].value_counts().items():
        logger.info(f"  {dx}: {count:,}")

    return df


def clean_nacc_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NACC special values (88, 888, 8888 = unknown/not done)."""

    # NACC uses 88, 888, 8888 for unknown/not applicable
    special_values = [88, 888, 8888, 88.88, 888.88, 8888.88, 9999, 9999.99]

    feature_cols = list(NACC_TO_ADNI_MAPPING.keys())

    for col in feature_cols:
        if col in df.columns:
            # Replace special values with NaN
            df[col] = df[col].replace(special_values, np.nan)

    return df


def map_features_to_adni(df: pd.DataFrame) -> pd.DataFrame:
    """Rename NACC features to ADNI-equivalent names."""

    # Create Subject column from NACCID
    df['Subject'] = df['NACCID']

    # Rename features
    rename_dict = {nacc: adni for nacc, adni in NACC_TO_ADNI_MAPPING.items() if nacc in df.columns}
    df = df.rename(columns=rename_dict)

    # Log feature availability
    mapped_features = list(rename_dict.values())
    logger.info(f"Mapped {len(mapped_features)} features to ADNI names")

    return df


def prepare_nacc_dataset(
    uds_csv: str,
    mri_csv: str = None,
    output_csv: str = None,
    mri_only: bool = False,
    task: str = 'cn_ad',
    first_visit_only: bool = True
):
    """Prepare NACC dataset for XGBoost training."""

    # Load data
    df = load_nacc_data(uds_csv, mri_csv, mri_only)

    # Take only first visit per subject (NACCVNUM=1 is baseline)
    if first_visit_only:
        before = len(df)
        df = df.sort_values(['NACCID', 'NACCVNUM']).groupby('NACCID').first().reset_index()
        logger.info(f"First visit only: {before:,} -> {len(df):,} samples ({len(df):,} subjects)")

    # Map diagnosis
    df = map_diagnosis(df)

    # Clean special values
    df = clean_nacc_values(df)

    # Map features to ADNI names
    df = map_features_to_adni(df)

    # Filter by task
    if task == 'cn_ad':
        df = df[df['DX'].isin(['CN', 'AD'])].copy()
        logger.info(f"Filtered to CN vs AD: {len(df):,} samples")
    elif task == 'cn_mci_ad':
        df = df[df['DX'].isin(['CN', 'MCI', 'AD'])].copy()
        logger.info(f"Filtered to CN vs MCI vs AD: {len(df):,} samples")

    # Select final columns
    adni_features = list(NACC_TO_ADNI_MAPPING.values())
    output_cols = ['Subject', 'DX', 'VISITYR'] + adni_features

    # Keep only available columns
    output_cols = [c for c in output_cols if c in df.columns]
    df_output = df[output_cols].copy()

    # Report missing values
    logger.info("\nFeature completeness:")
    for col in adni_features:
        if col in df_output.columns:
            missing_pct = 100 * df_output[col].isna().sum() / len(df_output)
            logger.info(f"  {col}: {100-missing_pct:.1f}% complete")

    # Save
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False)
        logger.info(f"\nSaved {len(df_output):,} samples to {output_csv}")

    return df_output


def main():
    parser = argparse.ArgumentParser(description='Build NACC tabular dataset')
    parser.add_argument('--uds-csv', type=str,
                        default='data/nacc/investigator_ftldlbd_nacc71.csv',
                        help='Path to NACC UDS CSV')
    parser.add_argument('--mri-csv', type=str,
                        default='data/nacc/investigator_mri_nacc71.csv',
                        help='Path to NACC MRI metadata CSV')
    parser.add_argument('--output', type=str,
                        default='data/nacc/nacc_tabular.csv',
                        help='Output CSV path')
    parser.add_argument('--mri-only', action='store_true',
                        help='Only include subjects with T1 MRI')
    parser.add_argument('--task', type=str, default='cn_ad',
                        choices=['cn_ad', 'cn_mci_ad', 'all'],
                        help='Classification task (determines which classes to include)')

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    uds_csv = project_root / args.uds_csv
    mri_csv = project_root / args.mri_csv if args.mri_csv else None
    output_csv = project_root / args.output

    prepare_nacc_dataset(
        uds_csv=str(uds_csv),
        mri_csv=str(mri_csv) if mri_csv else None,
        output_csv=str(output_csv),
        mri_only=args.mri_only,
        task=args.task if args.task != 'all' else 'cn_mci_ad'
    )


if __name__ == '__main__':
    main()
