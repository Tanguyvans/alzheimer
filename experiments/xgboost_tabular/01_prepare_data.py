#!/usr/bin/env python3
"""
Prepare tabular data for XGBoost training
CN vs AD+MCI-to-AD binary classification

Usage:
    python 01_prepare_data.py \
        --input-csv ../../data/AD_CN_MCI_to_AD.csv \
        --output-dir data/splits \
        --seed 42
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and engineer features from raw tabular data

    Args:
        df: Raw dataframe from CSV

    Returns:
        Processed dataframe with engineered features
    """
    logger.info("Preparing features...")

    # Create a copy
    df_prep = df.copy()

    # Calculate age from birth year using actual acquisition date
    if 'Acq Date' in df_prep.columns:
        # Parse acquisition date and extract year
        df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
        df_prep['acq_year'] = df_prep['Acq Date'].dt.year
        df_prep['AGE'] = df_prep['acq_year'] - df_prep['PTDOBYY']
        logger.info("Calculated AGE using actual acquisition date (Acq Date)")
    elif 'EXAMDATE' in df_prep.columns:
        # Fallback to EXAMDATE if Acq Date not available
        df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
        df_prep['exam_year'] = df_prep['EXAMDATE'].dt.year
        df_prep['AGE'] = df_prep['exam_year'] - df_prep['PTDOBYY']
        logger.info("Calculated AGE using exam date (EXAMDATE)")
    else:
        # Last resort: use fixed year (2010) as approximation
        current_year = 2010
        df_prep['AGE'] = current_year - df_prep['PTDOBYY']
        logger.warning(f"No date column found. Using approximate year {current_year} for AGE calculation")

    # BMI calculation (weight in kg, height in cm)
    df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Create binary label: CN=0, AD+MCI-to-AD=1
    # Use 'DX' column which includes MCI patients who converted to AD
    df_prep['label'] = (df_prep['DX'] == 'AD').astype(int)

    # Select clinical features for training
    clinical_features = [
        # Demographics
        'AGE',
        'PTGENDER',      # Gender
        'PTEDUCAT',      # Education years
        'PTRACCAT',      # Race
        'PTHAND',        # Handedness
        'PTMARRY',       # Marital status

        # Physical measurements
        'VSWEIGHT',      # Weight
        'VSHEIGHT',      # Height
        'BMI',           # Body mass index

        # Medical history (binary flags)
        'MH14ALCH',      # Alcohol use
        'MH17MALI',      # Malignancy
        'MH16SMOK',      # Smoking
        'MH15DRUG',      # Drug use
        'MH4CARD',       # Cardiovascular
        'MHPSYCH',       # Psychiatric
        'MH2NEURL',      # Neurological
        'MH6HEPAT',      # Hepatic
        'MH12RENA',      # Renal

        # Cognitive scores (key predictors)
        'MMSCORE',       # Mini-Mental State Exam (0-30)
        'TRAASCOR',      # Trail Making A
        'TRABSCOR',      # Trail Making B
        'TRABERRCOM',    # Trail Making errors
        'CATANIMSC',     # Category fluency (animals)
        'CLOCKSCOR',     # Clock drawing score
        'BNTTOTAL',      # Boston Naming Test
        'DSPANFOR',      # Digit span forward
        'DSPANBAC',      # Digit span backward

        # Clinical assessments
        'CDGLOBAL',      # Clinical Dementia Rating
        'BCFAQ',         # Functional Activities Questionnaire
        'BCDEPRES',      # Depression score
    ]

    # Keep only relevant columns
    df_features = df_prep[clinical_features + ['label', 'PTID']].copy()

    # Handle any remaining missing values
    logger.info(f"Missing values before imputation:\n{df_features.isnull().sum()[df_features.isnull().sum() > 0]}")

    # For numerical features, fill with median
    numerical_features = df_features.select_dtypes(include=[np.number]).columns
    for col in numerical_features:
        if col not in ['label']:
            median_val = df_features[col].median()
            df_features[col].fillna(median_val, inplace=True)

    logger.info(f"Features prepared: {len(clinical_features)} features")

    return df_features


def create_patient_level_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits at patient level

    Multiple scans per patient stay in same split to prevent data leakage
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    # Get unique patients per class
    cn_patients = df[df['label'] == 0]['PTID'].unique()
    ad_patients = df[df['label'] == 1]['PTID'].unique()

    logger.info(f"\nUnique patients:")
    logger.info(f"  CN: {len(cn_patients)}")
    logger.info(f"  AD+MCI-to-AD: {len(ad_patients)}")

    # Split patients (not scans) for each class
    def split_patients(patients):
        train_patients, temp_patients = train_test_split(
            patients,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )

        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )

        return train_patients, val_patients, test_patients

    cn_train, cn_val, cn_test = split_patients(cn_patients)
    ad_train, ad_val, ad_test = split_patients(ad_patients)

    # Create splits including all scans from assigned patients
    train_df = df[df['PTID'].isin(np.concatenate([cn_train, ad_train]))]
    val_df = df[df['PTID'].isin(np.concatenate([cn_val, ad_val]))]
    test_df = df[df['PTID'].isin(np.concatenate([cn_test, ad_test]))]

    logger.info(f"\nSplit statistics (samples, not patients):")
    logger.info(f"Train: {len(train_df)} samples (CN: {len(train_df[train_df['label']==0])}, AD: {len(train_df[train_df['label']==1])})")
    logger.info(f"Val:   {len(val_df)} samples (CN: {len(val_df[val_df['label']==0])}, AD: {len(val_df[val_df['label']==1])})")
    logger.info(f"Test:  {len(test_df)} samples (CN: {len(test_df[test_df['label']==0])}, AD: {len(test_df[test_df['label']==1])})")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare tabular data for XGBoost')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to AD_CN_MCI_to_AD.csv')
    parser.add_argument('--output-dir', type=str, default='data/splits',
                        help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} samples")

    # Prepare features
    df_features = prepare_features(df)

    # Create patient-level splits
    train_df, val_df, test_df = create_patient_level_splits(
        df_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop PTID before saving (not a feature)
    train_df.drop('PTID', axis=1).to_csv(output_dir / 'train.csv', index=False)
    val_df.drop('PTID', axis=1).to_csv(output_dir / 'val.csv', index=False)
    test_df.drop('PTID', axis=1).to_csv(output_dir / 'test.csv', index=False)

    logger.info(f"\n✓ Saved splits to {output_dir}/")
    logger.info(f"  - train.csv: {len(train_df)} samples")
    logger.info(f"  - val.csv: {len(val_df)} samples")
    logger.info(f"  - test.csv: {len(test_df)} samples")

    # Print feature summary
    logger.info(f"\nFeature summary:")
    feature_cols = [col for col in train_df.columns if col != 'label']
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Features: {feature_cols}")


if __name__ == '__main__':
    main()
