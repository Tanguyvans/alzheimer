#!/usr/bin/env python3
"""
Prepare 4-class tabular data: CN | MCI-stable | MCI→AD | AD

Usage:
    python 01_prepare_data_4class.py \
        --all-groups-csv ../../data/clinical_data_all_groups.csv \
        --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
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
        df: Raw dataframe

    Returns:
        Processed dataframe with engineered features
    """
    logger.info("Preparing features...")

    df_prep = df.copy()

    # Calculate AGE from EXAMDATE or birth year
    if 'EXAMDATE' in df_prep.columns:
        df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
        df_prep['exam_year'] = df_prep['EXAMDATE'].dt.year
        df_prep['AGE'] = df_prep['exam_year'] - df_prep['PTDOBYY']
        logger.info("✓ Calculated AGE using EXAMDATE")
    else:
        current_year = 2010
        df_prep['AGE'] = current_year - df_prep['PTDOBYY']
        logger.warning(f"⚠ Using approximate year {current_year} for AGE")

    # Calculate BMI
    df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Select clinical features
    clinical_features = [
        # Demographics
        'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY',
        # Physical
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

    # Keep only available features
    available_features = [f for f in clinical_features if f in df_prep.columns]
    df_features = df_prep[available_features + ['label', 'PTID']].copy()

    # Fill missing values with median for numerical features
    logger.info(f"Missing values before imputation: {df_features.isnull().sum().sum()}")

    numerical_features = df_features.select_dtypes(include=[np.number]).columns
    for col in numerical_features:
        if col not in ['label']:
            median_val = df_features[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_features[col] = df_features[col].fillna(median_val)

    logger.info(f"✓ Features prepared: {len(available_features)} features")
    logger.info(f"  Missing values remaining: {df_features.isnull().sum().sum()}")

    return df_features


def load_4class_data(all_groups_csv: str, converters_csv: str) -> pd.DataFrame:
    """
    Load and combine data for 4-class classification

    Classes:
        0: CN (Cognitively Normal)
        1: MCI-stable (MCI that did NOT convert to AD)
        2: MCI→AD (MCI that converted to AD)
        3: AD (Alzheimer's Disease)

    Args:
        all_groups_csv: Path to clinical_data_all_groups.csv
        converters_csv: Path to AD_CN_MCI_to_AD.csv (contains MCI→AD)

    Returns:
        Combined dataframe with 4 classes
    """
    logger.info("Loading 4-class data...")

    # Load both datasets
    df_all = pd.read_csv(all_groups_csv)
    df_converters = pd.read_csv(converters_csv)

    # Get MCI converter patient IDs
    mci_converter_ids = df_converters[df_converters['Group'] == 'MCI']['PTID'].unique()
    logger.info(f"  MCI converters: {len(mci_converter_ids)} unique patients")

    # Extract each class from clinical_data_all_groups.csv
    df_cn = df_all[df_all['Group'] == 'CN'].copy()
    df_cn['label'] = 0

    df_ad = df_all[df_all['Group'] == 'AD'].copy()
    df_ad['label'] = 3

    # MCI-stable: MCI patients NOT in converter list
    df_mci_all = df_all[df_all['Group'] == 'MCI'].copy()
    df_mci_stable = df_mci_all[~df_mci_all['PTID'].isin(mci_converter_ids)].copy()
    df_mci_stable['label'] = 1

    # MCI→AD: MCI patients in converter list (use data from all_groups before conversion)
    df_mci_to_ad = df_mci_all[df_mci_all['PTID'].isin(mci_converter_ids)].copy()
    df_mci_to_ad['label'] = 2

    # Combine all classes
    df_combined = pd.concat([df_cn, df_mci_stable, df_mci_to_ad, df_ad], ignore_index=True)

    # Class distribution
    logger.info(f"\n4-Class distribution:")
    logger.info(f"  CN (0):        {len(df_cn)} samples, {df_cn['PTID'].nunique()} patients")
    logger.info(f"  MCI-stable (1): {len(df_mci_stable)} samples, {df_mci_stable['PTID'].nunique()} patients")
    logger.info(f"  MCI→AD (2):     {len(df_mci_to_ad)} samples, {df_mci_to_ad['PTID'].nunique()} patients")
    logger.info(f"  AD (3):        {len(df_ad)} samples, {df_ad['PTID'].nunique()} patients")
    logger.info(f"  Total:         {len(df_combined)} samples")

    return df_combined


def create_patient_level_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits at PATIENT level for 4 classes

    Splits each class separately to maintain class distribution
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    logger.info("\nCreating patient-level splits...")

    # Split each class separately
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

    # Get unique patients for each class
    class_patients = {}
    splits = {'train': [], 'val': [], 'test': []}

    for label in [0, 1, 2, 3]:
        class_patients[label] = df[df['label'] == label]['PTID'].unique()
        train, val, test = split_patients(class_patients[label])
        splits['train'].append(train)
        splits['val'].append(val)
        splits['test'].append(test)

    # Combine all classes for each split
    train_patients = np.concatenate(splits['train'])
    val_patients = np.concatenate(splits['val'])
    test_patients = np.concatenate(splits['test'])

    # Create dataframes with all scans from assigned patients
    train_df = df[df['PTID'].isin(train_patients)].copy()
    val_df = df[df['PTID'].isin(val_patients)].copy()
    test_df = df[df['PTID'].isin(test_patients)].copy()

    # Print split statistics
    logger.info("\nSplit statistics:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        logger.info(f"\n{split_name}: {len(split_df)} samples")
        for label, name in [(0, 'CN'), (1, 'MCI-stable'), (2, 'MCI→AD'), (3, 'AD')]:
            count = (split_df['label'] == label).sum()
            logger.info(f"  {name}: {count}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare 4-class tabular data')
    parser.add_argument('--all-groups-csv', type=str, required=True,
                        help='Path to clinical_data_all_groups.csv')
    parser.add_argument('--converters-csv', type=str, required=True,
                        help='Path to AD_CN_MCI_to_AD.csv')
    parser.add_argument('--output-dir', type=str, default='data/splits',
                        help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Load 4-class data
    df_combined = load_4class_data(args.all_groups_csv, args.converters_csv)

    # Prepare features
    df_features = prepare_features(df_combined)

    # Create patient-level splits
    train_df, val_df, test_df = create_patient_level_splits(
        df_features,
        random_seed=args.seed
    )

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop PTID before saving
    train_df.drop('PTID', axis=1).to_csv(output_dir / 'train.csv', index=False)
    val_df.drop('PTID', axis=1).to_csv(output_dir / 'val.csv', index=False)
    test_df.drop('PTID', axis=1).to_csv(output_dir / 'test.csv', index=False)

    logger.info(f"\n✓ Saved splits to {output_dir}/")
    logger.info(f"  - train.csv: {len(train_df)} samples")
    logger.info(f"  - val.csv: {len(val_df)} samples")
    logger.info(f"  - test.csv: {len(test_df)} samples")

    # Feature summary
    feature_cols = [col for col in train_df.columns if col not in ['label', 'PTID']]
    logger.info(f"\nFeatures ({len(feature_cols)}): {feature_cols}")


if __name__ == '__main__':
    main()
