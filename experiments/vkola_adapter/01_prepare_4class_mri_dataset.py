#!/usr/bin/env python3
"""
Prepare 4-class MRI+Tabular Dataset for vkola-lab

Creates CSV with:
- PTID, RID, scan_path (from MRI)
- 30 tabular features
- COG label (0=CN, 1=MCI-stable, 2=MCI→AD, 3=AD)
- ADD label (0=Non-AD, 1=AD)

Usage:
    python 01_prepare_4class_mri_dataset.py \
        --clinical-csv ../../data/clinical_data_all_groups.csv \
        --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
        --skull-dir /Volumes/KINGSTON/ADNI-skull \
        --output-dir lookupcsv/CrossValid
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_4class_data(clinical_csv, converters_csv):
    """
    Load 4-class data: CN, MCI-stable, MCI→AD, AD

    Returns:
        DataFrame with PTID, features, label
    """
    logger.info("Loading clinical data...")
    df_all = pd.read_csv(clinical_csv)
    df_converters = pd.read_csv(converters_csv)

    # MCI converter patient IDs
    mci_converter_ids = df_converters[df_converters['Group'] == 'MCI']['PTID'].unique()

    logger.info(f"Total patients: {df_all['PTID'].nunique()}")
    logger.info(f"MCI converters: {len(mci_converter_ids)}")

    # Extract each class
    df_cn = df_all[df_all['Group'] == 'CN'].copy()
    df_cn['label'] = 0
    df_cn['group'] = 'CN'

    df_ad = df_all[df_all['Group'] == 'AD'].copy()
    df_ad['label'] = 3
    df_ad['group'] = 'AD'

    df_mci_all = df_all[df_all['Group'] == 'MCI'].copy()

    # MCI-stable: NOT in converter list
    df_mci_stable = df_mci_all[~df_mci_all['PTID'].isin(mci_converter_ids)].copy()
    df_mci_stable['label'] = 1
    df_mci_stable['group'] = 'MCI-stable'

    # MCI→AD: IN converter list
    df_mci_to_ad = df_mci_all[df_mci_all['PTID'].isin(mci_converter_ids)].copy()
    df_mci_to_ad['label'] = 2
    df_mci_to_ad['group'] = 'MCI→AD'

    # Combine
    df_combined = pd.concat([df_cn, df_mci_stable, df_mci_to_ad, df_ad], ignore_index=True)

    logger.info(f"\nClass distribution:")
    for label, name in [(0, 'CN'), (1, 'MCI-stable'), (2, 'MCI→AD'), (3, 'AD')]:
        count = len(df_combined[df_combined['label'] == label])
        pct = count / len(df_combined) * 100
        logger.info(f"  {name:12}: {count:4} ({pct:5.1f}%)")

    return df_combined


def map_to_mri_scans(df, skull_dir):
    """
    Map patients to their MRI scans

    Args:
        df: DataFrame with PTID, features, label
        skull_dir: Path to ADNI-skull directory

    Returns:
        DataFrame with scan_path column added
    """
    logger.info(f"\nMapping to MRI scans in {skull_dir}...")

    skull_path = Path(skull_dir)
    if not skull_path.exists():
        logger.warning(f"⚠️  Skull directory not found: {skull_dir}")
        logger.warning("    Creating placeholder paths for testing")

    scan_records = []
    missing_count = 0

    for _, row in df.iterrows():
        ptid = row['PTID']
        patient_folder = skull_path / ptid

        if not patient_folder.exists():
            # Create placeholder path for testing
            scan_path = f"{skull_dir}/{ptid}/placeholder_registered_skull_stripped.nii.gz"
            missing_count += 1
        else:
            # Find MRI scan
            scans = list(patient_folder.glob('*_registered_skull_stripped.nii.gz'))
            if not scans:
                scan_path = f"{skull_dir}/{ptid}/placeholder_registered_skull_stripped.nii.gz"
                missing_count += 1
            else:
                scan_path = str(scans[0])

        scan_records.append({
            'PTID': ptid,
            'RID': row.get('RID', ''),
            'PHASE': row.get('PHASE', ''),
            'group': row['group'],
            'label': row['label'],
            'scan_path': scan_path,
            **{col: row[col] for col in df.columns if col not in ['PTID', 'RID', 'PHASE', 'group', 'label']}
        })

    df_with_mri = pd.DataFrame(scan_records)

    logger.info(f"  Total patients: {len(df_with_mri)}")
    logger.info(f"  Missing/placeholder scans: {missing_count}")
    logger.info(f"  Real scans: {len(df_with_mri) - missing_count}")

    return df_with_mri


def create_vkola_format(df):
    """
    Convert to vkola-lab CSV format

    Required columns:
        - filename: MRI filename
        - PTID: Patient ID
        - Features: 30 tabular features
        - COG: 4-class label (0=CN, 1=MCI-stable, 2=MCI→AD, 3=AD)
        - ADD: Binary AD label (0=Non-AD, 1=AD)
        - scan_path: Full path to MRI
    """

    # Extract filename from scan_path
    df['filename'] = df['scan_path'].apply(lambda x: Path(x).name)

    # Create COG and ADD labels
    df['COG'] = df['label']  # Keep original label column too

    # ADD: 0=Non-AD (CN + MCI-stable), 1=AD (MCI→AD + AD)
    df['ADD'] = df['COG'].apply(lambda x: 0 if x in [0, 1] else 1)

    # Feature columns (30 features)
    feature_cols = [
        # Demographics (6)
        'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY',
        # Physical (3)
        'VSWEIGHT', 'VSHEIGHT', 'BMI',
        # Medical history (9)
        'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
        'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
        # Cognitive scores (9)
        'MMSCORE', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
        'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
        # Clinical assessments (3)
        'CDGLOBAL', 'BCFAQ', 'BCDEPRES'
    ]

    # Select columns
    cols_ordered = ['filename', 'PTID', 'RID', 'PHASE'] + feature_cols + ['label', 'COG', 'ADD', 'scan_path', 'group']

    # Keep only existing columns
    cols_existing = [col for col in cols_ordered if col in df.columns]

    df_vkola = df[cols_existing].copy()

    # Check missing features
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"⚠️  Missing features: {missing_features}")

    return df_vkola


def create_patient_level_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create patient-level stratified splits

    Ensures all scans from same patient stay in same split
    """
    logger.info("\nCreating patient-level splits...")

    # Get one row per patient (first visit)
    df_patients = df.groupby('PTID').first().reset_index()

    # Stratified split by label
    y = df_patients['label'].values
    groups = df_patients['PTID'].values

    # Train+Val vs Test
    train_val_ratio = train_ratio + val_ratio
    splitter_test = GroupShuffleSplit(n_splits=1, train_size=train_val_ratio, random_state=seed)
    train_val_idx, test_idx = next(splitter_test.split(df_patients, y, groups))

    df_train_val = df_patients.iloc[train_val_idx]
    df_test = df_patients.iloc[test_idx]

    # Train vs Val
    val_ratio_adjusted = val_ratio / train_val_ratio
    y_train_val = df_train_val['label'].values
    groups_train_val = df_train_val['PTID'].values

    splitter_val = GroupShuffleSplit(n_splits=1, train_size=1-val_ratio_adjusted, random_state=seed)
    train_idx, val_idx = next(splitter_val.split(df_train_val, y_train_val, groups_train_val))

    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    # Get PTIDs for each split
    train_ptids = df_train['PTID'].tolist()
    val_ptids = df_val['PTID'].tolist()
    test_ptids = df_test['PTID'].tolist()

    # Filter full dataset by PTID
    df_train_full = df[df['PTID'].isin(train_ptids)].copy()
    df_val_full = df[df['PTID'].isin(val_ptids)].copy()
    df_test_full = df[df['PTID'].isin(test_ptids)].copy()

    logger.info(f"\nSplit results:")
    logger.info(f"  Train: {len(df_train_full)} scans, {len(train_ptids)} patients")
    logger.info(f"  Val:   {len(df_val_full)} scans, {len(val_ptids)} patients")
    logger.info(f"  Test:  {len(df_test_full)} scans, {len(test_ptids)} patients")

    # Print class distribution per split
    for split_name, split_df in [('Train', df_train_full), ('Val', df_val_full), ('Test', df_test_full)]:
        logger.info(f"\n{split_name} class distribution:")
        for label, name in [(0, 'CN'), (1, 'MCI-stable'), (2, 'MCI→AD'), (3, 'AD')]:
            count = len(split_df[split_df['label'] == label])
            pct = count / len(split_df) * 100 if len(split_df) > 0 else 0
            logger.info(f"  {name:12}: {count:4} ({pct:5.1f}%)")

    return df_train_full, df_val_full, df_test_full


def main():
    parser = argparse.ArgumentParser(description='Prepare 4-class MRI+Tabular dataset')
    parser.add_argument('--clinical-csv', type=str,
                        default='../../data/clinical_data_all_groups.csv',
                        help='Clinical data CSV (all groups)')
    parser.add_argument('--converters-csv', type=str,
                        default='../../data/AD_CN_MCI_to_AD.csv',
                        help='MCI converters CSV')
    parser.add_argument('--skull-dir', type=str,
                        default='/Volumes/KINGSTON/ADNI-skull',
                        help='Directory with skull-stripped MRI')
    parser.add_argument('--output-dir', type=str,
                        default='lookupcsv/CrossValid',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PREPARE 4-CLASS MRI+TABULAR DATASET FOR VKOLA-LAB")
    logger.info("=" * 80)

    # Load 4-class data
    df = load_4class_data(args.clinical_csv, args.converters_csv)

    # Map to MRI scans
    df_with_mri = map_to_mri_scans(df, args.skull_dir)

    # Create vkola-lab format
    df_vkola = create_vkola_format(df_with_mri)

    # Create splits
    df_train, df_val, df_test = create_patient_level_splits(df_vkola, seed=args.seed)

    # Save CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'valid.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)

    logger.info(f"\n✓ Saved to {output_dir}/")
    logger.info(f"  train.csv: {len(df_train)} samples")
    logger.info(f"  valid.csv: {len(df_val)} samples")
    logger.info(f"  test.csv: {len(df_test)} samples")

    # Preview
    logger.info(f"\nPreview of train.csv (first 3 rows):")
    print(df_train.head(3)[['PTID', 'filename', 'COG', 'ADD']].to_string())

    logger.info("\n" + "=" * 80)
    logger.info("✓ DATASET PREPARATION COMPLETED!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
