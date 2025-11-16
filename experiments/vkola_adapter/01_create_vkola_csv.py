#!/usr/bin/env python3
"""
Create vkola-lab format CSV from xgboost_4class splits

This script converts your existing tabular data + MRI paths
into the format expected by vkola-lab/ncomms2022

Input:
    - experiments/xgboost_4class/data/splits/*.csv (tabular features + labels)
    - experiments/cn_vs_ad_baseline/data/splits/*.csv (imaging paths)

Output:
    - lookupcsv/CrossValid/train.csv
    - lookupcsv/CrossValid/valid.csv
    - lookupcsv/CrossValid/test.csv

Format:
    filename,PTID,AGE,PTGENDER,...,COG,ADD
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tabular_splits(tabular_dir):
    """Load tabular splits (features + labels)"""
    train_tab = pd.read_csv(f"{tabular_dir}/train.csv")
    val_tab = pd.read_csv(f"{tabular_dir}/val.csv")
    test_tab = pd.read_csv(f"{tabular_dir}/test.csv")

    logger.info(f"Loaded tabular splits:")
    logger.info(f"  Train: {len(train_tab)} samples")
    logger.info(f"  Val:   {len(val_tab)} samples")
    logger.info(f"  Test:  {len(test_tab)} samples")

    return train_tab, val_tab, test_tab


def load_imaging_splits(imaging_dir):
    """Load imaging splits (PTID → scan_path mapping)"""
    train_img = pd.read_csv(f"{imaging_dir}/train.csv")
    val_img = pd.read_csv(f"{imaging_dir}/val.csv")
    test_img = pd.read_csv(f"{imaging_dir}/test.csv")

    # Keep only PTID and scan_path
    cols = ['PTID', 'scan_path']
    train_img = train_img[cols]
    val_img = val_img[cols]
    test_img = test_img[cols]

    logger.info(f"Loaded imaging splits:")
    logger.info(f"  Train: {len(train_img)} samples with MRI")
    logger.info(f"  Val:   {len(val_img)} samples with MRI")
    logger.info(f"  Test:  {len(test_img)} samples with MRI")

    return train_img, val_img, test_img


def merge_tabular_imaging(df_tabular, df_imaging):
    """Merge tabular features with imaging paths"""
    merged = df_tabular.merge(df_imaging, on='PTID', how='inner')

    logger.info(f"Merge results:")
    logger.info(f"  Tabular samples: {len(df_tabular)}")
    logger.info(f"  Imaging samples: {len(df_imaging)}")
    logger.info(f"  Merged samples:  {len(merged)}")
    logger.info(f"  Missing MRI:     {len(df_tabular) - len(merged)}")

    if len(merged) < len(df_tabular):
        logger.warning(f"⚠️  {len(df_tabular) - len(merged)} samples have no MRI!")

    return merged


def create_vkola_format(df):
    """
    Convert to vkola-lab format

    Required columns:
        - filename: MRI filename (extracted from scan_path)
        - PTID: Patient ID
        - Features: AGE, PTGENDER, PTEDUCAT, etc. (30 features)
        - COG: Cognitive status (0=NC, 1=MCI, 2=AD, 3=nADD)
        - ADD: Alzheimer's disease (0=Non-AD, 1=AD) [optional]
    """

    # Extract filename from scan_path
    df['filename'] = df['scan_path'].apply(lambda x: Path(x).name)

    # Create ADD label from COG
    # COG: 0=CN, 1=MCI-stable, 2=MCI→AD, 3=AD
    # ADD: 0=Non-AD (CN + MCI-stable), 1=AD (MCI→AD + AD)
    def cog_to_add(cog):
        if cog in [0, 1]:  # CN or MCI-stable
            return 0  # Non-AD
        else:  # MCI→AD or AD
            return 1  # AD

    df['COG'] = df['label']  # Rename label → COG
    df['ADD'] = df['COG'].apply(cog_to_add)

    # Reorder columns: filename, PTID, features, COG, ADD
    feature_cols = [
        # Demographics (6 features)
        'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTHAND', 'PTMARRY',
        # Physical (3 features)
        'VSWEIGHT', 'VSHEIGHT', 'BMI',
        # Medical history (9 features)
        'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
        'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
        # Cognitive scores (9 features)
        'MMSCORE', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
        'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
        # Clinical assessments (3 features)
        'CDGLOBAL', 'BCFAQ', 'BCDEPRES'
    ]

    # Check if all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"⚠️  Missing features: {missing_features}")

    # Select columns in order
    cols_ordered = ['filename', 'PTID'] + feature_cols + ['COG', 'ADD', 'scan_path']

    # Keep only existing columns
    cols_existing = [col for col in cols_ordered if col in df.columns]

    df_vkola = df[cols_existing].copy()

    return df_vkola


def print_class_distribution(df, split_name):
    """Print class distribution"""
    logger.info(f"\nClass distribution in {split_name}:")

    cog_counts = df['COG'].value_counts().sort_index()
    total = len(df)

    class_names = {0: 'CN', 1: 'MCI-stable', 2: 'MCI→AD', 3: 'AD'}

    for cog, count in cog_counts.items():
        pct = count / total * 100
        logger.info(f"  {class_names[cog]:12} ({cog}): {count:4} samples ({pct:5.1f}%)")

    add_counts = df['ADD'].value_counts().sort_index()
    logger.info(f"\nADD distribution:")
    logger.info(f"  Non-AD (0): {add_counts[0]} samples ({add_counts[0]/total*100:.1f}%)")
    logger.info(f"  AD (1):     {add_counts[1]} samples ({add_counts[1]/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Create vkola-lab format CSV')
    parser.add_argument('--tabular-dir', type=str,
                        default='../xgboost_4class/data/splits',
                        help='Directory with tabular splits')
    parser.add_argument('--imaging-dir', type=str,
                        default='../cn_vs_ad_baseline/data/splits',
                        help='Directory with imaging splits')
    parser.add_argument('--output-dir', type=str,
                        default='lookupcsv/CrossValid',
                        help='Output directory for vkola-lab CSV')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CREATE VKOLA-LAB FORMAT CSV")
    logger.info("=" * 80)

    # Load splits
    logger.info("\n1. Loading tabular splits...")
    train_tab, val_tab, test_tab = load_tabular_splits(args.tabular_dir)

    logger.info("\n2. Loading imaging splits...")
    train_img, val_img, test_img = load_imaging_splits(args.imaging_dir)

    # Merge tabular + imaging
    logger.info("\n3. Merging tabular + imaging...")
    train_merged = merge_tabular_imaging(train_tab, train_img)
    val_merged = merge_tabular_imaging(val_tab, val_img)
    test_merged = merge_tabular_imaging(test_tab, test_img)

    # Convert to vkola-lab format
    logger.info("\n4. Converting to vkola-lab format...")
    train_vkola = create_vkola_format(train_merged)
    val_vkola = create_vkola_format(val_merged)
    test_vkola = create_vkola_format(test_merged)

    # Print statistics
    print_class_distribution(train_vkola, 'train')
    print_class_distribution(val_vkola, 'valid')
    print_class_distribution(test_vkola, 'test')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    logger.info(f"\n5. Saving vkola-lab CSV to {output_dir}/")
    train_vkola.to_csv(output_dir / 'train.csv', index=False)
    val_vkola.to_csv(output_dir / 'valid.csv', index=False)
    test_vkola.to_csv(output_dir / 'test.csv', index=False)

    logger.info(f"✓ train.csv: {len(train_vkola)} samples")
    logger.info(f"✓ valid.csv: {len(val_vkola)} samples")
    logger.info(f"✓ test.csv:  {len(test_vkola)} samples")

    # Preview
    logger.info("\n6. Preview of train.csv (first 3 rows):")
    print(train_vkola.head(3).to_string())

    logger.info("\n" + "=" * 80)
    logger.info("✓ VKOLA-LAB CSV CREATED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Copy these CSV to vkola-lab repo:")
    logger.info(f"   cp {output_dir}/*.csv ncomms2022/lookupcsv/CrossValid/")
    logger.info("2. Verify MRI paths are accessible on other machine")
    logger.info("3. Run vkola-lab training")


if __name__ == '__main__':
    main()
