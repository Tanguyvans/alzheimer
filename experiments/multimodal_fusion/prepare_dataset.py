#!/usr/bin/env python3
"""
Prepare Multi-Modal Dataset

Creates train/val/test splits with both MRI paths and tabular features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_multimodal_dataset(
    clinical_csv: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Prepare multi-modal dataset from clinical CSV

    Args:
        clinical_csv: Path to clinical CSV with MRI paths and features
        output_dir: Output directory for train/val/test CSVs
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
    """
    logger.info(f"Loading clinical data from {clinical_csv}")
    df = pd.read_csv(clinical_csv)

    logger.info(f"Total rows: {len(df)}")

    # Check MRI path column
    mri_col = 'nii_path' if 'nii_path' in df.columns else 'scan_path'
    if mri_col not in df.columns:
        raise ValueError(f"No MRI path column found. Available: {df.columns.tolist()}")

    # Rename to standard column name
    df = df.rename(columns={mri_col: 'scan_path'})

    # Check which MRI files exist
    logger.info("Checking MRI file availability...")
    df['mri_exists'] = df['scan_path'].apply(lambda x: Path(x).exists() if pd.notna(x) else False)
    available = df['mri_exists'].sum()
    logger.info(f"  Available: {available}/{len(df)} ({100*available/len(df):.1f}%)")

    # Filter to existing files only
    df = df[df['mri_exists']].copy()
    logger.info(f"Using {len(df)} samples with available MRI")

    # Map classes to CN vs AD-trajectory
    class_col = 'CLASS_4' if 'CLASS_4' in df.columns else 'Group'

    if class_col == 'CLASS_4':
        # Map CLASS_4 labels
        class_mapping = {
            'CN': 0,
            'MCI_stable': None,  # Exclude
            'MCI_to_AD': 1,      # AD trajectory
            'AD': 1              # AD trajectory
        }
    else:
        # Map Group labels
        class_mapping = {
            'CN': 0,
            'MCIs': None,  # Exclude stable MCI
            'MCIc': 1,     # Converters -> AD trajectory
            'AD': 1
        }

    df['label'] = df[class_col].map(class_mapping)

    # Remove samples without valid labels (MCIs)
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    logger.info(f"\nClass distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        name = 'CN' if label == 0 else 'AD_trajectory'
        logger.info(f"  {name}: {count} ({100*count/len(df):.1f}%)")

    # Keep one sample per subject - prefer 'sc' (screening) visits which have tabular data
    if 'VISCODE' in df.columns:
        # Prefer screening (sc) visits which have clinical scores (AGE, MMSCORE, etc.)
        # bl visits are missing these values
        df['visit_priority'] = df['VISCODE'].apply(lambda x: 0 if x == 'sc' else 1)
        df = df.sort_values(['Subject', 'visit_priority'])
        df = df.drop_duplicates(subset=['Subject'], keep='first')
        df = df.drop(columns=['visit_priority'])
        logger.info(f"After deduplication: {len(df)} samples")

    # Filter to samples with complete key tabular features
    key_features = ['AGE', 'MMSCORE', 'CDGLOBAL', 'PTEDUCAT']
    available_features = [f for f in key_features if f in df.columns]
    if available_features:
        before = len(df)
        df = df.dropna(subset=available_features)
        logger.info(f"After filtering for complete tabular data: {len(df)} samples (removed {before - len(df)})")

    logger.info(f"\nFinal class distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        name = 'CN' if label == 0 else 'AD_trajectory'
        logger.info(f"  {name}: {count} ({100*count/len(df):.1f}%)")

    # Stratified split
    logger.info(f"\nSplitting data ({train_ratio}/{val_ratio}/{test_ratio})...")

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=seed
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['label'],
        random_state=seed
    )

    logger.info(f"  Train: {len(train_df)}")
    logger.info(f"  Val: {len(val_df)}")
    logger.info(f"  Test: {len(test_df)}")

    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'CN': int((df['label'] == 0).sum()),
            'AD_trajectory': int((df['label'] == 1).sum())
        },
        'tabular_features_available': [
            col for col in ['AGE', 'PTGENDER', 'PTEDUCAT', 'MMSCORE', 'CDGLOBAL',
                           'CLOCKSCOR', 'CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'BMI']
            if col in df.columns
        ],
        'seed': seed
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDataset saved to {output_dir}")
    logger.info(f"Available tabular features: {metadata['tabular_features_available']}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare Multi-Modal Dataset')
    parser.add_argument('--clinical-csv', type=str,
                       default='../../data/adni/ALL_4class_clinical.csv',
                       help='Path to clinical CSV')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    prepare_multimodal_dataset(
        clinical_csv=args.clinical_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
