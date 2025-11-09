#!/usr/bin/env python3
"""
Create train/val/test splits for cn_mci_ad (NPPY) dataset
Filters to CN vs AD only (binary classification)

This script scans the cn_mci_ad directory for brain scans and creates
patient-level splits ensuring no patient appears in multiple splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_nppy_dataset(nppy_dir: Path, dxsum_csv: Path):
    """
    Scan cn_mci_ad directory and match with diagnosis info from dxsum.csv

    Args:
        nppy_dir: Path to cn_mci_ad directory
        dxsum_csv: Path to dxsum.csv with diagnosis information

    Returns:
        DataFrame with scan_path, patient_id, label
    """
    logger.info(f"Scanning NPPY dataset: {nppy_dir}")
    logger.info(f"Loading diagnosis info: {dxsum_csv}")

    # Load diagnosis info
    dx_df = pd.read_csv(dxsum_csv)
    logger.info(f"Loaded {len(dx_df)} diagnosis records")

    # Find all .nii.gz files in cn_mci_ad directory
    scan_files = list(nppy_dir.rglob("*.nii.gz"))
    logger.info(f"Found {len(scan_files)} .nii.gz files")

    # Extract patient IDs and create dataframe
    data = []
    for scan_path in scan_files:
        # Extract patient ID from path
        # Assuming structure: cn_mci_ad/XXX_S_XXXX/scan.nii.gz
        patient_id = scan_path.parent.name

        if not patient_id.startswith(('011_S', '022_S', '023_S', '127_S', '128_S')):
            logger.warning(f"Unexpected patient ID format: {patient_id}")
            continue

        data.append({
            'scan_path': str(scan_path),
            'patient_id': patient_id
        })

    scans_df = pd.DataFrame(data)
    logger.info(f"Processed {len(scans_df)} scans with valid patient IDs")

    # Match with diagnosis info
    # Merge on patient_id to get diagnosis labels
    merged_df = scans_df.merge(
        dx_df[['RID', 'DX']].drop_duplicates(subset=['RID']),
        left_on='patient_id',
        right_on='RID',
        how='left'
    )

    # Map DX to labels (CN=0, MCI=1, AD=2)
    label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    merged_df['label'] = merged_df['DX'].map(label_map)

    # Filter out rows without diagnosis
    merged_df = merged_df.dropna(subset=['label'])
    merged_df['label'] = merged_df['label'].astype(int)

    logger.info(f"\nDiagnosis distribution:")
    logger.info(f"  CN: {len(merged_df[merged_df['label'] == 0])}")
    logger.info(f"  MCI: {len(merged_df[merged_df['label'] == 1])}")
    logger.info(f"  AD: {len(merged_df[merged_df['label'] == 2])}")

    return merged_df[['scan_path', 'patient_id', 'label']]


def filter_cn_ad(df: pd.DataFrame):
    """Filter to keep only CN (0) and AD (2), relabel AD to 1"""
    logger.info("\nFiltering to CN vs AD binary classification...")

    # Keep only CN and AD
    df_binary = df[df['label'].isin([0, 2])].copy()

    # Relabel AD from 2 to 1
    df_binary.loc[df_binary['label'] == 2, 'label'] = 1

    logger.info(f"Binary dataset: {len(df_binary)} samples")
    logger.info(f"  CN (0): {len(df_binary[df_binary['label'] == 0])}")
    logger.info(f"  AD (1): {len(df_binary[df_binary['label'] == 1])}")

    return df_binary


def create_patient_level_splits(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create train/val/test splits at patient level to prevent data leakage

    Args:
        df: DataFrame with scan_path, patient_id, label
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_seed: Random seed for reproducibility
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    np.random.seed(random_seed)

    # Get unique patients per class
    cn_patients = df[df['label'] == 0]['patient_id'].unique()
    ad_patients = df[df['label'] == 1]['patient_id'].unique()

    logger.info(f"\nUnique patients:")
    logger.info(f"  CN: {len(cn_patients)}")
    logger.info(f"  AD: {len(ad_patients)}")

    # Shuffle patients
    np.random.shuffle(cn_patients)
    np.random.shuffle(ad_patients)

    # Split patients
    def split_patients(patients):
        n = len(patients)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return {
            'train': patients[:train_end],
            'val': patients[train_end:val_end],
            'test': patients[val_end:]
        }

    cn_splits = split_patients(cn_patients)
    ad_splits = split_patients(ad_patients)

    # Create train/val/test dataframes
    train_df = df[df['patient_id'].isin(np.concatenate([cn_splits['train'], ad_splits['train']]))]
    val_df = df[df['patient_id'].isin(np.concatenate([cn_splits['val'], ad_splits['val']]))]
    test_df = df[df['patient_id'].isin(np.concatenate([cn_splits['test'], ad_splits['test']]))]

    logger.info(f"\nSplit statistics:")
    logger.info(f"Train: {len(train_df)} samples (CN: {len(train_df[train_df['label']==0])}, AD: {len(train_df[train_df['label']==1])})")
    logger.info(f"Val:   {len(val_df)} samples (CN: {len(val_df[val_df['label']==0])}, AD: {len(val_df[val_df['label']==1])})")
    logger.info(f"Test:  {len(test_df)} samples (CN: {len(test_df[test_df['label']==0])}, AD: {len(test_df[test_df['label']==1])})")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare CN vs AD splits from NPPY dataset')
    parser.add_argument('--nppy-dir', type=str, required=True,
                        help='Path to cn_mci_ad directory')
    parser.add_argument('--dxsum-csv', type=str, required=True,
                        help='Path to dxsum.csv with diagnosis info')
    parser.add_argument('--output-dir', type=str, default='data/splits_nppy',
                        help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Scan dataset
    df = scan_nppy_dataset(Path(args.nppy_dir), Path(args.dxsum_csv))

    # Filter to binary CN vs AD
    df_binary = filter_cn_ad(df)

    # Create patient-level splits
    train_df, val_df, test_df = create_patient_level_splits(
        df_binary,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df[['scan_path', 'label']].to_csv(output_dir / 'train.csv', index=False)
    val_df[['scan_path', 'label']].to_csv(output_dir / 'val.csv', index=False)
    test_df[['scan_path', 'label']].to_csv(output_dir / 'test.csv', index=False)

    logger.info(f"\n✓ Saved splits to {output_dir}/")
    logger.info(f"  - train.csv")
    logger.info(f"  - val.csv")
    logger.info(f"  - test.csv")


if __name__ == "__main__":
    main()
