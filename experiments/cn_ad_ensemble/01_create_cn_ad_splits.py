#!/usr/bin/env python3
"""
Create CN vs AD binary classification splits from 3-class data
Filters out MCI samples to create a cleaner 2-class problem
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_cn_ad(csv_path: Path, output_path: Path):
    """
    Filter CSV to keep only CN (label=0) and AD (label=2) samples
    Relabel AD from 2 to 1 for binary classification

    Args:
        csv_path: Path to 3-class CSV (with CN=0, MCI=1, AD=2)
        output_path: Path to save filtered 2-class CSV (with CN=0, AD=1)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    logger.info(f"Original dataset: {len(df)} samples")
    logger.info(f"  CN (0): {len(df[df['label'] == 0])}")
    logger.info(f"  MCI (1): {len(df[df['label'] == 1])}")
    logger.info(f"  AD (2): {len(df[df['label'] == 2])}")

    # Filter to keep only CN and AD
    df_filtered = df[df['label'].isin([0, 2])].copy()

    # Relabel AD from 2 to 1 for binary classification
    df_filtered.loc[df_filtered['label'] == 2, 'label'] = 1

    logger.info(f"\nFiltered dataset: {len(df_filtered)} samples")
    logger.info(f"  CN (0): {len(df_filtered[df_filtered['label'] == 0])}")
    logger.info(f"  AD (1): {len(df_filtered[df_filtered['label'] == 1])}")

    # Save filtered CSV
    df_filtered.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Create CN vs AD splits from 3-class data')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to 3-class train.csv')
    parser.add_argument('--val-csv', type=str, required=True,
                        help='Path to 3-class val.csv')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to 3-class test.csv')
    parser.add_argument('--output-dir', type=str, default='data/splits',
                        help='Output directory for CN/AD splits')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Creating CN vs AD splits")
    logger.info("=" * 60)

    # Process each split
    logger.info("\n--- Training Set ---")
    train_df = filter_cn_ad(Path(args.train_csv), output_dir / 'train.csv')

    logger.info("\n--- Validation Set ---")
    val_df = filter_cn_ad(Path(args.val_csv), output_dir / 'val.csv')

    logger.info("\n--- Test Set ---")
    test_df = filter_cn_ad(Path(args.test_csv), output_dir / 'test.csv')

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_df)} samples (CN: {len(train_df[train_df['label']==0])}, AD: {len(train_df[train_df['label']==1])})")
    logger.info(f"Val:   {len(val_df)} samples (CN: {len(val_df[val_df['label']==0])}, AD: {len(val_df[val_df['label']==1])})")
    logger.info(f"Test:  {len(test_df)} samples (CN: {len(test_df[test_df['label']==0])}, AD: {len(test_df[test_df['label']==1])})")
    logger.info(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

    # Class balance
    total_cn = len(train_df[train_df['label']==0]) + len(val_df[val_df['label']==0]) + len(test_df[test_df['label']==0])
    total_ad = len(train_df[train_df['label']==1]) + len(val_df[val_df['label']==1]) + len(test_df[test_df['label']==1])
    logger.info(f"\nOverall class distribution:")
    logger.info(f"  CN: {total_cn} ({100*total_cn/(total_cn+total_ad):.1f}%)")
    logger.info(f"  AD: {total_ad} ({100*total_ad/(total_cn+total_ad):.1f}%)")

    logger.info(f"\nSplits saved to {output_dir}/")


if __name__ == "__main__":
    main()
