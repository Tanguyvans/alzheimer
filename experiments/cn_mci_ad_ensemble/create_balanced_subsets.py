#!/usr/bin/env python3
"""
Create balanced subsets for IMBALMED-style ensemble training

This script:
1. Loads the training CSV
2. Creates multiple balanced subsets by downsampling to minority class
3. Saves each subset as a separate CSV for ensemble training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_balanced_subsets(
    train_csv: str,
    output_dir: str,
    num_subsets: int = 5,
    random_seed: int = 42
):
    """
    Create balanced training subsets

    Args:
        train_csv: Path to training CSV
        output_dir: Directory to save subset CSVs
        num_subsets: Number of balanced subsets to create
        random_seed: Random seed for reproducibility
    """
    logger.info("="*80)
    logger.info("CREATING BALANCED SUBSETS FOR ENSEMBLE")
    logger.info("="*80)

    # Load training data
    df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(df)} training samples from {train_csv}")

    # Check column names
    path_col = 'scan_path' if 'scan_path' in df.columns else 'path'

    # Get class distribution
    label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    logger.info("\nOriginal class distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {label_names[label]}: {count} samples ({pct:.1f}%)")

    # Find minority class size
    class_counts = df['label'].value_counts()
    min_class_size = class_counts.min()
    logger.info(f"\nMinority class size: {min_class_size}")
    logger.info(f"Balanced subset size: {min_class_size * 3} samples (equal class distribution)")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create balanced subsets
    np.random.seed(random_seed)

    for i in range(num_subsets):
        logger.info(f"\nCreating subset {i+1}/{num_subsets}...")

        subset_dfs = []

        for label in sorted(df['label'].unique()):
            # Get all samples for this class
            class_df = df[df['label'] == label].copy()

            # Randomly sample min_class_size samples
            if len(class_df) > min_class_size:
                sampled_df = class_df.sample(n=min_class_size, random_state=random_seed + i)
            else:
                sampled_df = class_df

            subset_dfs.append(sampled_df)

            logger.info(f"  {label_names[label]}: {len(sampled_df)} samples")

        # Combine all classes
        subset_df = pd.concat(subset_dfs, ignore_index=True)

        # Shuffle
        subset_df = subset_df.sample(frac=1, random_state=random_seed + i).reset_index(drop=True)

        # Save to CSV
        subset_path = output_dir / f'train_subset_{i+1}.csv'
        subset_df.to_csv(subset_path, index=False)
        logger.info(f"  Saved to {subset_path}")

    logger.info("\n" + "="*80)
    logger.info(f"✓ Created {num_subsets} balanced subsets")
    logger.info(f"  Each subset: {min_class_size * 3} samples ({min_class_size} per class)")
    logger.info(f"  Saved to: {output_dir}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Create balanced subsets for ensemble training')
    parser.add_argument('--train-csv', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--output-dir', type=str, default='balanced_subsets',
                       help='Output directory for subset CSVs')
    parser.add_argument('--num-subsets', type=int, default=5,
                       help='Number of balanced subsets to create')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    create_balanced_subsets(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        num_subsets=args.num_subsets,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
