#!/usr/bin/env python3
"""
Create balanced CN vs AD subsets for ensemble training
Similar to IMBALMED approach but for binary classification
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_balanced_subsets(
    train_csv: Path,
    output_dir: Path,
    num_subsets: int = 5,
    random_seed: int = 42
):
    """
    Create balanced CN/AD subsets for ensemble diversity

    Args:
        train_csv: Path to training CSV with CN=0, AD=1
        output_dir: Directory to save subset CSVs
        num_subsets: Number of balanced subsets to create
        random_seed: Random seed for reproducibility
    """
    # Read training data
    df = pd.read_csv(train_csv)

    # Get class counts
    cn_count = len(df[df['label'] == 0])
    ad_count = len(df[df['label'] == 1])

    logger.info(f"Training set: {len(df)} samples")
    logger.info(f"  CN: {cn_count}")
    logger.info(f"  AD: {ad_count}")

    # Determine subset size (use minority class)
    subset_size = min(cn_count, ad_count)
    logger.info(f"\nCreating {num_subsets} balanced subsets")
    logger.info(f"  Samples per class: {subset_size}")
    logger.info(f"  Total per subset: {2 * subset_size}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get samples by class
    cn_samples = df[df['label'] == 0].copy()
    ad_samples = df[df['label'] == 1].copy()

    # Create subsets
    np.random.seed(random_seed)

    for i in range(num_subsets):
        logger.info(f"\n--- Subset {i+1} ---")

        # Sample from each class
        cn_subset = cn_samples.sample(n=subset_size, random_state=random_seed + i, replace=False)
        ad_subset = ad_samples.sample(n=subset_size, random_state=random_seed + i, replace=False)

        # Combine and shuffle
        subset_df = pd.concat([cn_subset, ad_subset], ignore_index=True)
        subset_df = subset_df.sample(frac=1, random_state=random_seed + i).reset_index(drop=True)

        # Verify balance
        cn_in_subset = len(subset_df[subset_df['label'] == 0])
        ad_in_subset = len(subset_df[subset_df['label'] == 1])

        logger.info(f"  CN: {cn_in_subset}, AD: {ad_in_subset}")
        logger.info(f"  Total: {len(subset_df)}")

        # Save subset
        output_path = output_dir / f'train_subset_{i+1}.csv'
        subset_df.to_csv(output_path, index=False)
        logger.info(f"  Saved to {output_path}")

    logger.info(f"\n✓ Created {num_subsets} balanced subsets in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Create balanced CN/AD subsets for ensemble')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to CN/AD train.csv')
    parser.add_argument('--output-dir', type=str, default='balanced_subsets',
                        help='Output directory for subset CSVs')
    parser.add_argument('--num-subsets', type=int, default=5,
                        help='Number of balanced subsets to create')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    create_balanced_subsets(
        train_csv=Path(args.train_csv),
        output_dir=Path(args.output_dir),
        num_subsets=args.num_subsets,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
