#!/usr/bin/env python3
"""
Step 0: Check dataset for corrupted/empty files before training.

Validates all MRI files in pairs.csv and creates a cleaned version
with only valid pairs.

Usage:
    python 00_check_dataset.py --config config.yaml
"""

import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = EXPERIMENT_DIR / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_nifti_file(filepath: str) -> tuple[bool, str]:
    """
    Check if a NIfTI file is valid and loadable.

    Returns:
        (is_valid, error_message)
    """
    try:
        if not Path(filepath).exists():
            return False, "File not found"

        if Path(filepath).stat().st_size == 0:
            return False, "Empty file (0 bytes)"

        if not HAS_NIBABEL:
            return False, "nibabel not installed"

        # Try to load the file
        img = nib.load(filepath)
        data = img.get_fdata()

        # Check for valid data
        if data.size == 0:
            return False, "Empty data array"

        if np.isnan(data).all():
            return False, "All NaN values"

        if data.max() == data.min() == 0:
            return False, "All zero values"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Check dataset for corrupted files')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--fix', action='store_true', help='Create cleaned pairs.csv without corrupted files')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    pairs_dir = EXPERIMENT_DIR / config['data']['pairs_dir']
    pairs_csv = pairs_dir / 'pairs.csv'

    if not pairs_csv.exists():
        logger.error(f"Pairs file not found: {pairs_csv}")
        logger.error("Run 01_prepare_pairs.py first")
        return

    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Checking {len(pairs_df)} pairs...")

    # Check each pair
    corrupted_baseline = []
    corrupted_followup = []
    valid_pairs = []

    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Validating files"):
        baseline_ok, baseline_err = check_nifti_file(row['baseline_path'])
        followup_ok, followup_err = check_nifti_file(row['followup_path'])

        if not baseline_ok:
            corrupted_baseline.append({
                'idx': idx,
                'ptid': row['ptid'],
                'path': row['baseline_path'],
                'error': baseline_err
            })

        if not followup_ok:
            corrupted_followup.append({
                'idx': idx,
                'ptid': row['ptid'],
                'path': row['followup_path'],
                'error': followup_err
            })

        if baseline_ok and followup_ok:
            valid_pairs.append(idx)

    # Report results
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal pairs: {len(pairs_df)}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Corrupted pairs: {len(pairs_df) - len(valid_pairs)}")

    if corrupted_baseline:
        print(f"\n❌ Corrupted BASELINE files ({len(corrupted_baseline)}):")
        for item in corrupted_baseline[:10]:  # Show first 10
            print(f"  - {item['ptid']}: {item['error']}")
            print(f"    {item['path']}")
        if len(corrupted_baseline) > 10:
            print(f"  ... and {len(corrupted_baseline) - 10} more")

    if corrupted_followup:
        print(f"\n❌ Corrupted FOLLOWUP files ({len(corrupted_followup)}):")
        for item in corrupted_followup[:10]:  # Show first 10
            print(f"  - {item['ptid']}: {item['error']}")
            print(f"    {item['path']}")
        if len(corrupted_followup) > 10:
            print(f"  ... and {len(corrupted_followup) - 10} more")

    # Save cleaned version if requested
    if args.fix and len(valid_pairs) < len(pairs_df):
        cleaned_df = pairs_df.loc[valid_pairs].reset_index(drop=True)

        # Backup original
        backup_path = pairs_dir / 'pairs_original.csv'
        if not backup_path.exists():
            pairs_df.to_csv(backup_path, index=False)
            logger.info(f"Backed up original to {backup_path}")

        # Save cleaned
        cleaned_df.to_csv(pairs_csv, index=False)
        logger.info(f"Saved cleaned pairs.csv with {len(cleaned_df)} valid pairs")

        # Update class distribution
        print(f"\nCleaned dataset:")
        print(f"  Total: {len(cleaned_df)} pairs")
        if 'is_converter' in cleaned_df.columns:
            converters = cleaned_df['is_converter'].sum()
            print(f"  Converters: {converters}")
            print(f"  Non-converters: {len(cleaned_df) - converters}")

    elif len(valid_pairs) == len(pairs_df):
        print("\n✅ All files are valid! Dataset is ready for training.")
    else:
        print(f"\n⚠️  Found {len(pairs_df) - len(valid_pairs)} corrupted pairs.")
        print("Run with --fix to create cleaned pairs.csv")


if __name__ == '__main__':
    main()
