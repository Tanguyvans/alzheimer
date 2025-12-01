#!/usr/bin/env python3
"""
Step 1: Prepare paired MRI data for Siamese network training.

Creates pairs of (baseline_mri, followup_mri) with conversion labels.

Usage:
    python 01_prepare_pairs.py --config config.yaml
"""

import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR.parent.parent / "data"


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = EXPERIMENT_DIR / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_diagnosis_data(dxsum_csv: str) -> pd.DataFrame:
    """Load ADNI diagnosis data and identify patient trajectories."""
    logger.info("Loading diagnosis data...")

    dx_df = pd.read_csv(dxsum_csv)
    logger.info(f"Loaded {len(dx_df):,} diagnosis records")

    # Filter valid diagnoses (1=CN, 2=MCI, 3=AD)
    dx_df = dx_df[dx_df['DIAGNOSIS'].notna()].copy()
    dx_df['DIAGNOSIS'] = dx_df['DIAGNOSIS'].astype(int)
    dx_df['EXAMDATE'] = pd.to_datetime(dx_df['EXAMDATE'], errors='coerce')

    # Get baseline diagnosis
    baseline = dx_df[dx_df['VISCODE'] == 'bl'][['PTID', 'DIAGNOSIS', 'EXAMDATE']].copy()
    baseline = baseline.rename(columns={'DIAGNOSIS': 'BL_DX', 'EXAMDATE': 'BL_DATE'})

    # Get last diagnosis
    last = dx_df.sort_values('EXAMDATE').groupby('PTID').last()[['DIAGNOSIS', 'EXAMDATE']].reset_index()
    last = last.rename(columns={'DIAGNOSIS': 'LAST_DX', 'EXAMDATE': 'LAST_DATE'})

    # Merge to get trajectories
    trajectories = baseline.merge(last, on='PTID', how='inner')

    # Assign trajectory labels
    def get_trajectory(row):
        bl, last = row['BL_DX'], row['LAST_DX']
        if bl == 1 and last == 1:
            return 'CN_stable'
        elif bl == 2 and last == 2:
            return 'MCI_stable'
        elif bl == 2 and last == 3:
            return 'MCI_to_AD'
        elif bl == 3 and last == 3:
            return 'AD_stable'
        elif bl == 1 and last == 2:
            return 'CN_to_MCI'
        elif bl == 1 and last == 3:
            return 'CN_to_AD'
        else:
            return 'other'

    trajectories['trajectory'] = trajectories.apply(get_trajectory, axis=1)

    # Binary converter label (MCI→AD = 1, else = 0) - kept for compatibility
    trajectories['is_converter'] = (trajectories['trajectory'] == 'MCI_to_AD').astype(int)

    logger.info(f"Identified {len(trajectories)} patient trajectories:")
    for traj in ['CN_stable', 'MCI_stable', 'MCI_to_AD', 'AD_stable']:
        count = (trajectories['trajectory'] == traj).sum()
        logger.info(f"  {traj}: {count}")

    return trajectories


def assign_class_labels(pairs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign class labels based on trajectory mapping from config."""
    trajectory_mapping = config['classes']['trajectory_mapping']
    class_names = config['classes']['names']

    # Map trajectory to class label
    pairs_df['label'] = pairs_df['trajectory'].map(trajectory_mapping)

    # Filter out unmapped trajectories (e.g., 'other', 'CN_to_MCI', 'CN_to_AD')
    n_before = len(pairs_df)
    pairs_df = pairs_df[pairs_df['label'].notna()].copy()
    pairs_df['label'] = pairs_df['label'].astype(int)
    n_after = len(pairs_df)

    if n_before > n_after:
        logger.info(f"Filtered {n_before - n_after} pairs with unmapped trajectories")

    # Add class name for readability
    pairs_df['class_name'] = pairs_df['label'].map(lambda x: class_names[x])

    return pairs_df


def find_mri_pairs(skull_dir: Path, trajectories: pd.DataFrame, min_days: int = 180) -> pd.DataFrame:
    """
    Find paired MRI scans (baseline + followup) from skull-stripped directory.

    Args:
        skull_dir: Directory with skull-stripped NIfTI files
        trajectories: DataFrame with patient trajectory information
        min_days: Minimum days between baseline and followup

    Returns:
        DataFrame with paired scan paths
    """
    logger.info(f"Searching for MRI pairs in {skull_dir}...")

    # Files are organized by PTID folder: skull_dir/PTID/*.nii.gz
    scans = []
    patient_folders = [d for d in skull_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patient_folders)} patient folders")

    for patient_folder in patient_folders:
        ptid = patient_folder.name  # Folder name is PTID (e.g., 002_S_0295)

        # Find all NIfTI files for this patient
        nifti_files = list(patient_folder.glob('*_registered_skull_stripped.nii.gz'))
        if not nifti_files:
            nifti_files = list(patient_folder.glob('*_mni_norm.nii.gz'))
        if not nifti_files:
            nifti_files = list(patient_folder.glob('*.nii.gz'))
        if not nifti_files:
            nifti_files = list(patient_folder.glob('*.nii'))

        for f in nifti_files:
            stem = f.name.replace('.nii.gz', '').replace('.nii', '')
            parts = stem.split('_')

            # Extract date (YYYYMMDD format) from filename
            scan_date = None
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    try:
                        scan_date = datetime.strptime(part, '%Y%m%d')
                        break
                    except:
                        pass

            scans.append({
                'ptid': ptid,
                'filepath': str(f),
                'date': scan_date
            })

    scans_df = pd.DataFrame(scans)

    if len(scans_df) == 0:
        logger.error("No scans found! Check skull_dir structure.")
        return pd.DataFrame()

    logger.info(f"Found {len(scans_df)} scans from {scans_df['ptid'].nunique()} patients")

    # Check how many scans have dates
    n_with_dates = scans_df['date'].notna().sum()
    logger.info(f"Scans with parsed dates: {n_with_dates}/{len(scans_df)}")

    # Create pairs
    pairs = []
    for ptid, group in scans_df.groupby('ptid'):
        if len(group) < 2:
            continue

        # Sort by date if available, otherwise by filename
        if group['date'].notna().all():
            group = group.sort_values('date')
            baseline = group.iloc[0]
            followup = group.iloc[-1]
            days_between = (followup['date'] - baseline['date']).days

            if days_between < min_days:
                continue
        else:
            # No dates - sort by filename and assume reasonable time gap
            group = group.sort_values('filepath')
            baseline = group.iloc[0]
            followup = group.iloc[-1]
            days_between = 365  # Default to 1 year if dates unknown

        pairs.append({
            'ptid': ptid,
            'baseline_path': baseline['filepath'],
            'followup_path': followup['filepath'],
            'baseline_date': baseline['date'],
            'followup_date': followup['date'],
            'days_between': days_between,
            'n_scans': len(group)
        })

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"Created {len(pairs_df)} scan pairs")

    if len(pairs_df) == 0:
        logger.error("No pairs created! Check scan dates or min_days setting.")
        return pd.DataFrame()

    # Merge with trajectory labels
    pairs_df = pairs_df.merge(
        trajectories[['PTID', 'trajectory', 'is_converter', 'BL_DX', 'LAST_DX']],
        left_on='ptid', right_on='PTID', how='inner'
    )
    pairs_df = pairs_df.drop(columns=['PTID'], errors='ignore')

    logger.info(f"Pairs with trajectory labels: {len(pairs_df)}")

    return pairs_df


def main():
    parser = argparse.ArgumentParser(description='Prepare MRI pairs for Siamese network')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Paths from config
    dxsum_csv = config['data']['dxsum_csv']
    skull_dir = Path(config['data']['skull_dir'])
    pairs_dir = EXPERIMENT_DIR / config['data']['pairs_dir']
    min_days = config['pair_selection']['min_days_between']

    # Create output directory
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # Check if skull directory exists
    if not skull_dir.exists():
        logger.error(f"Skull directory not found: {skull_dir}")
        logger.error("Please mount the external drive or update config.yaml")
        return

    # Load diagnosis data
    trajectories = load_diagnosis_data(dxsum_csv)

    # Find MRI pairs
    pairs_df = find_mri_pairs(skull_dir, trajectories, min_days)

    if len(pairs_df) == 0:
        logger.error("No valid pairs found!")
        return

    # Assign class labels (CN=0, MCI=1, AD=2)
    pairs_df = assign_class_labels(pairs_df, config)
    class_names = config['classes']['names']

    # Save pairs
    output_path = pairs_dir / 'pairs.csv'
    pairs_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(pairs_df)} pairs to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SIAMESE NETWORK - PAIRED MRI DATA (3-CLASS)")
    print("=" * 60)
    print(f"\nTotal pairs: {len(pairs_df)}")

    print(f"\nBy trajectory:")
    for traj in ['CN_stable', 'MCI_stable', 'MCI_to_AD', 'AD_stable']:
        count = (pairs_df['trajectory'] == traj).sum()
        print(f"  {traj}: {count}")

    print(f"\nBy class (CN / MCI / AD):")
    for label, name in enumerate(class_names):
        count = (pairs_df['label'] == label).sum()
        print(f"  {name} (label={label}): {count}")

    print(f"\nTime between scans:")
    print(f"  Mean: {pairs_df['days_between'].mean():.0f} days ({pairs_df['days_between'].mean()/365.25:.1f} years)")
    print(f"  Min: {pairs_df['days_between'].min():.0f} days")
    print(f"  Max: {pairs_df['days_between'].max():.0f} days")


if __name__ == '__main__':
    main()
