#!/usr/bin/env python3
"""
Step 1: Prepare paired MRI data for Siamese network training.

Creates pairs of (baseline_mri, followup_mri) with conversion labels.

Usage:
    python 01_prepare_pairs.py --npy-dir /path/to/npy --output data/pairs.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def parse_scan_filename(filename: str) -> dict:
    """
    Parse patient ID and date from scan filename.
    Expected format: XXX_S_XXXX_YYYYMMDD.npy or similar
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    # Try to extract PTID (format: XXX_S_XXXX)
    if len(parts) >= 3 and parts[1] == 'S':
        ptid = '_'.join(parts[:3])
        # Rest might contain date
        rest = '_'.join(parts[3:])
    else:
        ptid = parts[0]
        rest = '_'.join(parts[1:])

    # Try to parse date from remaining parts
    scan_date = None
    for part in parts:
        # Try YYYYMMDD format
        if len(part) == 8 and part.isdigit():
            try:
                scan_date = datetime.strptime(part, '%Y%m%d')
            except:
                pass
        # Try YYYY-MM-DD format
        elif '-' in part:
            try:
                scan_date = datetime.strptime(part, '%Y-%m-%d')
            except:
                pass

    return {'ptid': ptid, 'date': scan_date, 'filename': filename}


def load_clinical_labels():
    """Load clinical data with conversion labels"""
    clinical_path = DATA_DIR / "ALL_4class_clinical.csv"

    if not clinical_path.exists():
        logger.warning(f"Clinical data not found at {clinical_path}")
        return None

    df = pd.read_csv(clinical_path, low_memory=False)

    # Get patient-level labels
    # BL_DX: baseline diagnosis (1=CN, 2=MCI, 3=AD)
    # LAST_DX: last diagnosis
    patient_labels = df.groupby('PTID').agg({
        'BL_DX': 'first',
        'LAST_DX': 'first',
        'DX': 'first'
    }).reset_index()

    # Create conversion label
    # 0 = stable (CN→CN or MCI→MCI)
    # 1 = converter (MCI→AD)
    patient_labels['is_converter'] = (
        (patient_labels['BL_DX'] == 2) & (patient_labels['LAST_DX'] == 3)
    ).astype(int)

    # Also track trajectory
    patient_labels['trajectory'] = 'unknown'
    patient_labels.loc[(patient_labels['BL_DX'] == 1) & (patient_labels['LAST_DX'] == 1), 'trajectory'] = 'CN_stable'
    patient_labels.loc[(patient_labels['BL_DX'] == 2) & (patient_labels['LAST_DX'] == 2), 'trajectory'] = 'MCI_stable'
    patient_labels.loc[(patient_labels['BL_DX'] == 2) & (patient_labels['LAST_DX'] == 3), 'trajectory'] = 'MCI_to_AD'
    patient_labels.loc[(patient_labels['BL_DX'] == 3) & (patient_labels['LAST_DX'] == 3), 'trajectory'] = 'AD_stable'

    return patient_labels


def find_scan_pairs(npy_dir: Path, min_days_between: int = 180):
    """
    Find pairs of scans for each patient.

    Args:
        npy_dir: Directory containing .npy MRI files
        min_days_between: Minimum days between baseline and followup

    Returns:
        DataFrame with scan pairs
    """
    npy_files = list(npy_dir.glob("*.npy"))
    logger.info(f"Found {len(npy_files)} .npy files")

    # Parse all filenames
    scans = []
    for f in npy_files:
        info = parse_scan_filename(f.name)
        info['filepath'] = str(f)
        scans.append(info)

    scans_df = pd.DataFrame(scans)

    # Group by patient
    pairs = []
    for ptid, group in scans_df.groupby('ptid'):
        if len(group) < 2:
            continue

        # Sort by date if available
        if group['date'].notna().all():
            group = group.sort_values('date')

        # Get baseline (first) and followup (last)
        baseline = group.iloc[0]
        followup = group.iloc[-1]

        # Check time difference
        if baseline['date'] and followup['date']:
            days_between = (followup['date'] - baseline['date']).days
            if days_between < min_days_between:
                continue
        else:
            days_between = None

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

    return pairs_df


def create_pairs_from_metadata(npy_dir: Path, metadata_csv: Path = None):
    """
    Create pairs using metadata CSV with proper date information.
    """
    # Load MRI metadata
    if metadata_csv is None:
        metadata_csv = DATA_DIR / "tabular" / "3D_MPRAGE_Imaging_Cohort_Key_MRI_10Oct2025.csv"

    if not metadata_csv.exists():
        logger.warning(f"Metadata not found: {metadata_csv}")
        return None

    mri_df = pd.read_csv(metadata_csv, low_memory=False, encoding='latin-1')
    logger.info(f"Loaded {len(mri_df)} MRI records from metadata")

    # Parse dates
    mri_df['image_date'] = pd.to_datetime(mri_df['image_date'], errors='coerce')

    # Map to npy files
    npy_files = {f.stem: str(f) for f in npy_dir.glob("*.npy")}

    # Try to match by image_id or subject_id
    pairs = []
    for ptid, group in mri_df.groupby('subject_id'):
        if len(group) < 2:
            continue

        group = group.sort_values('image_date')

        # Get baseline and last scan
        baseline = group.iloc[0]
        followup = group.iloc[-1]

        days_between = (followup['image_date'] - baseline['image_date']).days
        if days_between < 180:  # At least 6 months
            continue

        # Find corresponding npy files
        bl_matches = [k for k in npy_files.keys() if ptid in k]
        if len(bl_matches) < 2:
            continue

        # Sort matches by any date info in filename
        bl_matches.sort()

        pairs.append({
            'ptid': ptid,
            'baseline_path': npy_files.get(bl_matches[0], ''),
            'followup_path': npy_files.get(bl_matches[-1], ''),
            'baseline_date': baseline['image_date'],
            'followup_date': followup['image_date'],
            'days_between': days_between,
            'n_scans': len(group)
        })

    return pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description='Prepare MRI pairs for Siamese network')
    parser.add_argument('--npy-dir', type=str, required=True, help='Directory with .npy MRI files')
    parser.add_argument('--output', type=str, default='data/pairs.csv', help='Output CSV path')
    parser.add_argument('--min-days', type=int, default=180, help='Minimum days between scans')
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find scan pairs
    logger.info(f"Scanning {npy_dir} for MRI pairs...")
    pairs_df = find_scan_pairs(npy_dir, args.min_days)

    if len(pairs_df) == 0:
        logger.error("No pairs found. Check npy directory and filename format.")
        return

    # Load and merge clinical labels
    labels_df = load_clinical_labels()
    if labels_df is not None:
        pairs_df = pairs_df.merge(
            labels_df[['PTID', 'is_converter', 'trajectory', 'BL_DX', 'LAST_DX']],
            left_on='ptid', right_on='PTID', how='left'
        )
        pairs_df = pairs_df.drop(columns=['PTID'], errors='ignore')

        # Filter to patients with labels
        pairs_df = pairs_df.dropna(subset=['is_converter'])
        logger.info(f"Pairs with clinical labels: {len(pairs_df)}")

    # Save
    pairs_df.to_csv(output_path, index=False)
    logger.info(f"Saved pairs to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PAIR SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {len(pairs_df)}")

    if 'trajectory' in pairs_df.columns:
        print(f"\nBy trajectory:")
        print(pairs_df['trajectory'].value_counts())

        print(f"\nConverters vs Non-converters:")
        print(pairs_df['is_converter'].value_counts())

    if 'days_between' in pairs_df.columns:
        print(f"\nTime between scans:")
        print(f"  Mean: {pairs_df['days_between'].mean():.0f} days")
        print(f"  Min: {pairs_df['days_between'].min():.0f} days")
        print(f"  Max: {pairs_df['days_between'].max():.0f} days")


if __name__ == '__main__':
    main()
