#!/usr/bin/env python3
"""
Prepare Multi-Modal Dataset

Creates train/val/test splits with both MRI paths and tabular features.
Supports ADNI, OASIS, NACC, and combined datasets.
Supports two tasks:
  - cn_ad_trajectory: CN vs (AD + MCI converters) - harder, includes early-stage prediction
  - cn_ad: CN vs stable AD only - easier, for comparison with literature

Usage:
    python prepare_dataset.py --dataset combined --task cn_ad_trajectory  # → data/combined_trajectory
    python prepare_dataset.py --dataset combined --task cn_ad             # → data/combined_cn_ad
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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default paths
DEFAULT_ADNI_CSV = DATA_DIR / "adni" / "adni_cn_ad_trajectory.csv"
DEFAULT_OASIS_CSV = DATA_DIR / "oasis" / "oasis_all_full.csv"
DEFAULT_OASIS_MRI_DIR = Path("/home/maxglo/tanguy/OASIS-skull")
DEFAULT_NACC_CSV = DATA_DIR / "nacc" / "nacc_tabular_t1.csv"
DEFAULT_NACC_MRI_META = DATA_DIR / "nacc" / "nacc-t1_12_16_2025.csv"
DEFAULT_NACC_MRI_DIR = Path("/home/tanguy/medical/NACC-skull")

# Common tabular features across ADNI and OASIS
COMMON_FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    'CATANIMSC', 'TRAASCOR', 'TRABSCOR'
]

# ADNI-specific features (superset)
ADNI_FEATURES = COMMON_FEATURES + [
    'VSWEIGHT', 'BMI',
    'MH14ALCH', 'MH16SMOK', 'MH4CARD', 'MHPSYCH', 'MH2NEURL',
    'TRABERRCOM', 'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC'
]


def load_adni_data(adni_csv: str, task: str = 'cn_ad_trajectory') -> pd.DataFrame:
    """Load ADNI data with MRI paths and tabular features.

    Args:
        adni_csv: Path to ADNI clinical CSV
        task: 'cn_ad_trajectory' (CN vs AD + MCI converters) or 'cn_ad' (CN vs stable AD only)
    """
    logger.info(f"Loading ADNI data from {adni_csv} for task: {task}")
    df = pd.read_csv(adni_csv)

    # Standard column names
    if 'scan_path' not in df.columns and 'nii_path' in df.columns:
        df = df.rename(columns={'nii_path': 'scan_path'})

    # Add source column
    df['source'] = 'ADNI'

    if task == 'cn_ad':
        # CN vs stable AD only - exclude MCI converters
        # Need to check original diagnosis, not trajectory
        if 'DX_original' in df.columns:
            # Use original diagnosis column if available
            df = df[df['DX_original'].isin(['CN', 'AD'])].copy()
            df['label'] = df['DX_original'].map({'CN': 0, 'AD': 1})
            df['DX'] = df['DX_original']
        elif 'DX' in df.columns:
            # Filter to only CN and AD (exclude trajectory labels)
            # First check if we have the raw diagnosis
            if 'AD_trajectory' in df['DX'].values:
                # This CSV has trajectory labels, need to filter differently
                # We need the original ADNI CSV to get stable AD
                logger.warning("CSV contains trajectory labels. For cn_ad task, need original diagnoses.")
                # Fall back to using trajectory but rename
                df = df[df['DX'].isin(['CN', 'AD_trajectory'])].copy()
                df['label'] = df['DX'].map({'CN': 0, 'AD_trajectory': 1})
            else:
                df = df[df['DX'].isin(['CN', 'AD'])].copy()
                df['label'] = df['DX'].map({'CN': 0, 'AD': 1})
        label_name = 'AD'
    else:
        # cn_ad_trajectory: CN vs (AD + MCI converters)
        if 'DX' in df.columns:
            df['label'] = df['DX'].map({'CN': 0, 'AD_trajectory': 1})
        elif 'trajectory' in df.columns:
            df['label'] = df['trajectory'].map({'CN': 0, 'AD_trajectory': 1})
        label_name = 'AD_trajectory'

    # Filter valid labels
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    logger.info(f"ADNI: {len(df)} samples loaded")
    logger.info(f"  CN: {(df['label'] == 0).sum()}, {label_name}: {(df['label'] == 1).sum()}")

    return df


def load_oasis_data(oasis_csv: str, oasis_mri_dir: str, task: str = 'cn_ad_trajectory') -> pd.DataFrame:
    """Load OASIS data and match with MRI paths.

    Args:
        oasis_csv: Path to OASIS clinical CSV
        oasis_mri_dir: Path to OASIS MRI directory
        task: 'cn_ad_trajectory' (CN vs AD + MCI converters) or 'cn_ad' (CN vs stable AD only)
    """
    logger.info(f"Loading OASIS data from {oasis_csv} for task: {task}")
    df = pd.read_csv(oasis_csv)

    # Take first visit per subject
    if 'days_to_visit' in df.columns:
        df = df.sort_values(['Subject', 'days_to_visit']).groupby('Subject').first().reset_index()
    else:
        df = df.groupby('Subject').first().reset_index()

    if task == 'cn_ad':
        # CN vs stable AD only - use first visit diagnosis directly
        # Only include subjects whose first visit is CN or AD (not MCI)
        df = df[df['DX'].isin(['CN', 'AD'])].copy()
        df['trajectory'] = df['DX']
        label_name = 'AD'
    else:
        # cn_ad_trajectory: CN vs (AD + MCI converters)
        # Need to reload to find patients who ever had AD
        df_all = pd.read_csv(oasis_csv)
        patients_with_ad = set(df_all[df_all['DX'] == 'AD']['Subject'].unique())
        logger.info(f"OASIS patients who ever had AD: {len(patients_with_ad)}")

        # Assign trajectory labels
        def assign_trajectory(row):
            if row['Subject'] in patients_with_ad:
                return 'AD_trajectory'
            elif row['DX'] == 'CN':
                return 'CN'
            else:
                return None  # MCI non-converters excluded

        df['trajectory'] = df.apply(assign_trajectory, axis=1)
        df = df[df['trajectory'].notna()].copy()
        label_name = 'AD_trajectory'

    # Find MRI paths
    oasis_mri_path = Path(oasis_mri_dir)
    scan_paths = []

    for subject in df['Subject']:
        subject_dir = oasis_mri_path / subject
        if subject_dir.exists():
            # Find skull-stripped NIfTI file
            nii_files = list(subject_dir.glob('*skull_stripped*.nii.gz'))
            if not nii_files:
                nii_files = list(subject_dir.glob('*.nii.gz'))
            scan_paths.append(str(nii_files[0]) if nii_files else None)
        else:
            scan_paths.append(None)

    df['scan_path'] = scan_paths
    df['subject_id'] = df['Subject']
    df['DX'] = df['trajectory']
    df['source'] = 'OASIS'

    # Map labels based on task
    if task == 'cn_ad':
        df['label'] = df['trajectory'].map({'CN': 0, 'AD': 1})
    else:
        df['label'] = df['trajectory'].map({'CN': 0, 'AD_trajectory': 1})

    # Filter to samples with MRI
    before = len(df)
    df = df[df['scan_path'].notna()].copy()
    logger.info(f"OASIS: {len(df)} samples with MRI (filtered {before - len(df)} without MRI)")
    logger.info(f"  CN: {(df['label'] == 0).sum()}, {label_name}: {(df['label'] == 1).sum()}")

    return df


def load_nacc_data(nacc_csv: str, nacc_mri_meta: str, nacc_mri_dir: str, task: str = 'cn_ad_trajectory') -> pd.DataFrame:
    """Load NACC data and match with MRI paths.

    Args:
        nacc_csv: Path to NACC tabular CSV
        nacc_mri_meta: Path to NACC MRI metadata CSV
        nacc_mri_dir: Path to NACC MRI directory
        task: 'cn_ad_trajectory' or 'cn_ad' (same for NACC since we only have CN/AD)
    """
    logger.info(f"Loading NACC data from {nacc_csv} for task: {task}")
    df = pd.read_csv(nacc_csv)

    logger.info(f"Loading NACC MRI metadata from {nacc_mri_meta}")
    mri_meta = pd.read_csv(nacc_mri_meta)

    # Create mapping from Subject to Image Data ID
    subject_to_image = mri_meta.set_index('Subject')['Image Data ID'].to_dict()

    # Filter to CN and AD only
    df = df[df['DX'].isin(['CN', 'AD'])].copy()
    logger.info(f"NACC: {len(df)} samples with CN or AD diagnosis")

    # Find MRI paths
    nacc_mri_path = Path(nacc_mri_dir)
    scan_paths = []

    for _, row in df.iterrows():
        subject = row['Subject']
        image_id = subject_to_image.get(subject)

        if image_id:
            subject_dir = nacc_mri_path / subject
            if subject_dir.exists():
                # Find skull-stripped NIfTI file containing the image ID
                nii_files = list(subject_dir.glob(f'*{image_id}*skull_stripped*.nii.gz'))
                if not nii_files:
                    nii_files = list(subject_dir.glob('*skull_stripped*.nii.gz'))
                if not nii_files:
                    nii_files = list(subject_dir.glob('*.nii.gz'))
                scan_paths.append(str(nii_files[0]) if nii_files else None)
            else:
                scan_paths.append(None)
        else:
            scan_paths.append(None)

    df['scan_path'] = scan_paths
    df['subject_id'] = df['Subject']
    df['source'] = 'NACC'

    # Map labels based on task
    if task == 'cn_ad':
        df['label'] = df['DX'].map({'CN': 0, 'AD': 1})
        label_name = 'AD'
    else:
        # Map AD to AD_trajectory for consistency
        df['DX'] = df['DX'].map({'CN': 'CN', 'AD': 'AD_trajectory'})
        df['label'] = df['DX'].map({'CN': 0, 'AD_trajectory': 1})
        label_name = 'AD_trajectory'

    # Filter to samples with MRI
    before = len(df)
    df = df[df['scan_path'].notna()].copy()
    logger.info(f"NACC: {len(df)} samples with MRI (filtered {before - len(df)} without MRI)")
    logger.info(f"  CN: {(df['label'] == 0).sum()}, {label_name}: {(df['label'] == 1).sum()}")

    return df


def prepare_multimodal_dataset(
    dataset: str = 'adni',
    output_dir: str = 'data/adni',
    task: str = 'cn_ad_trajectory',
    adni_csv: str = None,
    oasis_csv: str = None,
    oasis_mri_dir: str = None,
    nacc_csv: str = None,
    nacc_mri_meta: str = None,
    nacc_mri_dir: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    check_mri_exists: bool = False
):
    """
    Prepare multi-modal dataset for training.

    Args:
        dataset: 'adni', 'oasis', 'nacc', or 'combined'
        output_dir: Output directory for train/val/test CSVs
        task: 'cn_ad_trajectory' (CN vs AD + MCI converters) or 'cn_ad' (CN vs stable AD)
        adni_csv: Path to ADNI clinical CSV (with MRI paths)
        oasis_csv: Path to OASIS clinical CSV
        oasis_mri_dir: Path to OASIS MRI directory
        nacc_csv: Path to NACC tabular CSV
        nacc_mri_meta: Path to NACC MRI metadata CSV
        nacc_mri_dir: Path to NACC MRI directory
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        check_mri_exists: If True, verify MRI files exist (slow, requires access to MRI dir)
    """
    # Set defaults
    adni_csv = adni_csv or str(DEFAULT_ADNI_CSV)
    oasis_csv = oasis_csv or str(DEFAULT_OASIS_CSV)
    oasis_mri_dir = oasis_mri_dir or str(DEFAULT_OASIS_MRI_DIR)
    nacc_csv = nacc_csv or str(DEFAULT_NACC_CSV)
    nacc_mri_meta = nacc_mri_meta or str(DEFAULT_NACC_MRI_META)
    nacc_mri_dir = nacc_mri_dir or str(DEFAULT_NACC_MRI_DIR)

    logger.info(f"=" * 60)
    logger.info(f"Preparing {dataset.upper()} multimodal dataset")
    logger.info(f"Task: {task}")
    logger.info(f"=" * 60)

    # Determine label name based on task
    label_name = 'AD' if task == 'cn_ad' else 'AD_trajectory'

    # Load data based on dataset type
    if dataset == 'adni':
        df = load_adni_data(adni_csv, task=task)
        features_to_use = ADNI_FEATURES
    elif dataset == 'oasis':
        df = load_oasis_data(oasis_csv, oasis_mri_dir, task=task)
        features_to_use = COMMON_FEATURES
    elif dataset == 'nacc':
        df = load_nacc_data(nacc_csv, nacc_mri_meta, nacc_mri_dir, task=task)
        features_to_use = ADNI_FEATURES  # NACC has same features mapped to ADNI names
    elif dataset == 'combined':
        adni_df = load_adni_data(adni_csv, task=task)
        oasis_df = load_oasis_data(oasis_csv, oasis_mri_dir, task=task)
        nacc_df = load_nacc_data(nacc_csv, nacc_mri_meta, nacc_mri_dir, task=task)

        # Use only common columns for combined dataset
        common_cols = ['subject_id', 'scan_path', 'DX', 'label', 'source'] + COMMON_FEATURES

        # Filter to common columns that exist
        adni_cols = [c for c in common_cols if c in adni_df.columns]
        oasis_cols = [c for c in common_cols if c in oasis_df.columns]
        nacc_cols = [c for c in common_cols if c in nacc_df.columns]

        df = pd.concat([adni_df[adni_cols], oasis_df[oasis_cols], nacc_df[nacc_cols]], ignore_index=True)
        features_to_use = COMMON_FEATURES
        logger.info(f"Combined: {len(df)} total samples")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Optionally check MRI files exist
    if check_mri_exists:
        logger.info("Checking MRI file availability...")
        df['mri_exists'] = df['scan_path'].apply(lambda x: Path(x).exists() if pd.notna(x) else False)
        available = df['mri_exists'].sum()
        logger.info(f"  Available: {available}/{len(df)} ({100*available/len(df):.1f}%)")
        df = df[df['mri_exists']].copy()
        df = df.drop(columns=['mri_exists'])

    # Report feature availability
    logger.info(f"\nFeature availability:")
    available_features = []
    for f in features_to_use:
        if f in df.columns:
            non_null = df[f].notna().sum()
            pct = 100 * non_null / len(df)
            logger.info(f"  {f}: {non_null}/{len(df)} ({pct:.1f}%)")
            available_features.append(f)
        else:
            logger.info(f"  {f}: NOT IN DATASET")

    logger.info(f"\nClass distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        name = 'CN' if label == 0 else label_name
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

    # Save all data
    df.to_csv(output_path / 'all.csv', index=False)

    # Save metadata
    metadata = {
        'dataset': dataset,
        'task': task,
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'CN': int((df['label'] == 0).sum()),
            label_name: int((df['label'] == 1).sum())
        },
        'tabular_features_available': available_features,
        'seed': seed
    }

    if dataset == 'combined':
        metadata['sources'] = {
            'ADNI': int((df['source'] == 'ADNI').sum()),
            'OASIS': int((df['source'] == 'OASIS').sum()),
            'NACC': int((df['source'] == 'NACC').sum())
        }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDataset saved to {output_path}")
    logger.info(f"Available tabular features: {available_features}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare Multi-Modal Dataset')
    parser.add_argument('--dataset', type=str, default='adni',
                       choices=['adni', 'oasis', 'nacc', 'combined'],
                       help='Dataset to prepare')
    parser.add_argument('--task', type=str, default='cn_ad_trajectory',
                       choices=['cn_ad_trajectory', 'cn_ad'],
                       help='Classification task: cn_ad_trajectory (CN vs AD+MCI converters) or cn_ad (CN vs stable AD)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data/{dataset} or data/{dataset}_cn_ad for cn_ad task)')
    parser.add_argument('--adni-csv', type=str, default=None,
                       help='Path to ADNI clinical CSV')
    parser.add_argument('--oasis-csv', type=str, default=None,
                       help='Path to OASIS clinical CSV')
    parser.add_argument('--oasis-mri-dir', type=str, default=None,
                       help='Path to OASIS MRI directory')
    parser.add_argument('--nacc-csv', type=str, default=None,
                       help='Path to NACC tabular CSV')
    parser.add_argument('--nacc-mri-meta', type=str, default=None,
                       help='Path to NACC MRI metadata CSV')
    parser.add_argument('--nacc-mri-dir', type=str, default=None,
                       help='Path to NACC MRI directory')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-mri', action='store_true',
                       help='Verify MRI files exist (requires access to MRI directory)')
    args = parser.parse_args()

    # Set default output dir based on dataset and task
    if args.output_dir:
        output_dir = args.output_dir
    elif args.task == 'cn_ad':
        output_dir = f'data/{args.dataset}_cn_ad'
    else:
        # cn_ad_trajectory task
        output_dir = f'data/{args.dataset}_trajectory'

    prepare_multimodal_dataset(
        dataset=args.dataset,
        output_dir=output_dir,
        task=args.task,
        adni_csv=args.adni_csv,
        oasis_csv=args.oasis_csv,
        oasis_mri_dir=args.oasis_mri_dir,
        nacc_csv=args.nacc_csv,
        nacc_mri_meta=args.nacc_mri_meta,
        nacc_mri_dir=args.nacc_mri_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        check_mri_exists=args.check_mri
    )


if __name__ == '__main__':
    main()
