#!/usr/bin/env python3
"""
Create Combined Dataset with All Available Features

This script enhances the existing combined dataset by adding additional tabular features
that are available across ADNI, OASIS, and NACC datasets.

Strategy:
- Use the existing combined dataset (data/combined/) which has proper MRI scan paths
- Load additional features from source tabular files
- Merge features by subject_id
- Preserve the same train/val/test split

Usage:
    python create_combined_dataset.py --output data/combined_v2
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Features to add to the existing combined dataset
ADDITIONAL_FEATURES = [
    # Physical (add if not present)
    'VSWEIGHT',    # Weight
    'BMI',         # Body Mass Index
    # Medical history (add if not present)
    'MH14ALCH',    # Alcohol abuse
    'MH16SMOK',    # Smoking history
    'MH4CARD',     # Cardiovascular disease
    'MH2NEURL',    # Neurological conditions
    # Cognitive tests (add if not present)
    'DSPANFOR',    # Digit span forward
    'DSPANBAC',    # Digit span backward
    'BNTTOTAL',    # Boston Naming Test
]

# All features in the final dataset
ALL_FEATURES = [
    # Demographics (already in combined)
    'AGE',
    'PTGENDER',
    'PTEDUCAT',
    'PTMARRY',
    # Cognitive tests (already in combined)
    'CATANIMSC',   # Category fluency - animals
    'TRAASCOR',    # Trail Making A - time
    'TRABSCOR',    # Trail Making B - time
    # Cognitive tests (to add)
    'DSPANFOR',    # Digit span forward
    'DSPANBAC',    # Digit span backward
    'BNTTOTAL',    # Boston Naming Test
    # Physical (to add)
    'VSWEIGHT',    # Weight
    'BMI',         # Body Mass Index
    # Medical history (to add)
    'MH14ALCH',    # Alcohol abuse
    'MH16SMOK',    # Smoking history
    'MH4CARD',     # Cardiovascular disease
    'MH2NEURL',    # Neurological conditions
]


def load_adni_features(data_dir: Path) -> pd.DataFrame:
    """Load ADNI tabular features."""
    adni_path = data_dir / 'adni' / 'adni_cn_ad_trajectory.csv'
    if not adni_path.exists():
        adni_path = data_dir / 'adni' / 'clinical_data_all_groups.csv'

    print(f"Loading ADNI features from: {adni_path}")
    df = pd.read_csv(adni_path)

    # Select only subject_id and additional features
    cols = ['subject_id'] + [f for f in ADDITIONAL_FEATURES if f in df.columns]
    df = df[cols].drop_duplicates(subset=['subject_id'], keep='first')

    print(f"  ADNI: {len(df)} subjects with features: {[c for c in cols if c != 'subject_id']}")
    return df


def load_oasis_features(data_dir: Path) -> pd.DataFrame:
    """Load OASIS tabular features."""
    oasis_path = data_dir / 'oasis' / 'oasis_tabular.csv'
    if not oasis_path.exists():
        oasis_path = data_dir / 'oasis' / 'oasis_all.csv'

    print(f"Loading OASIS features from: {oasis_path}")
    df = pd.read_csv(oasis_path)

    # Rename subject column
    if 'Subject' in df.columns:
        df = df.rename(columns={'Subject': 'subject_id'})

    # Select only subject_id and additional features
    cols = ['subject_id'] + [f for f in ADDITIONAL_FEATURES if f in df.columns]
    df = df[cols].drop_duplicates(subset=['subject_id'], keep='first')

    print(f"  OASIS: {len(df)} subjects with features: {[c for c in cols if c != 'subject_id']}")
    return df


def load_nacc_features(data_dir: Path) -> pd.DataFrame:
    """Load NACC tabular features."""
    nacc_path = data_dir / 'nacc' / 'nacc_tabular_mri.csv'
    if not nacc_path.exists():
        nacc_path = data_dir / 'nacc' / 'nacc_tabular.csv'

    print(f"Loading NACC features from: {nacc_path}")
    df = pd.read_csv(nacc_path)

    # Rename subject column
    if 'Subject' in df.columns:
        df = df.rename(columns={'Subject': 'subject_id'})

    # Select only subject_id and additional features
    cols = ['subject_id'] + [f for f in ADDITIONAL_FEATURES if f in df.columns]
    df = df[cols].drop_duplicates(subset=['subject_id'], keep='first')

    print(f"  NACC: {len(df)} subjects with features: {[c for c in cols if c != 'subject_id']}")
    return df


def check_feature_coverage(df: pd.DataFrame, features: list) -> dict:
    """Check how many non-null values each feature has."""
    coverage = {}
    for feat in features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            coverage[feat] = {
                'non_null': int(non_null),
                'total': len(df),
                'coverage': round(non_null / len(df) * 100, 1)
            }
    return coverage


def enhance_combined_dataset(data_dir: Path, existing_combined_dir: Path, output_dir: Path, seed: int = 42):
    """Enhance existing combined dataset with additional features."""

    print("=" * 60)
    print("Enhancing Combined Dataset with Additional Features")
    print("=" * 60)

    # Load existing combined dataset (preserves scan_path and train/val/test split)
    print("\nLoading existing combined dataset...")
    train_df = pd.read_csv(existing_combined_dir / 'train.csv')
    val_df = pd.read_csv(existing_combined_dir / 'val.csv')
    test_df = pd.read_csv(existing_combined_dir / 'test.csv')

    print(f"  Existing train: {len(train_df)} samples")
    print(f"  Existing val: {len(val_df)} samples")
    print(f"  Existing test: {len(test_df)} samples")

    # Load additional features from source datasets
    print("\nLoading additional features from source datasets...")
    adni_features = load_adni_features(data_dir)
    oasis_features = load_oasis_features(data_dir)
    nacc_features = load_nacc_features(data_dir)

    # Combine all feature sources
    all_features_df = pd.concat([adni_features, oasis_features, nacc_features], ignore_index=True)
    all_features_df = all_features_df.drop_duplicates(subset=['subject_id'], keep='first')
    print(f"\nTotal subjects with additional features: {len(all_features_df)}")

    # Merge additional features into existing datasets
    print("\nMerging additional features...")

    def merge_features(df, features_df):
        """Merge additional features, preserving existing columns."""
        # Identify which additional features to add (not already in df)
        new_features = [f for f in ADDITIONAL_FEATURES if f not in df.columns and f in features_df.columns]
        if new_features:
            merge_cols = ['subject_id'] + new_features
            df = df.merge(features_df[merge_cols], on='subject_id', how='left')
        return df

    train_df = merge_features(train_df, all_features_df)
    val_df = merge_features(val_df, all_features_df)
    test_df = merge_features(test_df, all_features_df)

    # Combine for statistics
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Check feature coverage
    print("\nFeature coverage:")
    coverage = check_feature_coverage(combined_df, ALL_FEATURES)
    for feat in ALL_FEATURES:
        if feat in coverage:
            stats = coverage[feat]
            print(f"  {feat}: {stats['coverage']}% ({stats['non_null']}/{stats['total']})")
        else:
            print(f"  {feat}: NOT FOUND")

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(combined_df)*100:.1f}%)")

    print(f"\nClass distribution:")
    print(f"  CN: {(combined_df['label']==0).sum()} ({(combined_df['label']==0).mean()*100:.1f}%)")
    print(f"  AD_trajectory: {(combined_df['label']==1).sum()} ({(combined_df['label']==1).mean()*100:.1f}%)")

    print(f"\nSource distribution:")
    for source in ['ADNI', 'OASIS', 'NACC']:
        n = (combined_df['source'] == source).sum()
        print(f"  {source}: {n} ({n/len(combined_df)*100:.1f}%)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    combined_df.to_csv(output_dir / 'all.csv', index=False)

    # Save metadata
    metadata = {
        'dataset': 'combined_v2',
        'description': 'Enhanced combined ADNI+OASIS+NACC with additional tabular features',
        'total_samples': len(combined_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'CN': int((combined_df['label'] == 0).sum()),
            'AD_trajectory': int((combined_df['label'] == 1).sum())
        },
        'tabular_features': ALL_FEATURES,
        'num_features': len(ALL_FEATURES),
        'feature_coverage': coverage,
        'sources': {
            'ADNI': int((combined_df['source'] == 'ADNI').sum()),
            'OASIS': int((combined_df['source'] == 'OASIS').sum()),
            'NACC': int((combined_df['source'] == 'NACC').sum())
        },
        'seed': seed,
        'note': 'Enhanced from existing combined dataset, preserving scan paths and train/val/test split'
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  - train.csv ({len(train_df)} samples)")
    print(f"  - val.csv ({len(val_df)} samples)")
    print(f"  - test.csv ({len(test_df)} samples)")
    print(f"  - all.csv ({len(combined_df)} samples)")
    print(f"  - metadata.json")

    return combined_df, metadata


def main():
    parser = argparse.ArgumentParser(description='Enhance combined dataset with all features')
    parser.add_argument('--data-dir', type=str, default='../../data',
                       help='Path to data directory containing adni/, oasis/, nacc/')
    parser.add_argument('--existing-combined', type=str, default='data/combined',
                       help='Path to existing combined dataset directory')
    parser.add_argument('--output', type=str, default='data/combined_v2',
                       help='Output directory for enhanced dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir

    existing_combined_dir = Path(args.existing_combined)
    if not existing_combined_dir.is_absolute():
        existing_combined_dir = script_dir / existing_combined_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    enhance_combined_dataset(data_dir, existing_combined_dir, output_dir, args.seed)


if __name__ == '__main__':
    main()
