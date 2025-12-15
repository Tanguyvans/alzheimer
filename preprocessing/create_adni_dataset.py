#!/usr/bin/env python3
"""
Create ADNI dataset CSVs from raw tables.

Supports multiple dataset configurations:
- all: All 4 classes (CN, MCI_stable, MCI_to_AD, AD)
- cn_ad: Binary CN vs AD only
- cn_ad_trajectory: Binary CN vs AD trajectory (MCI_to_AD + AD)
- cn_mci_ad: 3-class (CN, MCI, AD)

Usage:
    python create_adni_dataset.py --variant cn_ad_trajectory
    python create_adni_dataset.py --variant all
    python create_adni_dataset.py --variant cn_mci_ad
"""

import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "adni" / "tabular_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "adni"
MRI_BASE_PATH = "/home/maxglo/tanguy/ADNI-skull"

# Dataset variants
VARIANTS = {
    'all': {
        'name': 'adni_all.csv',
        'classes': ['CN', 'MCI_stable', 'MCI_to_AD', 'AD'],
        'description': 'All 4 classes'
    },
    'cn_ad': {
        'name': 'adni_cn_ad.csv',
        'classes': ['CN', 'AD'],
        'description': 'Binary CN vs AD only'
    },
    'cn_ad_trajectory': {
        'name': 'adni_cn_ad_trajectory.csv',
        'classes': ['CN', 'MCI_to_AD', 'AD'],
        'dx_mapping': {'CN': 'CN', 'MCI_to_AD': 'AD_trajectory', 'AD': 'AD_trajectory'},
        'description': 'Binary CN vs AD trajectory (MCI_to_AD + AD)'
    },
    'cn_mci_ad': {
        'name': 'adni_cn_mci_ad.csv',
        'classes': ['CN', 'MCI_stable', 'MCI_to_AD', 'AD'],
        'dx_mapping': {'CN': 'CN', 'MCI_stable': 'MCI', 'MCI_to_AD': 'MCI', 'AD': 'AD'},
        'description': '3-class CN vs MCI vs AD'
    }
}

# Tabular features to extract
TABULAR_FEATURES = [
    # Demographics
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    # Cognitive screen
    'MMSCORE', 'CDGLOBAL',
    # Neuropsych tests
    'CLOCKSCOR', 'CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM',
    'DSPANFOR', 'DSPANBAC', 'BNTTOTAL',
    # Medical history
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    # Other
    'BCDEPRES', 'BCFAQ',
    # Vitals
    'VSWEIGHT', 'VSHEIGHT', 'BMI'
]


def load_tables():
    """Load all required tables."""
    tables = {}

    tables['my_table'] = pd.read_csv(
        RAW_PATH / "3D_MPRAGE_Imaging_Cohort_My_Table_10Oct2025.csv",
        low_memory=False
    )
    tables['study_entry'] = pd.read_csv(
        RAW_PATH / "3D_MPRAGE_Imaging_Cohort_Study_Entry_10Oct2025.csv"
    )
    tables['dxsum'] = pd.read_csv(
        RAW_PATH / "3D_MPRAGE_Imaging_Cohort_DXSUM_10Oct2025.csv"
    )
    tables['key_mri'] = pd.read_csv(
        RAW_PATH / "3D_MPRAGE_Imaging_Cohort_Key_MRI_10Oct2025.csv"
    )

    return tables


def get_diagnosis_trajectory(dxsum):
    """
    Determine diagnosis trajectory for each subject.

    Categories:
    - CN: Always cognitively normal (DIAGNOSIS=1)
    - MCI_stable: MCI that never progressed to AD
    - MCI_to_AD: MCI that progressed to AD
    - AD: Already AD at baseline or progressed to AD
    """
    dxsum = dxsum.copy()
    dxsum = dxsum.dropna(subset=['DIAGNOSIS'])
    dxsum['DIAGNOSIS'] = dxsum['DIAGNOSIS'].astype(int)

    trajectories = []

    for ptid, group in dxsum.groupby('PTID'):
        diagnoses = set(group['DIAGNOSIS'].unique())
        first_dx = group.sort_values('EXAMDATE')['DIAGNOSIS'].iloc[0]
        last_dx = group.sort_values('EXAMDATE')['DIAGNOSIS'].iloc[-1]

        if 3 in diagnoses:
            if first_dx == 3:
                trajectory = 'AD'
            else:
                trajectory = 'MCI_to_AD'
        elif 2 in diagnoses:
            trajectory = 'MCI_stable'
        else:
            trajectory = 'CN'

        trajectories.append({
            'PTID': ptid,
            'trajectory': trajectory,
            'first_dx': first_dx,
            'last_dx': last_dx
        })

    return pd.DataFrame(trajectories)


def get_existing_mri_paths():
    """Get MRI paths from existing ALL_4class_clinical.csv."""
    existing = pd.read_csv(OUTPUT_DIR / "ALL_4class_clinical.csv")

    def extract_image_id(path):
        match = re.search(r'_I(\d+)_', str(path))
        return int(match.group(1)) if match else None

    existing['image_id'] = existing['nii_path'].apply(extract_image_id)
    existing['nii_path'] = existing['nii_path'].str.replace(
        '/Volumes/KINGSTON/ADNI-skull',
        MRI_BASE_PATH,
        regex=False
    )

    return existing[['nii_path', 'image_id', 'Subject']].drop_duplicates()


def create_dataset(variant: str):
    """Create dataset for specified variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")

    config = VARIANTS[variant]
    print(f"\n{'='*60}")
    print(f"Creating: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Load tables
    print("\nLoading tables...")
    tables = load_tables()

    # Get trajectories
    print("Getting diagnosis trajectories...")
    trajectories = get_diagnosis_trajectory(tables['dxsum'])
    print(f"  Trajectory distribution:")
    print(trajectories['trajectory'].value_counts())

    # Get MRI paths
    print("\nGetting existing MRI paths...")
    mri_paths = get_existing_mri_paths()
    print(f"  Found {len(mri_paths)} unique MRI paths")

    # Merge with Key_MRI
    print("\nMerging with Key_MRI...")
    key_mri = tables['key_mri'][['image_id', 'subject_id', 'image_visit', 'image_date']]
    df = mri_paths.merge(key_mri, on='image_id', how='left')
    df['subject_id'] = df['subject_id'].fillna(df['Subject'])

    # Merge trajectories
    print("Merging with trajectories...")
    df = df.merge(
        trajectories[['PTID', 'trajectory', 'first_dx', 'last_dx']],
        left_on='subject_id',
        right_on='PTID',
        how='left'
    )

    # Merge tabular features from sc visits
    print("\nMerging tabular features...")
    my_table = tables['my_table'].copy()

    sc_cols = ['subject_id', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'MMSCORE', 'CDGLOBAL',
               'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
               'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
               'VSWEIGHT', 'VSHEIGHT']
    sc_cols = [c for c in sc_cols if c in my_table.columns]
    sc_data = my_table[my_table['visit'] == 'sc'][sc_cols].drop_duplicates(subset=['subject_id'], keep='first')

    bl_cols = ['subject_id', 'CLOCKSCOR', 'CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM',
               'DSPANFOR', 'DSPANBAC', 'BNTTOTAL', 'BCDEPRES', 'BCFAQ']
    bl_cols = [c for c in bl_cols if c in my_table.columns]
    bl_data = my_table[my_table['visit'] == 'bl'][bl_cols].drop_duplicates(subset=['subject_id'], keep='first')

    df = df.merge(sc_data, on='subject_id', how='left')
    df = df.merge(bl_data, on='subject_id', how='left')

    # Calculate BMI
    if 'VSWEIGHT' in df.columns and 'VSHEIGHT' in df.columns:
        df['BMI'] = df['VSWEIGHT'] / ((df['VSHEIGHT'] / 100) ** 2)

    # Merge age
    study_entry = tables['study_entry'][['subject_id', 'entry_age']]
    df = df.merge(study_entry, on='subject_id', how='left')
    df = df.rename(columns={'entry_age': 'AGE'})

    # Filter classes
    print(f"\nFiltering for classes: {config['classes']}")
    df = df[df['trajectory'].isin(config['classes'])]

    # Apply DX mapping if specified
    if 'dx_mapping' in config:
        df['DX'] = df['trajectory'].map(config['dx_mapping'])
    else:
        df['DX'] = df['trajectory']

    # Clean up columns
    df = df.rename(columns={'nii_path': 'scan_path'})
    keep_cols = ['subject_id', 'scan_path', 'trajectory', 'DX'] + TABULAR_FEATURES
    df = df[[c for c in keep_cols if c in df.columns]]

    # Remove duplicates
    df = df.drop_duplicates(subset=['subject_id'], keep='first')

    # Print summary
    print(f"\n=== Final Dataset ===")
    print(f"Total samples: {len(df)}")
    print(f"DX distribution:")
    print(df['DX'].value_counts())
    print(f"\nTabular features coverage:")
    for col in TABULAR_FEATURES:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = 100 * non_null / len(df)
            if pct > 0:
                print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")

    # Save
    output_path = OUTPUT_DIR / config['name']
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Create ADNI dataset CSVs')
    parser.add_argument('--variant', type=str, default='cn_ad_trajectory',
                        choices=list(VARIANTS.keys()),
                        help=f'Dataset variant: {list(VARIANTS.keys())}')
    parser.add_argument('--all', action='store_true',
                        help='Create all variants')
    args = parser.parse_args()

    if args.all:
        for variant in VARIANTS.keys():
            create_dataset(variant)
    else:
        create_dataset(args.variant)


if __name__ == "__main__":
    main()
