#!/usr/bin/env python3
"""
Build complete 4-class dataset from raw ADNI tabular data.

Uses DXSUM for longitudinal diagnosis to identify:
- CN: Cognitively Normal (stayed CN)
- MCI_stable: MCI patients who did NOT convert to AD
- MCI_to_AD: MCI patients who converted to AD
- AD: Alzheimer's Disease

Merges with existing clinical features CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_dxsum(tabular_dir):
    """Load diagnosis summary and identify patient trajectories"""
    dxsum_file = tabular_dir / '3D_MPRAGE_Imaging_Cohort_DXSUM_10Oct2025.csv'
    df = pd.read_csv(dxsum_file)

    # DIAGNOSIS: 1=CN, 2=MCI, 3=AD
    df = df[df['DIAGNOSIS'].notna()].copy()
    df['DIAGNOSIS'] = df['DIAGNOSIS'].astype(int)

    # Get baseline diagnosis
    baseline = df[df['VISCODE'] == 'bl'][['PTID', 'DIAGNOSIS', 'EXAMDATE']].copy()
    baseline = baseline.rename(columns={'DIAGNOSIS': 'BL_DX', 'EXAMDATE': 'BL_DATE'})

    # Get last diagnosis (most recent)
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    last = df.sort_values('EXAMDATE').groupby('PTID').last()[['DIAGNOSIS', 'EXAMDATE']].reset_index()
    last = last.rename(columns={'DIAGNOSIS': 'LAST_DX', 'EXAMDATE': 'LAST_DATE'})

    # Merge baseline and last
    trajectories = baseline.merge(last, on='PTID', how='inner')

    # Assign 4-class labels
    def assign_class(row):
        bl = row['BL_DX']
        last = row['LAST_DX']

        if bl == 1 and last == 1:
            return 'CN'
        elif bl == 1 and last == 2:
            return 'CN_to_MCI'  # Will exclude or treat as MCI_stable
        elif bl == 1 and last == 3:
            return 'CN_to_AD'  # Rare, treat as AD
        elif bl == 2 and last == 2:
            return 'MCI_stable'
        elif bl == 2 and last == 3:
            return 'MCI_to_AD'
        elif bl == 3:
            return 'AD'
        else:
            return 'Other'

    trajectories['CLASS_4'] = trajectories.apply(assign_class, axis=1)

    print(f"Patient trajectories from DXSUM:")
    print(trajectories['CLASS_4'].value_counts())

    return trajectories


def main():
    parser = argparse.ArgumentParser(description='Build 4-class dataset from raw ADNI data')
    parser.add_argument('--tabular-dir', type=str,
                        default='/Users/tanguyvans/Desktop/umons/alzheimer/data/tabular',
                        help='Directory with raw ADNI tabular CSVs')
    parser.add_argument('--clinical-csv', type=str,
                        default='/Users/tanguyvans/Desktop/umons/alzheimer/data/clinical_data_all_groups.csv',
                        help='Existing clinical features CSV')
    parser.add_argument('--output-csv', type=str,
                        default='/Users/tanguyvans/Desktop/umons/alzheimer/data/ALL_4class_clinical.csv',
                        help='Output CSV path')

    args = parser.parse_args()
    tabular_dir = Path(args.tabular_dir)

    print("="*60)
    print("Building 4-class dataset from raw ADNI data")
    print("="*60)

    # Load diagnosis trajectories
    trajectories = load_dxsum(tabular_dir)

    # Load existing clinical features (has all demographics, cognitive scores, etc.)
    print(f"\nLoading clinical features from {args.clinical_csv}")
    clinical = pd.read_csv(args.clinical_csv)
    print(f"Loaded {len(clinical)} samples with {len(clinical.columns)} columns")

    # Get unique patients from clinical data
    if 'Subject' in clinical.columns:
        clinical['PTID'] = clinical['Subject']
    elif 'PTID' not in clinical.columns:
        raise ValueError("No patient ID column found in clinical CSV")

    # Merge trajectories with clinical features
    # Keep only patients that have trajectory info
    df = clinical.merge(trajectories[['PTID', 'CLASS_4', 'BL_DX', 'LAST_DX']], on='PTID', how='inner')
    print(f"Matched {len(df)} samples with trajectory info")

    # Filter to main 4 classes
    main_classes = ['CN', 'MCI_stable', 'MCI_to_AD', 'AD']
    df_4class = df[df['CLASS_4'].isin(main_classes)].copy()

    # Also keep CN→MCI as MCI_stable (optional - they showed cognitive decline)
    cn_to_mci = df[df['CLASS_4'] == 'CN_to_MCI'].copy()
    if len(cn_to_mci) > 0:
        print(f"\nNote: {len(cn_to_mci)} CN→MCI samples (excluding from dataset)")

    cn_to_ad = df[df['CLASS_4'] == 'CN_to_AD'].copy()
    if len(cn_to_ad) > 0:
        print(f"Note: {len(cn_to_ad)} CN→AD samples (excluding from dataset)")

    # Add standard column names for compatibility
    df_4class['Subject'] = df_4class['PTID']
    df_4class['Group'] = df_4class['BL_DX'].map({1: 'CN', 2: 'MCI', 3: 'AD'})
    df_4class['DX'] = df_4class['LAST_DX'].map({1: 'CN', 2: 'MCI', 3: 'AD'})

    # Calculate AGE if not present
    if 'AGE' not in df_4class.columns and 'PTDOBYY' in df_4class.columns:
        if 'EXAMDATE' in df_4class.columns:
            df_4class['EXAMDATE'] = pd.to_datetime(df_4class['EXAMDATE'])
            df_4class['AGE'] = df_4class['EXAMDATE'].dt.year - df_4class['PTDOBYY']
        else:
            df_4class['AGE'] = 2015 - df_4class['PTDOBYY']

    # Calculate BMI if not present
    if 'BMI' not in df_4class.columns:
        if 'VSWEIGHT' in df_4class.columns and 'VSHEIGHT' in df_4class.columns:
            df_4class['BMI'] = df_4class['VSWEIGHT'] / ((df_4class['VSHEIGHT'] / 100) ** 2)

    print(f"\n{'='*60}")
    print("FINAL 4-CLASS DATASET")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_4class)}")
    print(f"\nClass distribution:")
    for cls in main_classes:
        count = sum(df_4class['CLASS_4'] == cls)
        pct = count / len(df_4class) * 100 if len(df_4class) > 0 else 0
        print(f"  {cls}: {count} ({pct:.1f}%)")

    print(f"\nUnique patients: {df_4class['PTID'].nunique()}")
    print(f"Features: {len(df_4class.columns)} columns")

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_4class.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
