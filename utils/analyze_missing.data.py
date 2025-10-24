#!/usr/bin/env python3
"""Analyze missing values in clinical_tabular_data.csv"""

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('/Users/tanguyvans/Desktop/umons/alzheimer/data/clinical_tabular_data.csv')

print("="*80)
print("MISSING DATA ANALYSIS")
print("="*80)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Calculate missing values
missing_stats = []
for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    if missing_count > 0:
        missing_stats.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Percent': missing_pct,
            'Data_Type': str(df[col].dtype)
        })

missing_df = pd.DataFrame(missing_stats)
missing_df = missing_df.sort_values('Missing_Percent', ascending=False)

print(f"\n📊 Columns with missing data: {len(missing_df)}/{len(df.columns)}")
print("\n" + "="*80)
print("MISSING VALUES BY COLUMN")
print("="*80)

for _, row in missing_df.iterrows():
    bar = '█' * int(row['Missing_Percent'] / 2)
    print(f"{row['Column']:20s}: {row['Missing_Count']:5d} ({row['Missing_Percent']:5.1f}%) {bar}")

# Group by category
print("\n" + "="*80)
print("MISSING VALUES BY CATEGORY")
print("="*80)

categories = {
    'Demographics': ['PTRACCAT', 'PTGENDER', 'PTDOBYY', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTNOTRT', 'PTTLANG'],
    'Medical History': ['MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD', 'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA'],
    'Vitals': ['VSWEIGHT', 'VSHEIGHT', 'VSWTUNIT', 'VSHTUNIT'],
    'Cognitive Tests': ['MMSCORE', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC', 'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC', 'LIMMTOTAL'],
    'Clinical Scores': ['CDGLOBAL', 'BCFAQ', 'BCDEPRES'],
    'MRI': ['nii_path']
}

for category, cols in categories.items():
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        total_cells = len(df) * len(available_cols)
        missing_cells = df[available_cols].isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        print(f"\n{category}:")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Missing: {missing_cells:,} ({missing_pct:.1f}%)")
        
        # Detail by column
        for col in available_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                pct = (missing / len(df)) * 100
                if missing > 0:
                    print(f"    • {col}: {missing} ({pct:.1f}%)")

# Rows with no missing values
complete_rows = df.dropna().shape[0]
print(f"\n" + "="*80)
print(f"Complete rows (no missing values): {complete_rows}/{len(df)} ({complete_rows/len(df)*100:.1f}%)")

# Most problematic fields
print(f"\n" + "="*80)
print("TOP 10 MOST INCOMPLETE FIELDS:")
print("="*80)
if len(missing_df) > 0:
    for i, row in missing_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['Column']:20s}: {row['Missing_Percent']:5.1f}% missing")