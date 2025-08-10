#!/usr/bin/env python3
"""
Create clinical features in exact notebook format
"""

import pandas as pd
import numpy as np

def main():
    # Load the existing mock data
    df = pd.read_csv('/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/mock_adni_clinical_data.csv')
    
    # Create APOE Genotype column (exact notebook format)
    conditions = [
        df['APOE4'] == 0,  # No APOE4 alleles
        df['APOE4'] == 1,  # One APOE4 allele  
        df['APOE4'] == 2   # Two APOE4 alleles
    ]
    choices = ['3,3', '3,4', '4,4']
    df['APOE Genotype'] = np.select(conditions, choices, default='3,3')
    
    # Select exact columns from notebook
    notebook_df = df[['AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'APOE4', 'MMSE', 'APOE Genotype', 'Group']]
    
    # Save in notebook format
    output_path = '/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise/adni_clinical_features.csv'
    notebook_df.to_csv(output_path, index=False)
    
    print(f"✅ Notebook format clinical features saved to: {output_path}")
    print(f"Shape: {notebook_df.shape}")
    print(f"Columns: {list(notebook_df.columns)}")
    print(f"\nClass distribution:")
    print(notebook_df['Group'].value_counts())

if __name__ == "__main__":
    main()