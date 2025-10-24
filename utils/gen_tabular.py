#!/usr/bin/env python3
"""
Generate Clinical Tabular Data for Alzheimer's Analysis

This script merges multiple ADNI clinical data files to create a comprehensive
tabular dataset with demographics, cognitive tests, physical measurements, and
MRI file paths.

Author: Alzheimer's Analysis Pipeline
Date: 2025-10-10
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TabularDataGenerator:
    """
    Generate comprehensive tabular data from ADNI clinical files.
    """
    
    def __init__(self, tabular_dir, mri_dir=None, output_path=None):
        """
        Initialize the generator.
        
        Args:
            tabular_dir: Path to directory containing ADNI CSV files
            mri_dir: Path to directory containing processed MRI files (optional)
            output_path: Path for output CSV file
        """
        self.tabular_dir = Path(tabular_dir)
        self.mri_dir = Path(mri_dir) if mri_dir else None
        self.output_path = Path(output_path) if output_path else Path('clinical_tabular_data.csv')
        
        # Data containers
        self.study_entry_df = None
        self.neurobat_df = None
        self.physical_df = None
        self.vitals_df = None
        self.medhist_df = None
        self.inithealth_df = None
        self.dxsum_df = None
        self.mri_df = None
        
        print("="*80)
        print("ADNI TABULAR DATA GENERATOR")
        print("="*80)
    
    def load_all_files(self):
        """Load all ADNI clinical data files."""
        print("\n📂 Loading ADNI clinical data files...")
        
        # 1. Study Entry (Demographics & Initial Group)
        entry_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_Study_Entry_10Oct2025.csv"
        if entry_file.exists():
            self.study_entry_df = pd.read_csv(entry_file)
            print(f"✅ Study Entry: {len(self.study_entry_df)} records")
        else:
            print(f"⚠️  Study Entry file not found")
        
        # 2. Neuropsychological Battery (Cognitive Tests)
        neurobat_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_NEUROBAT_10Oct2025.csv"
        if neurobat_file.exists():
            self.neurobat_df = pd.read_csv(neurobat_file)
            print(f"✅ NEUROBAT: {len(self.neurobat_df)} records")
        else:
            print(f"⚠️  NEUROBAT file not found")
        
        # 3. Physical Exam
        physical_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_PHYSICAL_10Oct2025.csv"
        if physical_file.exists():
            self.physical_df = pd.read_csv(physical_file)
            print(f"✅ Physical: {len(self.physical_df)} records")
        else:
            print(f"⚠️  Physical file not found")
        
        # 4. Vitals (Weight, Height, etc.)
        vitals_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_VITALS_10Oct2025.csv"
        if vitals_file.exists():
            self.vitals_df = pd.read_csv(vitals_file)
            print(f"✅ Vitals: {len(self.vitals_df)} records")
        else:
            print(f"⚠️  Vitals file not found")
        
        # 5. Medical History
        medhist_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_MEDHIST_10Oct2025.csv"
        if medhist_file.exists():
            self.medhist_df = pd.read_csv(medhist_file)
            print(f"✅ Medical History: {len(self.medhist_df)} records")
        else:
            print(f"⚠️  Medical History file not found")
        
        # 6. Initial Health Screen
        inithealth_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_INITHEALTH_10Oct2025.csv"
        if inithealth_file.exists():
            self.inithealth_df = pd.read_csv(inithealth_file)
            print(f"✅ Initial Health: {len(self.inithealth_df)} records")
        else:
            print(f"⚠️  Initial Health file not found")
        
        # 7. Diagnosis Summary
        dxsum_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_DXSUM_10Oct2025.csv"
        if dxsum_file.exists():
            self.dxsum_df = pd.read_csv(dxsum_file)
            print(f"✅ DXSUM: {len(self.dxsum_df)} records")
        else:
            print(f"⚠️  DXSUM file not found")
        
        # 8. MRI Key
        mri_file = self.tabular_dir / "3D_MPRAGE_Imaging_Cohort_Key_MRI_10Oct2025.csv"
        if mri_file.exists():
            self.mri_df = pd.read_csv(mri_file)
            print(f"✅ MRI Key: {len(self.mri_df)} records")
        else:
            print(f"⚠️  MRI Key file not found")
    
    def aggregate_patient_data(self, df, patient_col='PTID', agg_method='mean'):
        """
        Aggregate multiple records per patient into single record.
        
        Args:
            df: DataFrame to aggregate
            patient_col: Column containing patient ID
            agg_method: Aggregation method ('mean', 'median', 'first', 'last')
        
        Returns:
            Aggregated DataFrame
        """
        if df is None or len(df) == 0:
            return None
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Remove patient_col from aggregation
        if patient_col in numeric_cols:
            numeric_cols.remove(patient_col)
        if patient_col in non_numeric_cols:
            non_numeric_cols.remove(patient_col)
        
        agg_dict = {}
        
        # Numeric columns: mean or median
        for col in numeric_cols:
            if agg_method == 'mean':
                agg_dict[col] = 'mean'
            elif agg_method == 'median':
                agg_dict[col] = 'median'
            elif agg_method == 'first':
                agg_dict[col] = 'first'
            elif agg_method == 'last':
                agg_dict[col] = 'last'
        
        # Non-numeric columns: first value
        for col in non_numeric_cols:
            agg_dict[col] = 'first'
        
        # Group by patient
        aggregated = df.groupby(patient_col).agg(agg_dict).reset_index()
        
        return aggregated
    
    def extract_features(self):
        """Extract and merge key features from all data sources."""
        print("\n🔧 Extracting and merging features...")
        
        # Start with Study Entry (base demographics)
        if self.study_entry_df is not None:
            base_df = self.study_entry_df.copy()
            base_df = base_df.rename(columns={
                'subject_id': 'PTID',
                'entry_research_group': 'Group'
            })
            print(f"  Base: {len(base_df)} patients from Study Entry")
        else:
            print("❌ Cannot proceed without Study Entry file")
            return None
        
        # Extract Initial Health demographics
        if self.inithealth_df is not None:
            # Aggregate by patient
            health_agg = self.aggregate_patient_data(self.inithealth_df, patient_col='PTID', agg_method='first')
            
            # Select demographic columns
            demo_cols = ['PTID', 'PTRACCAT', 'PTGENDER', 'PTDOBYY', 'PTHAND', 
                        'PTMARRY', 'PTEDUCAT', 'PTNOTRT', 'PTTLANG']
            demo_cols = [col for col in demo_cols if col in health_agg.columns]
            health_demo = health_agg[demo_cols]
            
            # Merge with base
            base_df = base_df.merge(health_demo, on='PTID', how='left')
            print(f"  + Initial Health demographics")
        
        # Extract Medical History
        if self.medhist_df is not None:
            # Aggregate by patient
            medhist_agg = self.aggregate_patient_data(self.medhist_df, patient_col='PTID', agg_method='first')
            
            # Select medical history columns
            mh_cols = ['PTID', 'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG',
                      'MH4CARD', 'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA']
            mh_cols = [col for col in mh_cols if col in medhist_agg.columns]
            medhist_selected = medhist_agg[mh_cols]
            
            # Merge
            base_df = base_df.merge(medhist_selected, on='PTID', how='left')
            print(f"  + Medical History")
        
        # Extract Vitals (Weight, Height)
        if self.vitals_df is not None:
            # Aggregate by patient
            vitals_agg = self.aggregate_patient_data(self.vitals_df, patient_col='PTID', agg_method='mean')
            
            # Select vitals columns
            vital_cols = ['PTID', 'VSWEIGHT', 'VSHEIGHT', 'VSWTUNIT', 'VSHTUNIT']
            vital_cols = [col for col in vital_cols if col in vitals_agg.columns]
            vitals_selected = vitals_agg[vital_cols]
            
            # Merge
            base_df = base_df.merge(vitals_selected, on='PTID', how='left')
            print(f"  + Vitals (weight, height)")
        
        # Extract Neuropsychological Tests
        if self.neurobat_df is not None:
            # Aggregate by patient (mean of all tests)
            neurobat_agg = self.aggregate_patient_data(self.neurobat_df, patient_col='PTID', agg_method='mean')
            
            # Select key cognitive test columns
            cognitive_cols = ['PTID', 'CLOCKSCOR', 'LIMMTOTAL', 'DSPANFOR', 'DSPANBAC',
                            'CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'BNTTOTAL',
                            'MMSCORE', 'CDGLOBAL', 'BCFAQ', 'BCDEPRES']
            
            # Add MMSCORE, CDGLOBAL, BCFAQ, BCDEPRES if they exist in other files
            # For now, use available columns
            cognitive_cols = [col for col in cognitive_cols if col in neurobat_agg.columns]
            neurobat_selected = neurobat_agg[cognitive_cols]
            
            # Merge
            base_df = base_df.merge(neurobat_selected, on='PTID', how='left')
            print(f"  + Neuropsychological tests ({len(cognitive_cols)-1} tests)")
        
        # Add placeholders for missing commonly-used columns
        if 'MMSCORE' not in base_df.columns:
            base_df['MMSCORE'] = np.nan
        if 'CDGLOBAL' not in base_df.columns:
            base_df['CDGLOBAL'] = np.nan
        if 'BCFAQ' not in base_df.columns:
            base_df['BCFAQ'] = np.nan
        if 'BCDEPRES' not in base_df.columns:
            base_df['BCDEPRES'] = np.nan
        if 'TRABERRCOM' not in base_df.columns:
            base_df['TRABERRCOM'] = np.nan
        
        # Add Subject column (same as PTID)
        base_df['Subject'] = base_df['PTID']
        
        print(f"\n✅ Merged dataset: {len(base_df)} patients, {len(base_df.columns)} features")
        
        return base_df
    
    def link_mri_files(self, clinical_df):
        """Link clinical data with MRI file paths."""
        print("\n🔗 Linking MRI files...")
        
        if self.mri_df is None:
            print("⚠️  No MRI metadata available")
            return clinical_df
        
        # Group MRI by subject
        mri_by_subject = self.mri_df.groupby('subject_id').size().reset_index(name='mri_count')
        print(f"  Found MRI scans for {len(mri_by_subject)} subjects")
        
        # If MRI directory is provided, try to find actual files
        if self.mri_dir and self.mri_dir.exists():
            print(f"  Searching for NIfTI files in {self.mri_dir}")
            
            # Create expanded dataset: one row per MRI scan
            expanded_rows = []
            
            for _, row in clinical_df.iterrows():
                ptid = row['PTID']
                
                # Find MRI files for this patient
                patient_mris = self.mri_df[self.mri_df['subject_id'] == ptid]
                
                if len(patient_mris) == 0:
                    # No MRI found, add row with null path
                    row_dict = row.to_dict()
                    row_dict['nii_path'] = None
                    expanded_rows.append(row_dict)
                else:
                    # Add one row per MRI scan
                    for _, mri_row in patient_mris.iterrows():
                        row_dict = row.to_dict()
                        
                        # Try to find the actual NIfTI file
                        # Search in AD, CN, MCI subdirectories
                        found_path = None
                        for group_dir in ['AD', 'CN', 'MCI']:
                            group_path = self.mri_dir / group_dir
                            if group_path.exists():
                                # Look for files with patient ID
                                matching_files = list(group_path.glob(f"*{ptid}*.nii.gz"))
                                if matching_files:
                                    found_path = str(matching_files[0])
                                    break
                        
                        row_dict['nii_path'] = found_path
                        expanded_rows.append(row_dict)
            
            expanded_df = pd.DataFrame(expanded_rows)
            print(f"  Expanded to {len(expanded_df)} rows (one per MRI scan)")
            print(f"  Found paths for {expanded_df['nii_path'].notna().sum()} scans")
            
            return expanded_df
        else:
            # Just add mri_count column
            clinical_df = clinical_df.merge(mri_by_subject, left_on='PTID', right_on='subject_id', how='left')
            clinical_df['mri_count'] = clinical_df['mri_count'].fillna(0).astype(int)
            clinical_df['nii_path'] = None
            
            return clinical_df
    
    def generate_dataset(self):
        """Generate complete clinical tabular dataset."""
        print("\n" + "="*80)
        print("GENERATING CLINICAL TABULAR DATASET")
        print("="*80)
        
        # 1. Load all files
        self.load_all_files()
        
        # 2. Extract and merge features
        clinical_df = self.extract_features()
        
        if clinical_df is None:
            print("\n❌ Failed to generate dataset")
            return False
        
        # 3. Link MRI files
        final_df = self.link_mri_files(clinical_df)
        
        # 4. Clean up and reorder columns
        # Reorder to match AD_CN_clinical_data.csv structure
        preferred_order = [
            'PTID', 'PTRACCAT', 'PTGENDER', 'PTDOBYY', 'PTHAND', 'PTMARRY', 
            'PTEDUCAT', 'PTNOTRT', 'PTTLANG',
            'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD', 
            'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
            'VSWEIGHT', 'VSHEIGHT', 'VSWTUNIT', 'VSHTUNIT',
            'MMSCORE', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
            'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
            'CDGLOBAL', 'BCFAQ', 'BCDEPRES',
            'Subject', 'Group', 'nii_path'
        ]
        
        # Keep only columns that exist
        columns_to_keep = [col for col in preferred_order if col in final_df.columns]
        
        # Add any remaining columns
        remaining_cols = [col for col in final_df.columns if col not in columns_to_keep]
        all_columns = columns_to_keep + remaining_cols
        
        final_df = final_df[all_columns]
        
        # 5. Save to CSV
        final_df.to_csv(self.output_path, index=False)
        
        # 6. Print summary
        print("\n" + "="*80)
        print("✅ DATASET GENERATION COMPLETE")
        print("="*80)
        print(f"\nOutput: {self.output_path}")
        print(f"Total records: {len(final_df)}")
        print(f"Unique patients: {final_df['PTID'].nunique()}")
        print(f"Features: {len(final_df.columns)}")
        
        print("\n📊 Diagnosis Distribution:")
        if 'Group' in final_df.columns:
            for group, count in final_df['Group'].value_counts().items():
                percentage = (count / len(final_df)) * 100
                print(f"  {group}: {count} ({percentage:.1f}%)")
        
        if 'nii_path' in final_df.columns:
            with_mri = final_df['nii_path'].notna().sum()
            print(f"\n🧠 MRI Files Linked: {with_mri}/{len(final_df)} ({with_mri/len(final_df)*100:.1f}%)")
        
        print(f"\n💾 Saved to: {self.output_path}")
        print("="*80 + "\n")
        
        return True


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate clinical tabular dataset from ADNI files'
    )
    parser.add_argument(
        '--tabular_dir',
        type=str,
        default='/Users/tanguyvans/Desktop/umons/alzheimer/data/tabular',
        help='Directory containing ADNI CSV files'
    )
    parser.add_argument(
        '--mri_dir',
        type=str,
        default='/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise',
        help='Directory containing processed MRI files (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/tanguyvans/Desktop/umons/alzheimer/data/clinical_tabular_data.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--no-mri',
        action='store_true',
        help='Skip MRI file linking'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = TabularDataGenerator(
        tabular_dir=args.tabular_dir,
        mri_dir=args.mri_dir if not args.no_mri else None,
        output_path=args.output
    )
    
    # Generate dataset
    success = generator.generate_dataset()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())