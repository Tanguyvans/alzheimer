#!/usr/bin/env python3
"""
Process ALL ADNI MRI scans to create comprehensive brain slice dataset
"""

import os
import sys
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import logging
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_mri(row, output_base_dir, min_slice_intensity=0.01):
    """Process a single MRI file to extract all brain slices"""
    
    try:
        # Load the NIfTI file
        nii_img = nib.load(row['file_path'])
        volume = nii_img.get_fdata()
        
        # Get dimensions
        depth = volume.shape[2]  # Usually the axial dimension
        
        # Create output directory for this patient/scan
        scan_output_dir = os.path.join(
            output_base_dir, 
            row['diagnosis'], 
            row['patient_id']
        )
        os.makedirs(scan_output_dir, exist_ok=True)
        
        slice_records = []
        
        # Extract each axial slice
        for slice_idx in range(depth):
            slice_data = volume[:, :, slice_idx]
            
            # Skip empty/low-intensity slices
            if np.max(slice_data) < min_slice_intensity:
                continue
                
            # Create unique slice filename
            slice_filename = f"{row['patient_id']}_scan_{row['image_id']}_axial_slice_{slice_idx:03d}.nii.gz"
            slice_path = os.path.join(scan_output_dir, slice_filename)
            
            # Save slice as NIfTI
            slice_nii = nib.Nifti1Image(slice_data, nii_img.affine)
            nib.save(slice_nii, slice_path)
            
            # Record slice information
            slice_record = {
                'slice_path': slice_path,
                'relative_path': os.path.relpath(slice_path, output_base_dir),
                'diagnosis': row['diagnosis'],
                'patient_id': row['patient_id'],
                'scan_id': row['image_id'],
                'slice_index': slice_idx,
                'original_file': row['filename'],
                'site_id': row['site_id'],
                'subject_id': row['subject_id'],
                'timestamp': row['timestamp'],
                'label': {'AD': 0, 'MCI': 1, 'CN': 2}[row['diagnosis']]
            }
            slice_records.append(slice_record)
        
        return slice_records, None
        
    except Exception as e:
        error_msg = f"Error processing {row['filename']}: {str(e)}"
        logger.error(error_msg)
        return [], error_msg

def main():
    """Main processing function"""
    
    # Load the file catalog
    catalog_path = "/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/adni_full_catalog.csv"
    
    if not os.path.exists(catalog_path):
        logger.error(f"Catalog file not found: {catalog_path}")
        logger.error("Please run regenerate_slices.py first to create the catalog")
        return
    
    df = pd.read_csv(catalog_path)
    logger.info(f"Loaded catalog with {len(df)} MRI files")
    
    # Set up output directory
    output_base_dir = "/Users/tanguyvans/Desktop/umons/alzheimer/full_brain_slices_dataset"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process files in parallel
    logger.info("Starting slice extraction...")
    
    all_slice_records = []
    error_records = []
    
    # Use partial to fix the output_base_dir parameter
    process_func = partial(process_single_mri, output_base_dir=output_base_dir)
    
    # Process with progress bar
    with tqdm(total=len(df), desc="Processing MRI files") as pbar:
        for idx, row in df.iterrows():
            slice_records, error = process_func(row)
            
            if error:
                error_records.append({'filename': row['filename'], 'error': error})
            else:
                all_slice_records.extend(slice_records)
            
            pbar.update(1)
            
            # Save progress periodically
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} files, generated {len(all_slice_records)} slices so far")
    
    logger.info(f"Processing complete!")
    logger.info(f"Total slices generated: {len(all_slice_records)}")
    logger.info(f"Files with errors: {len(error_records)}")
    
    # Save slice dataset
    if all_slice_records:
        slice_df = pd.DataFrame(all_slice_records)
        
        # Add some statistics
        logger.info("Slice distribution by diagnosis:")
        logger.info(slice_df['diagnosis'].value_counts())
        
        logger.info("Slice distribution by patient:")
        logger.info(slice_df.groupby('diagnosis')['patient_id'].nunique())
        
        # Save the complete slice dataset
        output_csv = os.path.join(output_base_dir, "full_brain_slices.csv")
        slice_df.to_csv(output_csv, index=False)
        logger.info(f"Complete slice dataset saved to: {output_csv}")
        
        # Create summary statistics
        summary = {
            'total_slices': len(slice_df),
            'total_patients': slice_df['patient_id'].nunique(),
            'total_scans': slice_df['scan_id'].nunique(),
            'slices_by_diagnosis': slice_df['diagnosis'].value_counts().to_dict(),
            'patients_by_diagnosis': slice_df.groupby('diagnosis')['patient_id'].nunique().to_dict(),
            'avg_slices_per_patient': slice_df.groupby(['diagnosis', 'patient_id']).size().groupby('diagnosis').mean().to_dict()
        }
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_base_dir, "dataset_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        logger.info("Dataset Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
    
    # Save error log
    if error_records:
        error_df = pd.DataFrame(error_records)
        error_path = os.path.join(output_base_dir, "processing_errors.csv")
        error_df.to_csv(error_path, index=False)
        logger.info(f"Error log saved to: {error_path}")

if __name__ == "__main__":
    main()
