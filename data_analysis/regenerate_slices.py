#!/usr/bin/env python3
"""
Regenerate Whole Brain Slices Dataset from Complete ADNI Directory
This script will process ALL available ADNI MRI scans to create a comprehensive slice dataset
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_patient_info_from_filename(filename):
    """Extract patient information from ADNI filename"""
    # ADNI filename format: ADNI_site_S_subject_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_timestamp_Ssequence_Iimage.nii.gz
    # Patient ID should be: site_S_subject (e.g., 002_S_0619)
    match = re.match(r'ADNI_(\d+)_S_(\d+)_MR_.*?_(\d{14}).*?_S(\d+)_I(\d+)\.nii\.gz$', filename)
    if match:
        site_id = match.group(1)
        subject_id = match.group(2)  # This is the unique subject identifier
        timestamp = match.group(3)
        sequence_id = match.group(4)
        image_id = match.group(5)
        patient_id = f"{site_id}_S_{subject_id}"  # Full unique patient ID
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'subject_id': subject_id,
            'timestamp': timestamp,
            'sequence_id': sequence_id,
            'image_id': image_id
        }
    else:
        # Try a simpler pattern if the first doesn't match
        simple_match = re.match(r'ADNI_(\d+)_S_(\d+)_MR_.*?\.nii\.gz$', filename)
        if simple_match:
            site_id = simple_match.group(1)
            subject_id = simple_match.group(2)
            patient_id = f"{site_id}_S_{subject_id}"
            return {
                'patient_id': patient_id,
                'site_id': site_id,
                'subject_id': subject_id,
                'timestamp': 'unknown',
                'sequence_id': 'unknown',
                'image_id': 'unknown'
            }
    return None

def scan_adni_directory():
    """Scan ADNI directory and catalog all available MRI files"""
    base_path = "/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise"
    
    all_files = []
    
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_path = os.path.join(base_path, diagnosis)
        
        if not os.path.exists(diagnosis_path):
            logger.warning(f"Directory {diagnosis_path} not found!")
            continue
            
        logger.info(f"Scanning {diagnosis} directory...")
        
        nii_files = [f for f in os.listdir(diagnosis_path) if f.endswith('.nii.gz')]
        logger.info(f"Found {len(nii_files)} .nii.gz files in {diagnosis}")
        
        for filename in nii_files:
            file_path = os.path.join(diagnosis_path, filename)
            patient_info = extract_patient_info_from_filename(filename)
            
            if patient_info:
                file_record = {
                    'diagnosis': diagnosis,
                    'filename': filename,
                    'file_path': file_path,
                    **patient_info
                }
                all_files.append(file_record)
            else:
                logger.warning(f"Could not parse filename: {filename}")
    
    df = pd.DataFrame(all_files)
    logger.info(f"Total files cataloged: {len(df)}")
    
    if len(df) > 0:
        logger.info(f"Files by diagnosis:\n{df['diagnosis'].value_counts()}")
        logger.info(f"Unique patients by diagnosis:\n{df.groupby('diagnosis')['patient_id'].nunique()}")
    else:
        logger.error("No files were successfully parsed! Check the regex pattern.")
    
    return df

def create_slice_generation_plan():
    """Create a plan for generating slices from all ADNI files"""
    
    logger.info("Creating slice generation plan...")
    
    # Scan all available files
    df = scan_adni_directory()
    
    if df.empty or len(df) == 0:
        logger.error("No files found! Cannot proceed.")
        return None, None
    
    # Save catalog
    catalog_path = "/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/adni_full_catalog.csv"
    df.to_csv(catalog_path, index=False)
    logger.info(f"Full catalog saved to {catalog_path}")
    
    # Create slice generation plan
    output_dir = "/Users/tanguyvans/Desktop/umons/alzheimer/full_brain_slices_dataset"
    
    plan = {
        'input_files': len(df),
        'unique_patients': df['patient_id'].nunique(),
        'output_directory': output_dir,
        'expected_slices_per_volume': 170,  # Typical brain volume slice count
        'estimated_total_slices': len(df) * 170,
        'diagnosis_breakdown': df.groupby('diagnosis').agg({
            'filename': 'count',
            'patient_id': 'nunique'
        }).rename(columns={'filename': 'mri_count', 'patient_id': 'patient_count'})
    }
    
    return df, plan

def create_slice_processing_script():
    """Create a comprehensive slice processing script"""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = "/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/process_full_adni.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    logger.info(f"Slice processing script created: {script_path}")
    
    return script_path

def main():
    """Main function"""
    logger.info("Starting ADNI dataset regeneration planning...")
    
    # Create the plan
    df, plan = create_slice_generation_plan()
    
    if plan is None or df is None:
        logger.error("Failed to create slice generation plan.")
        return
    
    # Print the plan
    logger.info("\n" + "="*60)
    logger.info("SLICE GENERATION PLAN")
    logger.info("="*60)
    logger.info(f"Input MRI files: {plan['input_files']}")
    logger.info(f"Unique patients: {plan['unique_patients']}")
    logger.info(f"Output directory: {plan['output_directory']}")
    logger.info(f"Estimated total slices: {plan['estimated_total_slices']:,}")
    
    logger.info(f"\nBreakdown by diagnosis:")
    for diagnosis, stats in plan['diagnosis_breakdown'].iterrows():
        logger.info(f"  {diagnosis}: {stats['patient_count']} patients, {stats['mri_count']} MRIs")
    
    # Create the processing script
    script_path = create_slice_processing_script()
    
    logger.info(f"\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info(f"1. Review the catalog: data_analysis/adni_full_catalog.csv")
    logger.info(f"2. Run the processing script: python3 {script_path}")
    logger.info(f"3. This will create a new dataset with ALL {plan['unique_patients']} patients")
    logger.info(f"4. Expected output: ~{plan['estimated_total_slices']:,} brain slices")
    
    # Comparison with current dataset
    logger.info(f"\n" + "="*60)
    logger.info("COMPARISON WITH CURRENT DATASET")
    logger.info("="*60)
    logger.info(f"Current dataset: 136 patients")
    logger.info(f"Full ADNI dataset: {plan['unique_patients']} patients")
    logger.info(f"Improvement: {plan['unique_patients']/136:.1f}x more patients!")

if __name__ == "__main__":
    main()