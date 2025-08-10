#!/usr/bin/env python3
"""
Extract 2D Hippocampus Chips from ADNIDenoise Dataset
Processes all AD, CN, and MCI scans to create training dataset
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import glob

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our segmentation functions
from seg import segment_hippocampus, extract_2d_chips
import nibabel as nib
import numpy as np
import nilearn.datasets
import nilearn.image

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chip_extraction.log'),
            logging.StreamHandler()
        ]
    )

def get_diagnosis_from_path(file_path):
    """Extract diagnosis label from file path"""
    path_parts = Path(file_path).parts
    if 'AD' in path_parts:
        return 'AD'
    elif 'CN' in path_parts:
        return 'CN'
    elif 'MCI' in path_parts:
        return 'MCI'
    else:
        return 'Unknown'

def process_single_image(input_path, output_folder, diagnosis_label):
    """
    Process a single NIfTI image to extract hippocampus-centered chips
    
    Parameters:
    input_path: Path to input NIfTI file
    output_folder: Output directory for this subject
    diagnosis_label: AD, CN, or MCI
    
    Returns:
    dict: Processing results and metadata
    """
    try:
        print(f"🧠 Processing: {os.path.basename(input_path)}")
        
        # Create output directory for this subject
        subject_output = os.path.join(output_folder, diagnosis_label, 
                                     os.path.basename(input_path).split('.')[0])
        os.makedirs(subject_output, exist_ok=True)
        
        # Load AAL atlas
        aal = nilearn.datasets.fetch_atlas_aal(version='SPM12')
        aal_img = nilearn.image.load_img(aal.maps)
        
        # Load input image and resample to AAL space
        input_img = nilearn.image.load_img(input_path)
        resampled_img = nilearn.image.resample_to_img(input_img, aal_img, interpolation='nearest')
        
        # Create hippocampus masks
        aal_data = aal_img.get_fdata()
        Hippocampus_L = 4101
        Hippocampus_R = 4102
        
        mask_left = (aal_data == Hippocampus_L).astype(int)
        mask_right = (aal_data == Hippocampus_R).astype(int)
        
        left_voxels = np.sum(mask_left)
        right_voxels = np.sum(mask_right)
        
        # Check if hippocampus was found
        if left_voxels == 0 and right_voxels == 0:
            print(f"   ⚠️  No hippocampus found - skipping")
            return None
        
        # Calculate volumes
        voxel_size = np.prod(aal_img.header.get_zooms()[:3])
        left_volume = left_voxels * voxel_size
        right_volume = right_voxels * voxel_size
        total_volume = left_volume + right_volume
        
        # Extract 2D chips
        subject_id = os.path.basename(input_path).split('.')[0]
        chips_metadata = extract_2d_chips(resampled_img, mask_left, mask_right, subject_output, subject_id)
        
        # Add diagnosis label to each chip
        for chip in chips_metadata:
            chip['diagnosis'] = diagnosis_label
            chip['subject_full_id'] = subject_id
            chip['left_hippocampus_mm3'] = left_volume
            chip['right_hippocampus_mm3'] = right_volume
            chip['total_hippocampus_mm3'] = total_volume
        
        print(f"   ✅ Successfully extracted {len(chips_metadata)} chips")
        
        return {
            'subject_id': subject_id,
            'diagnosis': diagnosis_label,
            'input_path': input_path,
            'output_path': subject_output,
            'chips_count': len(chips_metadata),
            'left_hippocampus_mm3': left_volume,
            'right_hippocampus_mm3': right_volume,
            'total_hippocampus_mm3': total_volume,
            'chips_metadata': chips_metadata,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"   ❌ Error processing {os.path.basename(input_path)}: {e}")
        logging.exception(f"Error processing {input_path}")
        return {
            'subject_id': os.path.basename(input_path).split('.')[0],
            'diagnosis': diagnosis_label,
            'input_path': input_path,
            'output_path': None,
            'chips_count': 0,
            'error': str(e),
            'status': 'failed'
        }

def process_diagnosis_group(diagnosis_folder, output_folder, diagnosis_label):
    """
    Process all images in a diagnosis group (AD, CN, or MCI)
    
    Parameters:
    diagnosis_folder: Path to diagnosis folder (e.g., ADNIDenoise/AD)
    output_folder: Main output directory
    diagnosis_label: 'AD', 'CN', or 'MCI'
    
    Returns:
    list: Processing results for all subjects
    """
    print(f"\\n🏥 Processing {diagnosis_label} group...")
    
    # Find all .nii and .nii.gz files
    nii_files = []
    nii_files.extend(glob.glob(os.path.join(diagnosis_folder, "*.nii")))
    nii_files.extend(glob.glob(os.path.join(diagnosis_folder, "*.nii.gz")))
    
    print(f"   Found {len(nii_files)} NIfTI files")
    
    if len(nii_files) == 0:
        print(f"   ⚠️  No NIfTI files found in {diagnosis_folder}")
        return []
    
    results = []
    
    # Process each file with progress bar
    for nii_file in tqdm(nii_files, desc=f"Processing {diagnosis_label}"):
        result = process_single_image(nii_file, output_folder, diagnosis_label)
        if result:
            results.append(result)
    
    success_count = len([r for r in results if r.get('status') == 'success'])
    print(f"   📊 {diagnosis_label}: {success_count}/{len(nii_files)} processed successfully")
    
    return results

def create_consolidated_metadata(all_results, output_folder):
    """
    Create consolidated CSV files with all chip metadata and processing summary
    """
    print("\\n📊 Creating consolidated metadata...")
    
    # Collect all chip metadata
    all_chips = []
    summary_data = []
    
    for result in all_results:
        if result.get('status') == 'success' and result.get('chips_metadata'):
            all_chips.extend(result['chips_metadata'])
        
        # Summary data
        summary_data.append({
            'subject_id': result['subject_id'],
            'diagnosis': result['diagnosis'],
            'input_path': result['input_path'],
            'output_path': result.get('output_path', ''),
            'chips_count': result.get('chips_count', 0),
            'left_hippocampus_mm3': result.get('left_hippocampus_mm3', 0),
            'right_hippocampus_mm3': result.get('right_hippocampus_mm3', 0),
            'total_hippocampus_mm3': result.get('total_hippocampus_mm3', 0),
            'status': result.get('status', 'unknown'),
            'error': result.get('error', '')
        })
    
    # Create DataFrames and save
    if all_chips:
        chips_df = pd.DataFrame(all_chips)
        chips_csv_path = os.path.join(output_folder, 'all_hippocampus_chips.csv')
        chips_df.to_csv(chips_csv_path, index=False)
        print(f"   💾 Saved all chips metadata: {chips_csv_path}")
        print(f"   📈 Total chips extracted: {len(all_chips)}")
        print(f"   📊 By diagnosis: AD={len(chips_df[chips_df['diagnosis']=='AD'])}, "
              f"CN={len(chips_df[chips_df['diagnosis']=='CN'])}, "
              f"MCI={len(chips_df[chips_df['diagnosis']=='MCI'])}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_folder, 'processing_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"   💾 Saved processing summary: {summary_csv_path}")
    
    # Print final statistics
    total_subjects = len(summary_df)
    successful_subjects = len(summary_df[summary_df['status'] == 'success'])
    
    print(f"\\n✅ Extraction completed successfully!")
    print(f"   📁 Subjects processed: {successful_subjects}/{total_subjects}")
    print(f"   🍟 Total chips extracted: {len(all_chips) if all_chips else 0}")
    
    if successful_subjects > 0:
        avg_chips = len(all_chips) / successful_subjects if all_chips else 0
        print(f"   📊 Average chips per subject: {avg_chips:.1f}")

def main():
    """Main extraction function"""
    parser = argparse.ArgumentParser(description='Extract 2D Hippocampus Chips from ADNI Dataset')
    parser.add_argument('--input', default='ADNIDenoise', 
                       help='Input directory containing AD, CN, MCI folders')
    parser.add_argument('--output', default='hippocampus_chips_dataset', 
                       help='Output directory for extracted chips')
    parser.add_argument('--diagnosis', choices=['AD', 'CN', 'MCI'], 
                       help='Process only specific diagnosis group')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    print("🧠 2D Hippocampus Chip Extraction")
    print("=" * 40)
    print(f"📂 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    
    # Check input directory
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine which diagnosis groups to process
    diagnosis_groups = []
    if args.diagnosis:
        diagnosis_groups = [args.diagnosis]
    else:
        # Check which diagnosis folders exist
        for diag in ['AD', 'CN', 'MCI']:
            diag_folder = os.path.join(args.input, diag)
            if os.path.exists(diag_folder):
                diagnosis_groups.append(diag)
    
    if not diagnosis_groups:
        print("❌ No diagnosis folders found (AD, CN, MCI)")
        return 1
    
    print(f"🏥 Processing diagnosis groups: {', '.join(diagnosis_groups)}")
    
    # Process each diagnosis group
    all_results = []
    
    for diagnosis in diagnosis_groups:
        diagnosis_folder = os.path.join(args.input, diagnosis)
        results = process_diagnosis_group(diagnosis_folder, args.output, diagnosis)
        all_results.extend(results)
    
    # Create consolidated metadata
    if all_results:
        create_consolidated_metadata(all_results, args.output)
    else:
        print("⚠️  No results to process")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())