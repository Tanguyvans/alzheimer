#!/usr/bin/env python3
"""
Extract Whole Brain 2D Slices for CNN Training
Following the GitHub repo approach: whole brain axial slices
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2

def extract_brain_slices(nifti_path, output_dir, plane='axial', num_slices=20):
    """
    Extract 2D slices from whole brain 3D volume
    Following the paper's approach
    """
    try:
        # Load NIfTI image
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        print(f"Processing {os.path.basename(nifti_path)}")
        print(f"  Original shape: {data.shape}")
        
        # Get patient info from filename
        filename = os.path.basename(nifti_path)
        if 'CN' in filename:
            diagnosis = 'CN'
            label = 0
        elif 'MCI' in filename:
            diagnosis = 'MCI' 
            label = 1
        elif 'AD' in filename:
            diagnosis = 'AD'
            label = 2
        else:
            print(f"  Warning: Cannot determine diagnosis from {filename}")
            return []
        
        # Extract patient ID
        patient_id = filename.split('_')[1] + '_' + filename.split('_')[2]
        
        # Create output directory
        patient_dir = os.path.join(output_dir, diagnosis, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        slices_info = []
        
        if plane == 'axial':
            # Extract axial slices (z-axis)
            total_slices = data.shape[2]
            
            # Select slices from middle region (skip top/bottom)
            start_idx = total_slices // 4
            end_idx = 3 * total_slices // 4
            
            # Sample uniformly
            slice_indices = np.linspace(start_idx, end_idx, num_slices, dtype=int)
            
            for i, z_idx in enumerate(slice_indices):
                slice_2d = data[:, :, z_idx]
                
                # Skip empty or very dark slices
                if slice_2d.max() <= slice_2d.min() or np.sum(slice_2d > 0) < 1000:
                    continue
                
                # Normalize slice
                slice_2d = normalize_slice(slice_2d)
                
                # Resize to 224x224 (standard CNN input)
                slice_2d = cv2.resize(slice_2d, (224, 224))
                
                # Save as NIfTI
                slice_filename = f"{patient_id}_axial_slice_{z_idx:03d}.nii.gz"
                slice_path = os.path.join(patient_dir, slice_filename)
                
                # Create 3D image with single slice for NIfTI format
                slice_3d = slice_2d[:, :, np.newaxis]
                slice_img = nib.Nifti1Image(slice_3d, affine=np.eye(4))
                nib.save(slice_img, slice_path)
                
                # Record slice info
                relative_path = os.path.join(diagnosis, patient_id, slice_filename)
                slices_info.append({
                    'patient_id': patient_id,
                    'scan_id': filename.replace('.nii.gz', ''),
                    'diagnosis': diagnosis,
                    'label': label,
                    'plane': plane,
                    'slice_idx': z_idx,
                    'slice_file': slice_filename,
                    'slice_path': slice_path,
                    'relative_path': relative_path
                })
        
        print(f"  Extracted {len(slices_info)} slices")
        return slices_info
        
    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")
        return []

def normalize_slice(slice_2d):
    """
    Normalize slice using min-max normalization (like the paper)
    """
    # Remove outliers
    p1, p99 = np.percentile(slice_2d[slice_2d > 0], [1, 99])
    slice_2d = np.clip(slice_2d, p1, p99)
    
    # Min-max normalization to [0, 1]
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    
    if slice_max > slice_min:
        slice_2d = (slice_2d - slice_min) / (slice_max - slice_min)
    
    return slice_2d

def main():
    parser = argparse.ArgumentParser(description='Extract whole brain 2D slices from ADNI data')
    parser.add_argument('--input_dir', type=str, 
                        default='/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise',
                        help='Input directory containing processed NIfTI files')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/tanguyvans/Desktop/umons/alzheimer/whole_brain_slices_dataset',
                        help='Output directory for extracted slices')
    parser.add_argument('--num_slices', type=int, default=20,
                        help='Number of slices to extract per volume')
    parser.add_argument('--plane', type=str, default='axial', choices=['axial', 'coronal', 'sagittal'],
                        help='Anatomical plane for slice extraction')
    
    args = parser.parse_args()
    
    print("🧠 Extracting Whole Brain 2D Slices")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Plane: {args.plane}")
    print(f"Slices per volume: {args.num_slices}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all NIfTI files from all diagnosis directories
    nifti_files = []
    # Extract all three classes
    diagnosis_dirs = ['AD', 'CN', 'MCI']
    
    for diagnosis in diagnosis_dirs:
        diagnosis_path = os.path.join(args.input_dir, diagnosis)
        if os.path.exists(diagnosis_path):
            print(f"Scanning {diagnosis} directory...")
            for root, dirs, files in os.walk(diagnosis_path):
                for file in files:
                    if file.endswith('.nii') or file.endswith('.nii.gz'):
                        nifti_files.append(os.path.join(root, file))
        else:
            print(f"Warning: {diagnosis} directory not found at {diagnosis_path}")
    
    # Also scan the main directory for any loose files
    for root, dirs, files in os.walk(args.input_dir):
        # Skip the diagnosis subdirectories we already processed
        if os.path.basename(root) not in diagnosis_dirs and root != args.input_dir:
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    nifti_files.append(os.path.join(root, file))
    
    print(f"Found {len(nifti_files)} NIfTI files")
    
    # Process all files
    all_slices_info = []
    
    for nifti_path in tqdm(nifti_files, desc="Processing volumes"):
        slices_info = extract_brain_slices(
            nifti_path, 
            args.output_dir, 
            plane=args.plane, 
            num_slices=args.num_slices
        )
        all_slices_info.extend(slices_info)
    
    # Create CSV metadata
    if all_slices_info:
        new_df = pd.DataFrame(all_slices_info)
        csv_path = os.path.join(args.output_dir, 'whole_brain_slices.csv')
        
        # Check if CSV already exists and append new data
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Appending {len(new_df)} new slices to existing {len(existing_df)} slices")
        else:
            df = new_df
            
        df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Extraction complete!")
        print(f"Total slices: {len(all_slices_info)}")
        print(f"CSV saved: {csv_path}")
        
        # Print class distribution
        print("\nClass distribution:")
        class_counts = df['diagnosis'].value_counts()
        for diagnosis, count in class_counts.items():
            print(f"  {diagnosis}: {count} slices")
        
        # Print patient distribution
        patient_counts = df.groupby('diagnosis')['patient_id'].nunique()
        print("\nPatient distribution:")
        for diagnosis, count in patient_counts.items():
            print(f"  {diagnosis}: {count} patients")
    
    else:
        print("❌ No slices extracted")

if __name__ == "__main__":
    main()