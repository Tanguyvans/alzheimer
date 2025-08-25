#!/usr/bin/env python3
"""
Visualize brain image rotation for Alzheimer's MRI data
Compare original orientation vs 90-degree rotated images
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import random

def rotate_image(image, angle_degrees):
    """Rotate image by specified angle in degrees"""
    if angle_degrees == 90:
        return np.rot90(image)
    elif angle_degrees == 180:
        return np.rot90(image, 2)
    elif angle_degrees == 270:
        return np.rot90(image, 3)
    else:
        return image

def select_hippocampus_slice(volume):
    """Select slice from hippocampus region"""
    depth = volume.shape[2]
    hippocampus_start = int(depth * 0.45)
    hippocampus_end = int(depth * 0.55)
    slice_idx = (hippocampus_start + hippocampus_end) // 2
    return slice_idx, volume[:, :, slice_idx]

def visualize_brain_rotations(adni_dir='../ADNIDenoise'):
    """Compare brain orientations: original vs rotated"""
    
    # Get sample files from each diagnosis
    sample_files = []
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_path = os.path.join(adni_dir, diagnosis)
        if os.path.exists(diagnosis_path):
            files = glob(os.path.join(diagnosis_path, '*.nii.gz'))
            if files:
                sample_files.append((diagnosis, random.choice(files)))
    
    if not sample_files:
        print("No files found!")
        return
    
    # Rotation angles to test
    rotations = [0, 90, 180, 270]
    rotation_labels = ['Original', '90° CW', '180°', '270° CW']
    
    # Create visualization
    fig, axes = plt.subplots(len(sample_files), len(rotations), figsize=(16, 4*len(sample_files)))
    
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (diagnosis, file_path) in enumerate(sample_files):
        # Load volume
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        
        # Normalize volume
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Get hippocampus slice
        slice_idx, original_slice = select_hippocampus_slice(volume)
        
        for j, (rotation, label) in enumerate(zip(rotations, rotation_labels)):
            # Rotate the slice
            rotated_slice = rotate_image(original_slice, rotation)
            
            ax = axes[i, j]
            im = ax.imshow(rotated_slice, cmap='gray', origin='upper')
            
            if i == 0:  # Add title only for first row
                ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            
            # Add diagnosis label on left side
            if j == 0:
                ax.set_ylabel(f'{diagnosis}\n{os.path.basename(file_path)[:20]}...', 
                            fontsize=10, rotation=0, ha='right', va='center')
            
            ax.axis('off')
            
            # Add colorbar for first column
            if j == 0 and i == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Brain Image Rotation Comparison\n(Hippocampus Region Slices)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('brain_rotation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed comparison for HuggingFace orientation analysis
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Use first sample file for detailed analysis
    diagnosis, file_path = sample_files[0]
    nii_img = nib.load(file_path)
    volume = nii_img.get_fdata()
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Get two different slice types
    slice_types = ['hippocampus', 'middle']
    slice_names = ['Hippocampus Region', 'Middle Slice']
    
    for row, (slice_type, slice_name) in enumerate(zip(slice_types, slice_names)):
        if slice_type == 'hippocampus':
            _, base_slice = select_hippocampus_slice(volume)
        else:
            middle_idx = volume.shape[2] // 2
            base_slice = volume[:, :, middle_idx]
        
        for col, (rotation, label) in enumerate(zip(rotations, rotation_labels)):
            rotated_slice = rotate_image(base_slice, rotation)
            
            ax = axes[row, col]
            im = ax.imshow(rotated_slice, cmap='gray', origin='upper')
            
            if row == 0:
                ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{slice_name}', fontsize=11, fontweight='bold')
            
            ax.axis('off')
            
            # Add rotation info
            ax.text(0.02, 0.98, f'{rotation}°', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=9, verticalalignment='top')
    
    plt.suptitle(f'Brain Rotation Analysis for HuggingFace Compatibility\n{diagnosis} Sample: {os.path.basename(file_path)}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('huggingface_rotation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recommendations
    print("\\n" + "="*60)
    print("BRAIN ROTATION ANALYSIS FOR HUGGINGFACE MODELS")
    print("="*60)
    print("\\nRotation Options:")
    print("• Original (0°):   Standard neurological convention")
    print("• 90° Clockwise:   Portrait orientation, may match HuggingFace training")
    print("• 180°:            Upside-down, rarely used")
    print("• 270° Clockwise:  Landscape orientation")
    print("\\nRecommendation:")
    print("Test both original (0°) and 90° rotations with HuggingFace models")
    print("to determine which orientation matches the pre-training data.")
    print("\\nFiles generated:")
    print("• brain_rotation_comparison.png - Multi-sample comparison")
    print("• huggingface_rotation_analysis.png - Detailed rotation analysis")

def create_rotation_function():
    """Create a reusable rotation function for training scripts"""
    rotation_code = '''
def apply_rotation_transform(image, rotation_degrees=90):
    """
    Apply rotation to brain image for HuggingFace model compatibility
    
    Args:
        image: 2D numpy array (H, W) 
        rotation_degrees: 0, 90, 180, or 270 degrees
    
    Returns:
        rotated_image: 2D numpy array
    """
    if rotation_degrees == 90:
        return np.rot90(image)
    elif rotation_degrees == 180:
        return np.rot90(image, 2) 
    elif rotation_degrees == 270:
        return np.rot90(image, 3)
    else:
        return image

# Usage in dataset preprocessing:
# rotated_slice = apply_rotation_transform(brain_slice, rotation_degrees=90)
'''
    
    with open('rotation_utils.py', 'w') as f:
        f.write('import numpy as np\n\n')
        f.write(rotation_code)
    
    print("\\nCreated rotation_utils.py with reusable rotation function")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize brain image rotations')
    parser.add_argument('--adni_dir', default='../ADNIDenoise', help='Path to ADNI directory')
    args = parser.parse_args()
    
    print("Analyzing brain image rotations for HuggingFace compatibility...")
    visualize_brain_rotations(args.adni_dir)
    create_rotation_function()
    print("\\nRotation analysis complete!")