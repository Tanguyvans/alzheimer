#!/usr/bin/env python3
"""
Visualize slice selection strategies for Alzheimer's MRI data
Shows examples of different slice selection methods
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import random

def select_slice(volume, method='hippocampus'):
    """Select which slice to extract based on strategy"""
    depth = volume.shape[2]
    
    if method == 'middle':
        slice_idx = depth // 2
    elif method == 'hippocampus':
        # Hippocampus region - key for Alzheimer's detection
        hippocampus_start = int(depth * 0.45)
        hippocampus_end = int(depth * 0.55)
        slice_idx = (hippocampus_start + hippocampus_end) // 2
    elif method == 'max_intensity':
        # Find slice with maximum mean intensity
        max_intensity = 0
        slice_idx = depth // 2
        start = int(depth * 0.2)
        end = int(depth * 0.8)
        
        for idx in range(start, end):
            slice_intensity = np.mean(volume[:, :, idx])
            if slice_intensity > max_intensity:
                max_intensity = slice_intensity
                slice_idx = idx
    elif method == 'random':
        start = int(depth * 0.2)
        end = int(depth * 0.8)
        slice_idx = np.random.randint(start, end)
    else:
        slice_idx = depth // 2
        
    return slice_idx, volume[:, :, slice_idx]


def visualize_slice_selections(adni_dir='../ADNIDenoise'):
    """Visualize different slice selection methods"""
    
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
    
    # Create visualization
    methods = ['middle', 'hippocampus', 'max_intensity']
    
    fig, axes = plt.subplots(len(sample_files), len(methods) + 1, figsize=(16, 4*len(sample_files)))
    
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (diagnosis, file_path) in enumerate(sample_files):
        # Load volume
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        
        # Normalize volume
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Show sagittal view with slice positions
        sagittal_slice = volume[volume.shape[0]//2, :, :]
        ax = axes[i, 0]
        # Consistent brain orientation: neurological convention
        ax.imshow(sagittal_slice, cmap='gray', origin='upper', aspect='auto')
        ax.set_title(f'{diagnosis} - Sagittal View\n{os.path.basename(file_path)[:30]}...', fontsize=10)
        ax.set_xlabel('Anterior-Posterior')
        ax.set_ylabel('Inferior-Superior')
        
        # Add lines showing slice positions
        depth = volume.shape[2]
        colors = {'middle': 'red', 'hippocampus': 'green', 'max_intensity': 'blue'}
        
        for method in methods:
            slice_idx, _ = select_slice(volume, method)
            ax.axhline(y=slice_idx, color=colors[method], linestyle='--', alpha=0.7, label=method)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # Show selected slices
        for j, method in enumerate(methods):
            slice_idx, slice_data = select_slice(volume, method)
            
            ax = axes[i, j+1]
            # Consistent brain orientation: neurological convention  
            im = ax.imshow(slice_data, cmap='gray', origin='upper')
            ax.set_title(f'{method.capitalize()}\nSlice {slice_idx}/{depth}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar for first row
            if i == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Slice Selection Strategies for Alzheimer\'s MRI Classification', fontsize=16)
    plt.tight_layout()
    plt.savefig('slice_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed hippocampus region visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (diagnosis, file_path) in enumerate(sample_files):
        if i >= 3:
            break
            
        # Load volume
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Get hippocampus slice
        _, hippocampus_slice = select_slice(volume, 'hippocampus')
        
        ax = axes[i]
        im = ax.imshow(hippocampus_slice, cmap='gray', origin='lower')
        ax.set_title(f'{diagnosis} - Hippocampus Region', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Hippocampus Region Slices (Optimal for Alzheimer\'s Detection)', fontsize=16)
    plt.tight_layout()
    plt.savefig('hippocampus_slices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print slice statistics
    print("\nSlice Selection Statistics:")
    print("-" * 50)
    
    for diagnosis, file_path in sample_files:
        nii_img = nib.load(file_path)
        volume = nii_img.get_fdata()
        depth = volume.shape[2]
        
        print(f"\n{diagnosis} sample:")
        print(f"  Total slices: {depth}")
        
        for method in methods:
            slice_idx, slice_data = select_slice(volume, method)
            intensity = np.mean(slice_data[slice_data > 0.01])
            print(f"  {method}: slice {slice_idx} (intensity: {intensity:.3f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize slice selection strategies')
    parser.add_argument('--adni_dir', default='../ADNIDenoise', help='Path to ADNI directory')
    args = parser.parse_args()
    
    print("Generating slice selection visualizations...")
    visualize_slice_selections(args.adni_dir)
    print("\nVisualizations saved as:")
    print("  - slice_selection_comparison.png")
    print("  - hippocampus_slices.png")
