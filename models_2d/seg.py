#!/usr/bin/env python3
"""
Hippocampus Segmentation using AAL Atlas
Based on the provided segmentation methodology
"""

import os
import numpy as np
import logging
import argparse

import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import entropy
import ants
import nilearn.datasets
import nilearn.image
import nilearn.plotting
from PIL import Image

def segment_hippocampus(input_image_path, output_folder):
    """
    Segment the hippocampus from an input image using the AAL atlas.
    
    Parameters:
    input_image_path (str): Path to the input NIfTI image file.
    output_folder (str): Path to the folder where output files will be saved.
    
    Returns:
    tuple: Paths to the left and right hippocampus mask files.
    """
    print(f"🧠 Segmenting hippocampus: {os.path.basename(input_image_path)}")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the AAL atlas
    print("📚 Loading AAL atlas...")
    aal = nilearn.datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nilearn.image.load_img(aal.maps)
    
    # Define the regions of interest (AAL atlas labels)
    Hippocampus_L = 4101
    Hippocampus_R = 4102
    
    print(f"   AAL atlas shape: {aal_img.shape}")
    print(f"   Hippocampus labels: L={Hippocampus_L}, R={Hippocampus_R}")
    
    # Load the input image
    print("📂 Loading input image...")
    input_img = nilearn.image.load_img(input_image_path)
    print(f"   Input image shape: {input_img.shape}")
    
    # Resample input image to match the AAL atlas
    print("🔄 Resampling to AAL atlas space...")
    resampled_img = nilearn.image.resample_to_img(input_img, aal_img, interpolation='nearest')
    print(f"   Resampled image shape: {resampled_img.shape}")
    
    # Create masks for left and right hippocampus
    print("🎯 Creating hippocampus masks...")
    aal_data = aal_img.get_fdata()
    
    mask_left = (aal_data == Hippocampus_L).astype(int)
    mask_right = (aal_data == Hippocampus_R).astype(int)
    
    left_voxels = np.sum(mask_left)
    right_voxels = np.sum(mask_right)
    
    print(f"   Left hippocampus mask: {left_voxels} voxels")
    print(f"   Right hippocampus mask: {right_voxels} voxels")
    
    # Create NIfTI images for the masks
    mask_left_img = nib.Nifti1Image(mask_left, aal_img.affine, aal_img.header)
    mask_right_img = nib.Nifti1Image(mask_right, aal_img.affine, aal_img.header)
    
    # Save the mask images
    left_mask_path = os.path.join(output_folder, 'hippocampus_left_mask.nii.gz')
    right_mask_path = os.path.join(output_folder, 'hippocampus_right_mask.nii.gz')
    nib.save(mask_left_img, left_mask_path)
    nib.save(mask_right_img, right_mask_path)
    
    print(f"💾 Masks saved: {left_mask_path}, {right_mask_path}")
    
    # Apply masks to the input image data
    print("✂️  Applying masks to extract hippocampus...")
    input_data = resampled_img.get_fdata()
    
    hippocampus_l_data = input_data * mask_left
    hippocampus_r_data = input_data * mask_right
    
    # Extract 2D chips for classification
    subject_id = os.path.basename(input_image_path).split('.')[0]
    chips_metadata = extract_2d_chips(resampled_img, mask_left, mask_right, output_folder, subject_id)
    
    # Calculate volumes
    voxel_size = np.prod(aal_img.header.get_zooms()[:3])  # mm³ per voxel
    left_volume = np.sum(mask_left) * voxel_size
    right_volume = np.sum(mask_right) * voxel_size
    total_volume = left_volume + right_volume
    
    print(f"📊 Hippocampus volumes:")
    print(f"   Left:  {left_volume:.1f} mm³")
    print(f"   Right: {right_volume:.1f} mm³") 
    print(f"   Total: {total_volume:.1f} mm³")
    print(f"   Asymmetry: {(left_volume-right_volume)/(left_volume+right_volume):.3f}")
    
    # Create new image objects for the masked data
    hippocampus_l_img = nilearn.image.new_img_like(aal_img, hippocampus_l_data)
    hippocampus_r_img = nilearn.image.new_img_like(aal_img, hippocampus_r_data)
    
    # Save the segmented hippocampus images
    left_hippo_path = os.path.join(output_folder, 'hippocampus_left_segmented.nii.gz')
    right_hippo_path = os.path.join(output_folder, 'hippocampus_right_segmented.nii.gz')
    nib.save(hippocampus_l_img, left_hippo_path)
    nib.save(hippocampus_r_img, right_hippo_path)
    
    print(f"💾 Segmented images saved: {left_hippo_path}, {right_hippo_path}")
    
    # Save volume results
    volume_results = {
        'subject_id': os.path.basename(input_image_path).split('.')[0],
        'left_hippocampus_mm3': left_volume,
        'right_hippocampus_mm3': right_volume,
        'total_hippocampus_mm3': total_volume,
        'asymmetry_index': (left_volume-right_volume)/(left_volume+right_volume) if total_volume > 0 else 0,
        'method': 'AAL_atlas'
    }
    
    import pandas as pd
    volume_df = pd.DataFrame([volume_results])
    volume_csv = os.path.join(output_folder, 'hippocampus_volumes.csv')
    volume_df.to_csv(volume_csv, index=False)
    print(f"📊 Volumes saved: {volume_csv}")
    
    # Save chips metadata
    if chips_metadata:
        chips_df = pd.DataFrame(chips_metadata)
        chips_csv = os.path.join(output_folder, 'hippocampus_chips.csv')
        chips_df.to_csv(chips_csv, index=False)
        print(f"🍟 Chips metadata saved: {chips_csv}")
    
    print(f"✅ Hippocampus segmentation and chip extraction completed!")
    
    return left_mask_path, right_mask_path, left_hippo_path, right_hippo_path

def find_middle_slice(data, axis):
    """Find middle slice containing data"""
    non_zero = np.any(data != 0, axis=(0, 1) if axis == 2 else ((1, 2) if axis == 0 else (0, 2)))
    if np.any(non_zero):
        return int(np.median(np.where(non_zero)[0]))
    else:
        return data.shape[axis] // 2

def hippocampal_windows_from_masks(aal_img, mask_left, mask_right, axial_half=10, coronal_half=8):
    """
    Calculate hippocampus-centered window ranges for 2D slice extraction.
    
    Parameters:
    aal_img: NIfTI image in AAL space
    mask_left: Left hippocampus binary mask (3D array)
    mask_right: Right hippocampus binary mask (3D array)
    axial_half: Half-width for axial window (±10 slices)
    coronal_half: Half-width for coronal window (±8 slices)
    
    Returns:
    tuple: ((axial_start, axial_end), (coronal_start, coronal_end))
    """
    # Combine left and right hippocampus masks
    hip = (mask_left > 0) | (mask_right > 0)
    
    # Get coordinates of all hippocampus voxels
    coords = np.array(np.where(hip)).T
    
    # If no hippocampus found, use image center
    if coords.size == 0:
        X, Y, Z = aal_img.shape
        return (Z//2-axial_half, Z//2+axial_half), (Y//2-coronal_half, Y//2+coronal_half)
    
    # Calculate centroid of hippocampus
    cx, cy, cz = coords.mean(axis=0)
    
    # Define windows around centroid
    axial_start = max(0, int(cz - axial_half))
    axial_end = min(aal_img.shape[2], int(cz + axial_half))
    
    coronal_start = max(0, int(cy - coronal_half))
    coronal_end = min(aal_img.shape[1], int(cy + coronal_half))
    
    return (axial_start, axial_end), (coronal_start, coronal_end)

def extract_2d_chips(input_img, mask_left, mask_right, output_folder, subject_id, target_size=224):
    """
    Extract 2D hippocampus-centered chips for classification.
    
    Parameters:
    input_img: NIfTI image in AAL space
    mask_left, mask_right: Hippocampus binary masks
    output_folder: Directory to save chips
    subject_id: Subject identifier
    target_size: Target size for resizing (default: 224x224)
    
    Returns:
    list: Metadata for extracted chips
    """
    print("🍟 Extracting 2D hippocampus-centered chips...")
    
    # Get input data
    input_data = input_img.get_fdata()
    
    # Calculate hippocampus-centered windows
    (axial_start, axial_end), (coronal_start, coronal_end) = hippocampal_windows_from_masks(
        input_img, mask_left, mask_right
    )
    
    print(f"   Axial window: {axial_start}-{axial_end}")
    print(f"   Coronal window: {coronal_start}-{coronal_end}")
    
    chips_data = []
    
    # Create output directories
    axial_dir = os.path.join(output_folder, 'axial_chips')
    coronal_dir = os.path.join(output_folder, 'coronal_chips')
    os.makedirs(axial_dir, exist_ok=True)
    os.makedirs(coronal_dir, exist_ok=True)
    
    # Extract axial slices (around hippocampus in Z direction)
    for z in range(axial_start + 1, axial_end - 1):  # Skip first/last to have i-1,i,i+1
        if z >= 1 and z < input_data.shape[2] - 1:
            # Create 3-channel chip [i-1, i, i+1]
            chip = np.stack([
                input_data[:, :, z-1],
                input_data[:, :, z],
                input_data[:, :, z+1]
            ], axis=-1)
            
            # Normalize chip
            chip = (chip - chip.min()) / (chip.max() - chip.min() + 1e-8)
            
            # Resize to target size (224x224x3)
            chip_resized = np.zeros((target_size, target_size, 3))
            for c in range(3):
                img = Image.fromarray((chip[:, :, c] * 255).astype(np.uint8))
                img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                chip_resized[:, :, c] = np.array(img_resized) / 255.0
            
            # Save as NIfTI file
            chip_path = os.path.join(axial_dir, f'{subject_id}_axial_z{z:03d}.nii.gz')
            chip_img = nib.Nifti1Image(chip_resized, affine=np.eye(4))
            nib.save(chip_img, chip_path)
            
            chips_data.append({
                'subject_id': subject_id,
                'view': 'axial',
                'slice_index': z,
                'chip_path': chip_path,
                'shape': f"{chip_resized.shape[0]}x{chip_resized.shape[1]}x{chip_resized.shape[2]}"
            })
    
    # Extract coronal slices (around hippocampus in Y direction)
    for y in range(coronal_start + 1, coronal_end - 1):  # Skip first/last to have i-1,i,i+1
        if y >= 1 and y < input_data.shape[1] - 1:
            # Create 3-channel chip [i-1, i, i+1]
            chip = np.stack([
                input_data[:, y-1, :],
                input_data[:, y, :],
                input_data[:, y+1, :]
            ], axis=-1)
            
            # Normalize chip
            chip = (chip - chip.min()) / (chip.max() - chip.min() + 1e-8)
            
            # Resize to target size (224x224x3)
            chip_resized = np.zeros((target_size, target_size, 3))
            for c in range(3):
                img = Image.fromarray((chip[:, :, c] * 255).astype(np.uint8))
                img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                chip_resized[:, :, c] = np.array(img_resized) / 255.0
            
            # Save as NIfTI file
            chip_path = os.path.join(coronal_dir, f'{subject_id}_coronal_y{y:03d}.nii.gz')
            chip_img = nib.Nifti1Image(chip_resized, affine=np.eye(4))
            nib.save(chip_img, chip_path)
            
            chips_data.append({
                'subject_id': subject_id,
                'view': 'coronal',
                'slice_index': y,
                'chip_path': chip_path,
                'shape': f"{chip_resized.shape[0]}x{chip_resized.shape[1]}x{chip_resized.shape[2]}"
            })
    
    print(f"   Extracted {len(chips_data)} chips ({len([c for c in chips_data if c['view']=='axial'])} axial, {len([c for c in chips_data if c['view']=='coronal'])} coronal)")
    
    return chips_data

def plot_hippocampus(left_hippo_path, right_hippo_path, output_path):
    """
    Plot the segmented left and right hippocampus.
    """
    print("📸 Creating hippocampus visualization...")
    
    # Load the hippocampus images
    left_hippo = nib.load(left_hippo_path)
    right_hippo = nib.load(right_hippo_path)
    
    left_data = left_hippo.get_fdata()
    right_data = right_hippo.get_fdata()
    
    # Find the middle slices where hippocampus is visible
    combined_data = left_data + right_data
    
    axial_slice = find_middle_slice(combined_data, 2)
    coronal_slice = find_middle_slice(combined_data, 1)
    sagittal_slice = find_middle_slice(combined_data, 0)
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Segmented Hippocampus (AAL Atlas)', fontsize=16)
    
    # Plot left hippocampus mask
    axes[0, 0].imshow(left_data[:, :, axial_slice].T, cmap='Reds', origin='lower')
    axes[0, 0].set_title('Left Hippocampus (Axial)')
    axes[0, 1].imshow(left_data[:, coronal_slice, :].T, cmap='Reds', origin='lower')
    axes[0, 1].set_title('Left Hippocampus (Coronal)')
    axes[0, 2].imshow(left_data[sagittal_slice, :, :].T, cmap='Reds', origin='lower')
    axes[0, 2].set_title('Left Hippocampus (Sagittal)')
    
    # Plot right hippocampus mask
    axes[1, 0].imshow(right_data[:, :, axial_slice].T, cmap='Blues', origin='lower')
    axes[1, 0].set_title('Right Hippocampus (Axial)')
    axes[1, 1].imshow(right_data[:, coronal_slice, :].T, cmap='Blues', origin='lower')
    axes[1, 1].set_title('Right Hippocampus (Coronal)')
    axes[1, 2].imshow(right_data[sagittal_slice, :, :].T, cmap='Blues', origin='lower')
    axes[1, 2].set_title('Right Hippocampus (Sagittal)')
    
    # Remove axis ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Hippocampus plot saved: {output_path}")
    
    # Create combined visualization
    combined_plot_path = output_path.replace('.png', '_combined.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Combined Hippocampus Segmentation', fontsize=16)
    
    # Combine masks (left=1, right=2)
    combined_mask = left_data + 2 * right_data
    
    axes[0].imshow(combined_mask[:, :, axial_slice].T, cmap='Set1', vmin=0, vmax=2, origin='lower')
    axes[0].set_title('Axial View')
    axes[1].imshow(combined_mask[:, coronal_slice, :].T, cmap='Set1', vmin=0, vmax=2, origin='lower')
    axes[1].set_title('Coronal View')
    axes[2].imshow(combined_mask[sagittal_slice, :, :].T, cmap='Set1', vmin=0, vmax=2, origin='lower')
    axes[2].set_title('Sagittal View')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Combined plot saved: {combined_plot_path}")

def main():
    """Main function for single image segmentation"""
    
    parser = argparse.ArgumentParser(description='Hippocampus Segmentation using AAL Atlas')
    parser.add_argument('--input', required=True, help='Input NIfTI file path')
    parser.add_argument('--output', default='hippocampus_segmentation', help='Output directory')
    
    args = parser.parse_args()
    
    print("🧠 Hippocampus Segmentation using AAL Atlas")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Perform segmentation
        left_mask, right_mask, left_hippo, right_hippo = segment_hippocampus(args.input, args.output)
        
        # Create visualization
        plot_path = os.path.join(args.output, 'hippocampus_segmentation.png')
        plot_hippocampus(left_hippo, right_hippo, plot_path)
        
        print(f"\\n✅ Segmentation completed successfully!")
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Volume results: {os.path.join(args.output, 'hippocampus_volumes.csv')}")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        logging.exception("Full traceback:")

if __name__ == "__main__":
    main()