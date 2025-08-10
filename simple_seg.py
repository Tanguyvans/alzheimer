#!/usr/bin/env python3
"""
Simple Hippocampus Segmentation using template-based approach
Works with Python libraries only (no FSL/FreeSurfer required)
"""

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt
import os
import argparse

def load_brain_image(image_path):
    """Load and preprocess brain image"""
    print(f"📂 Loading image: {os.path.basename(image_path)}")
    
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        
        print(f"   Image shape: {data.shape}")
        print(f"   Voxel size: {img.header.get_zooms()[:3]} mm")
        print(f"   Intensity range: {data.min():.1f} - {data.max():.1f}")
        
        return img, data
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None, None

def brain_extraction(data, threshold_percentile=2):
    """Simple brain extraction using intensity thresholding"""
    print("🧠 Performing brain extraction...")
    
    # Calculate threshold (remove background)
    threshold = np.percentile(data[data > 0], threshold_percentile)
    
    # Create brain mask
    brain_mask = data > threshold
    
    # Clean up mask with morphological operations
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_closing(brain_mask, morphology.ball(2))
    
    print(f"   Brain voxels: {np.sum(brain_mask):,}")
    
    return brain_mask

def create_hippocampus_roi(data, brain_mask, img_affine):
    """Create approximate hippocampus ROI using anatomical constraints"""
    print("🎯 Creating hippocampus ROI...")
    
    # Get brain dimensions
    x_size, y_size, z_size = data.shape
    
    # Hippocampus is located in medial temporal lobe
    # Approximate coordinates in image space (these are rough estimates)
    
    # Define search regions for left and right hippocampus
    # These coordinates assume RAS+ orientation
    left_roi = {
        'x_range': (int(x_size * 0.35), int(x_size * 0.48)),  # Left hemisphere
        'y_range': (int(y_size * 0.25), int(y_size * 0.65)),  # Anterior-posterior
        'z_range': (int(z_size * 0.15), int(z_size * 0.45))   # Inferior-superior
    }
    
    right_roi = {
        'x_range': (int(x_size * 0.52), int(x_size * 0.65)),  # Right hemisphere
        'y_range': (int(y_size * 0.25), int(y_size * 0.65)),  # Anterior-posterior  
        'z_range': (int(z_size * 0.15), int(z_size * 0.45))   # Inferior-superior
    }
    
    print(f"   Left ROI: x={left_roi['x_range']}, y={left_roi['y_range']}, z={left_roi['z_range']}")
    print(f"   Right ROI: x={right_roi['x_range']}, y={right_roi['y_range']}, z={right_roi['z_range']}")
    
    return left_roi, right_roi

def segment_hippocampus_region(data, brain_mask, roi, side_name):
    """Segment hippocampus in specified ROI using intensity and shape constraints"""
    print(f"✂️  Segmenting {side_name} hippocampus...")
    
    # Extract ROI
    x_range, y_range, z_range = roi['x_range'], roi['y_range'], roi['z_range']
    
    roi_data = data[x_range[0]:x_range[1], 
                   y_range[0]:y_range[1], 
                   z_range[0]:z_range[1]]
    
    roi_mask = brain_mask[x_range[0]:x_range[1], 
                         y_range[0]:y_range[1], 
                         z_range[0]:z_range[1]]
    
    # Hippocampus typically has gray matter intensity
    # Use intensity-based segmentation within brain mask
    if np.sum(roi_mask) == 0:
        print(f"   ⚠️  No brain tissue in {side_name} ROI")
        return np.zeros_like(roi_data)
    
    # Calculate intensity statistics within ROI
    roi_intensities = roi_data[roi_mask]
    
    if len(roi_intensities) == 0:
        return np.zeros_like(roi_data)
    
    # Target gray matter intensities (typically 40th-80th percentile)
    lower_thresh = np.percentile(roi_intensities, 30)
    upper_thresh = np.percentile(roi_intensities, 85)
    
    # Create initial segmentation
    hippo_mask = (roi_data >= lower_thresh) & (roi_data <= upper_thresh) & roi_mask
    
    # Morphological operations to clean up
    hippo_mask = morphology.remove_small_objects(hippo_mask, min_size=50)
    hippo_mask = morphology.binary_closing(hippo_mask, morphology.ball(1))
    hippo_mask = ndimage.binary_fill_holes(hippo_mask)
    
    # Keep only the largest connected component (main hippocampus)
    if np.sum(hippo_mask) > 0:
        labels = measure.label(hippo_mask)
        props = measure.regionprops(labels)
        
        if props:
            # Keep largest component
            largest_area = max(props, key=lambda x: x.area)
            hippo_mask = (labels == largest_area.label)
    
    volume_voxels = np.sum(hippo_mask)
    print(f"   {side_name} hippocampus: {volume_voxels} voxels")
    
    return hippo_mask

def create_full_segmentation(data, left_roi, right_roi, left_mask, right_mask):
    """Combine left and right hippocampus segmentations"""
    print("🔗 Combining hippocampus segmentations...")
    
    # Initialize full segmentation
    full_seg = np.zeros_like(data)
    
    # Place left hippocampus (label = 1)
    x_range, y_range, z_range = left_roi['x_range'], left_roi['y_range'], left_roi['z_range']
    full_seg[x_range[0]:x_range[1], 
            y_range[0]:y_range[1], 
            z_range[0]:z_range[1]][left_mask] = 1
    
    # Place right hippocampus (label = 2)  
    x_range, y_range, z_range = right_roi['x_range'], right_roi['y_range'], right_roi['z_range']
    full_seg[x_range[0]:x_range[1], 
            y_range[0]:y_range[1], 
            z_range[0]:z_range[1]][right_mask] = 2
    
    return full_seg

def calculate_volumes(segmentation, voxel_size):
    """Calculate hippocampus volumes"""
    print("📊 Calculating volumes...")
    
    voxel_volume_mm3 = np.prod(voxel_size)
    
    left_voxels = np.sum(segmentation == 1)
    right_voxels = np.sum(segmentation == 2) 
    total_voxels = left_voxels + right_voxels
    
    volumes = {
        'left_hippocampus_mm3': left_voxels * voxel_volume_mm3,
        'right_hippocampus_mm3': right_voxels * voxel_volume_mm3,
        'total_hippocampus_mm3': total_voxels * voxel_volume_mm3,
        'left_voxels': left_voxels,
        'right_voxels': right_voxels,
        'total_voxels': total_voxels,
        'asymmetry_index': (left_voxels - right_voxels) / (left_voxels + right_voxels) if total_voxels > 0 else 0
    }
    
    print(f"   Left hippocampus:  {volumes['left_hippocampus_mm3']:.1f} mm³ ({left_voxels} voxels)")
    print(f"   Right hippocampus: {volumes['right_hippocampus_mm3']:.1f} mm³ ({right_voxels} voxels)")
    print(f"   Total hippocampus: {volumes['total_hippocampus_mm3']:.1f} mm³ ({total_voxels} voxels)")
    print(f"   Asymmetry index:   {volumes['asymmetry_index']:.3f}")
    
    return volumes

def create_qc_images(data, segmentation, output_dir, subject_id):
    """Create quality control images"""
    print("📸 Creating QC images...")
    
    # Create slices for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get middle slices
    x_mid, y_mid, z_mid = [s//2 for s in data.shape]
    
    slices = [
        ('Sagittal', data[x_mid, :, :], segmentation[x_mid, :, :]),
        ('Coronal', data[:, y_mid, :], segmentation[:, y_mid, :]),
        ('Axial', data[:, :, z_mid], segmentation[:, :, z_mid])
    ]
    
    for i, (plane, img_slice, seg_slice) in enumerate(slices):
        # Original image
        axes[0, i].imshow(img_slice.T, cmap='gray', origin='lower')
        axes[0, i].set_title(f'{plane} - Original')
        axes[0, i].axis('off')
        
        # Overlay segmentation
        axes[1, i].imshow(img_slice.T, cmap='gray', origin='lower', alpha=0.7)
        
        # Color overlay for hippocampus
        overlay = np.zeros((*img_slice.T.shape, 3))
        overlay[seg_slice.T == 1] = [1, 0, 0]  # Red for left
        overlay[seg_slice.T == 2] = [0, 0, 1]  # Blue for right
        
        axes[1, i].imshow(overlay, origin='lower', alpha=0.5)
        axes[1, i].set_title(f'{plane} - Segmentation')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Hippocampus Segmentation QC: {subject_id}')
    plt.tight_layout()
    
    qc_file = os.path.join(output_dir, f'{subject_id}_qc.png')
    plt.savefig(qc_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   QC image saved: {qc_file}")

def main():
    parser = argparse.ArgumentParser(description='Simple Hippocampus Segmentation')
    parser.add_argument('--input', required=True, help='Input NIfTI file')
    parser.add_argument('--output', default='simple_seg_output', help='Output directory')
    
    args = parser.parse_args()
    
    print("🧠 Simple Hippocampus Segmentation")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'segmentations'), exist_ok=True)
    
    # Get subject ID
    subject_id = os.path.basename(args.input).split('.')[0]
    
    # Load image
    img, data = load_brain_image(args.input)
    if img is None:
        return
    
    # Get voxel size
    voxel_size = img.header.get_zooms()[:3]
    
    # Brain extraction
    brain_mask = brain_extraction(data)
    
    # Define hippocampus ROIs
    left_roi, right_roi = create_hippocampus_roi(data, brain_mask, img.affine)
    
    # Segment both hippocampi
    left_hippo = segment_hippocampus_region(data, brain_mask, left_roi, "Left")
    right_hippo = segment_hippocampus_region(data, brain_mask, right_roi, "Right")
    
    # Create full segmentation
    full_segmentation = create_full_segmentation(data, left_roi, right_roi, left_hippo, right_hippo)
    
    # Calculate volumes
    volumes = calculate_volumes(full_segmentation, voxel_size)
    volumes['subject_id'] = subject_id
    volumes['method'] = 'simple_template'
    
    # Save segmentation
    seg_file = os.path.join(args.output, 'segmentations', f'{subject_id}_hippocampus.nii.gz')
    seg_img = nib.Nifti1Image(full_segmentation, img.affine, img.header)
    nib.save(seg_img, seg_file)
    print(f"💾 Segmentation saved: {seg_file}")
    
    # Save volumes
    volumes_file = os.path.join(args.output, f'{subject_id}_volumes.csv')
    pd.DataFrame([volumes]).to_csv(volumes_file, index=False)
    print(f"📊 Volumes saved: {volumes_file}")
    
    # Create QC images
    create_qc_images(data, full_segmentation, args.output, subject_id)
    
    print(f"\n✅ Segmentation completed for {subject_id}")
    print(f"   Total hippocampus volume: {volumes['total_hippocampus_mm3']:.1f} mm³")

if __name__ == "__main__":
    main()