import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Configuration
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"
SYNTHSTRIP_DIR = os.path.join(BASE_DIR, "ms-1-synthstrip-skull")  # SynthStrip results
HDBET_DIR = os.path.join(BASE_DIR, "ms-1-skull")  # HD-BET results
OUTPUT_DIR = os.path.join(BASE_DIR, "synthstrip-hdbet-comparison-plots")

def extract_subject_info(filename):
    """Extract subject ID and timepoint from filename"""
    # Pattern for files like: SEP-MRI-001_T0_6f1_synthstrip_skull.nii.gz or SEP-MRI-001_T0_6f1_skull.nii.gz
    pattern = r'(SEP-MRI-\d+)_(T\d+)_([a-f0-9]+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

def find_matching_files():
    """Find matching files between SynthStrip and HD-BET directories"""
    synthstrip_files = [f for f in os.listdir(SYNTHSTRIP_DIR) if f.endswith('_synthstrip_skull.nii.gz')]
    hdbet_files = [f for f in os.listdir(HDBET_DIR) if f.endswith('_registered_n4.nii.gz')]
    
    print(f"Found {len(synthstrip_files)} SynthStrip files")
    print(f"Found {len(hdbet_files)} HD-BET files")
    
    # Create mapping of subject_timepoint_hash to files
    synthstrip_map = {}
    hdbet_map = {}
    
    for file in synthstrip_files:
        subject, timepoint, hash_code = extract_subject_info(file)
        if subject and timepoint and hash_code:
            key = f"{subject}_{timepoint}_{hash_code}"
            synthstrip_map[key] = file
            print(f"SynthStrip: {key} -> {file}")
    
    for file in hdbet_files:
        subject, timepoint, hash_code = extract_subject_info(file)
        if subject and timepoint and hash_code:
            key = f"{subject}_{timepoint}_{hash_code}"
            hdbet_map[key] = file
            print(f"HD-BET: {key} -> {file}")
    
    # Find common subjects
    common_keys = set(synthstrip_map.keys()) & set(hdbet_map.keys())
    print(f"Found {len(common_keys)} matching pairs")
    
    matching_pairs = []
    for key in sorted(common_keys):
        matching_pairs.append({
            'key': key,
            'synthstrip_file': synthstrip_map[key],
            'hdbet_file': hdbet_map[key]
        })
    
    return matching_pairs

def create_comparison_plot(synthstrip_file, hdbet_file, output_path, slice_idx=None):
    """Create side-by-side comparison plot"""
    print(f"Creating comparison plot for {synthstrip_file} vs {hdbet_file}")
    
    # Load images
    synthstrip_img = nib.load(os.path.join(SYNTHSTRIP_DIR, synthstrip_file))
    hdbet_img = nib.load(os.path.join(HDBET_DIR, hdbet_file))
    
    synthstrip_data = synthstrip_img.get_fdata()
    hdbet_data = hdbet_img.get_fdata()
    
    print(f"SynthStrip shape: {synthstrip_data.shape}")
    print(f"HD-BET shape: {hdbet_data.shape}")
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = synthstrip_data.shape[2] // 2
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Skull Stripping Comparison: {synthstrip_file.split("_synthstrip_skull")[0]}', fontsize=16)
    
    # Axial view
    axes[0, 0].imshow(synthstrip_data[:, :, slice_idx], cmap='gray', aspect='equal')
    axes[0, 0].set_title('SynthStrip - Axial')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(hdbet_data[:, :, slice_idx], cmap='gray', aspect='equal')
    axes[1, 0].set_title('HD-BET - Axial')
    axes[1, 0].axis('off')
    
    # Coronal view
    coronal_idx = synthstrip_data.shape[1] // 2
    axes[0, 1].imshow(synthstrip_data[:, coronal_idx, :], cmap='gray', aspect='equal')
    axes[0, 1].set_title('SynthStrip - Coronal')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(hdbet_data[:, coronal_idx, :], cmap='gray', aspect='equal')
    axes[1, 1].set_title('HD-BET - Coronal')
    axes[1, 1].axis('off')
    
    # Sagittal view
    sagittal_idx = synthstrip_data.shape[0] // 2
    axes[0, 2].imshow(synthstrip_data[sagittal_idx, :, :], cmap='gray', aspect='equal')
    axes[0, 2].set_title('SynthStrip - Sagittal')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(hdbet_data[sagittal_idx, :, :], cmap='gray', aspect='equal')
    axes[1, 2].set_title('HD-BET - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")

def create_overlay_comparison(synthstrip_file, hdbet_file, output_path):
    """Create overlay comparison to highlight differences"""
    print(f"Creating overlay comparison for {synthstrip_file} vs {hdbet_file}")
    
    # Load images
    synthstrip_img = nib.load(os.path.join(SYNTHSTRIP_DIR, synthstrip_file))
    hdbet_img = nib.load(os.path.join(HDBET_DIR, hdbet_file))
    
    synthstrip_data = synthstrip_img.get_fdata()
    hdbet_data = hdbet_img.get_fdata()
    
    # Normalize data
    synthstrip_norm = (synthstrip_data - synthstrip_data.min()) / (synthstrip_data.max() - synthstrip_data.min())
    hdbet_norm = (hdbet_data - hdbet_data.min()) / (hdbet_data.max() - hdbet_data.min())
    
    # Create difference map
    diff_map = np.abs(synthstrip_norm - hdbet_norm)
    
    slice_idx = synthstrip_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Detailed Comparison: {synthstrip_file.split("_synthstrip_skull")[0]}', fontsize=16)
    
    # SynthStrip
    axes[0].imshow(synthstrip_data[:, :, slice_idx], cmap='gray')
    axes[0].set_title('SynthStrip')
    axes[0].axis('off')
    
    # HD-BET
    axes[1].imshow(hdbet_data[:, :, slice_idx], cmap='gray')
    axes[1].set_title('HD-BET')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(synthstrip_data[:, :, slice_idx], cmap='gray', alpha=0.7)
    axes[2].imshow(hdbet_data[:, :, slice_idx], cmap='Reds', alpha=0.3)
    axes[2].set_title('Overlay (Gray: SynthStrip, Red: HD-BET)')
    axes[2].axis('off')
    
    # Difference map
    im = axes[3].imshow(diff_map[:, :, slice_idx], cmap='hot')
    axes[3].set_title('Difference Map')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay comparison to: {output_path}")

def main():
    """Main function to create all comparison plots"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check if directories exist
    if not os.path.exists(SYNTHSTRIP_DIR):
        print(f"SynthStrip directory not found: {SYNTHSTRIP_DIR}")
        return
    
    if not os.path.exists(HDBET_DIR):
        print(f"HD-BET directory not found: {HDBET_DIR}")
        return
    
    # Find matching files
    matching_pairs = find_matching_files()
    
    if not matching_pairs:
        print("No matching files found between SynthStrip and HD-BET directories!")
        return
    
    print(f"Found {len(matching_pairs)} matching pairs to compare")
    
    # Create comparison plots for each pair
    for i, pair in enumerate(matching_pairs):
        print(f"\nProcessing {i+1}/{len(matching_pairs)}: {pair['key']}")
        
        # Create basic comparison plot
        basic_output = os.path.join(OUTPUT_DIR, f"{pair['key']}_comparison.png")
        create_comparison_plot(pair['synthstrip_file'], pair['hdbet_file'], basic_output)
        
        # Create overlay comparison
        overlay_output = os.path.join(OUTPUT_DIR, f"{pair['key']}_overlay.png")
        create_overlay_comparison(pair['synthstrip_file'], pair['hdbet_file'], overlay_output)
    
    print(f"\nComparison plots saved to: {OUTPUT_DIR}")
    print(f"Created {len(matching_pairs)} basic comparison plots")
    print(f"Created {len(matching_pairs)} overlay comparison plots")

if __name__ == "__main__":
    main() 