#!/usr/bin/env python3
"""
Visualize extracted 2D hippocampus-centered chips
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_chips(chips_folder, output_path="chip_visualization.png"):
    """
    Visualize sample 2D chips from hippocampus extraction
    
    Parameters:
    chips_folder: Folder containing chips and metadata
    output_path: Path to save visualization
    """
    print("📊 Creating chip visualization...")
    
    # Load chips metadata
    metadata_path = os.path.join(chips_folder, 'hippocampus_chips.csv')
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    
    # Get sample chips (first 4 axial, first 4 coronal)
    axial_chips = df[df['view'] == 'axial'].head(4)
    coronal_chips = df[df['view'] == 'coronal'].head(4)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Hippocampus-Centered 2D Chips (224x224x3)', fontsize=16)
    
    # Plot axial chips
    for i, (_, row) in enumerate(axial_chips.iterrows()):
        if i < 4:
            chip = np.load(row['chip_path'])
            # Display middle channel (i) of the 3-channel chip
            axes[0, i].imshow(chip[:, :, 1], cmap='gray', origin='lower')
            axes[0, i].set_title(f"Axial Z={row['slice_index']}")
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
    
    # Plot coronal chips  
    for i, (_, row) in enumerate(coronal_chips.iterrows()):
        if i < 4:
            chip = np.load(row['chip_path'])
            # Display middle channel (i) of the 3-channel chip
            axes[1, i].imshow(chip[:, :, 1], cmap='gray', origin='lower')
            axes[1, i].set_title(f"Coronal Y={row['slice_index']}")
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel('Axial View', fontsize=12, rotation=90, va='center')
    axes[1, 0].set_ylabel('Coronal View', fontsize=12, rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Chip visualization saved: {output_path}")
    
    # Print summary statistics
    print(f"\n📈 Chip Extraction Summary:")
    print(f"   Total chips: {len(df)}")
    print(f"   Axial chips: {len(df[df['view'] == 'axial'])}")
    print(f"   Coronal chips: {len(df[df['view'] == 'coronal'])}")
    print(f"   Chip size: {df['shape'].iloc[0]}")

def visualize_3channel_chip(chip_path, output_path="3channel_visualization.png"):
    """
    Visualize the 3-channel structure of a single chip
    """
    print("🎨 Creating 3-channel chip visualization...")
    
    chip = np.load(chip_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'3-Channel Chip Structure\n{os.path.basename(chip_path)}', fontsize=14)
    
    axes[0].imshow(chip[:, :, 0], cmap='gray', origin='lower')
    axes[0].set_title('Channel 0 (i-1)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].imshow(chip[:, :, 1], cmap='gray', origin='lower')
    axes[1].set_title('Channel 1 (i)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    axes[2].imshow(chip[:, :, 2], cmap='gray', origin='lower')
    axes[2].set_title('Channel 2 (i+1)')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 3-channel visualization saved: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize extracted hippocampus chips')
    parser.add_argument('--chips-folder', required=True, help='Folder containing chips and metadata')
    parser.add_argument('--output', default='chip_visualization.png', help='Output visualization path')
    parser.add_argument('--sample-chip', help='Path to specific chip for 3-channel visualization')
    
    args = parser.parse_args()
    
    # Create main chip overview
    visualize_chips(args.chips_folder, args.output)
    
    # Create 3-channel visualization if requested
    if args.sample_chip:
        output_3ch = args.output.replace('.png', '_3channel.png')
        visualize_3channel_chip(args.sample_chip, output_3ch)