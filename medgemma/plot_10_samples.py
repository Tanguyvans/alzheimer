#!/usr/bin/env python3
"""
Plot 2 coronal + 2 axial slices from 10 different samples.
"""

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'medgemma')
from utils.slice_extractor import MultiViewSliceExtractor
import pandas as pd

def main():
    # Get 10 samples (mix of CN and AD)
    df = pd.read_csv('experiments/multimodal_fusion/data/combined_trajectory/test.csv')

    # Get 5 CN and 5 AD
    cn_samples = df[df['label'] == 0].head(5)
    ad_samples = df[df['label'] == 1].head(5)
    samples = pd.concat([cn_samples, ad_samples])

    extractor = MultiViewSliceExtractor(
        n_coronal=2,
        n_axial=2,
        coronal_region=(0.45, 0.55),
        axial_region=(0.35, 0.45),  # Adjusted - axial 2 was good at ~0.40
        output_size=448
    )

    # Create grid: 10 rows x 4 columns (2 coronal + 2 axial)
    fig, axes = plt.subplots(10, 4, figsize=(16, 40))
    fig.suptitle('2 Coronal + 2 Axial Slices from 10 Samples (5 CN, 5 AD)', fontsize=16, y=1.01)

    # Column headers
    col_titles = ['Coronal 1', 'Coronal 2', 'Axial 1', 'Axial 2']

    for row_idx, (_, row) in enumerate(samples.iterrows()):
        label = "CN" if row['label'] == 0 else "AD"

        try:
            coronal, axial = extractor.extract_from_nifti(row['scan_path'])
            all_slices = coronal + axial

            for col_idx, img in enumerate(all_slices):
                axes[row_idx, col_idx].imshow(img)
                if col_idx == 0:
                    axes[row_idx, col_idx].set_ylabel(f"Sample {row_idx+1}\n{label}", fontsize=10)
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(col_titles[col_idx])
                axes[row_idx, col_idx].axis('off')
        except Exception as e:
            print(f"Error with sample {row_idx+1}: {e}")
            for col_idx in range(4):
                axes[row_idx, col_idx].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig('medgemma/10_samples_multiview.png', dpi=100, bbox_inches='tight')
    print(f"✓ Saved to: medgemma/10_samples_multiview.png")


if __name__ == "__main__":
    main()
