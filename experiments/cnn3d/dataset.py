#!/usr/bin/env python3
"""
3D MRI Dataset for CNN3D Alzheimer's Classification

Reuses the preprocessing pipeline from mri_vit_ad:
- 1.75mm isotropic voxel spacing
- RAS orientation
- 128x128x128 output resolution

Expects CSV with columns:
- scan_path: path to .nii.gz file
- label: integer class label (0=CN, 1=AD or per task)
"""

import sys
from pathlib import Path

# Add parent experiments directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mri_vit_ad'))

from dataset import MRIDataset, get_dataloaders  # noqa: E402, F401

# Re-export for direct import from this module
__all__ = ['MRIDataset', 'get_dataloaders']


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    dataset = MRIDataset(csv_path, target_shape=(128, 128, 128))

    print(f"\nDataset size: {len(dataset)}")

    import torch
    image, label = dataset[0]
    print(f"First sample:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label}")
