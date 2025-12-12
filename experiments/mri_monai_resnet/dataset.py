#!/usr/bin/env python3
"""
3D MRI Dataset for Alzheimer's Classification

Supports:
- Loading NIfTI files (.nii, .nii.gz)
- Data augmentation with MONAI transforms
- Flexible target shape
"""

import torch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# MONAI transforms
try:
    from monai.transforms import (
        Compose, RandFlip, RandRotate, RandAffine,
        RandGaussianNoise, RandAdjustContrast, ScaleIntensity
    )
    MONAI_TRANSFORMS = True
except ImportError:
    MONAI_TRANSFORMS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI scans

    Expects CSV with columns:
    - scan_path: path to .nii.gz file
    - label: integer class label (0=CN, 1=MCI, 2=AD)
    """

    def __init__(
        self,
        csv_path: str,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        augment: bool = False,
        normalize: str = 'percentile',  # 'percentile', 'minmax', 'zscore'
    ):
        """
        Args:
            csv_path: Path to CSV file
            target_shape: Target volume shape (D, H, W)
            augment: Whether to apply data augmentation
            normalize: Normalization method
        """
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.augment = augment
        self.normalize = normalize

        # Setup augmentation transforms
        self.transform = self._build_transforms() if augment and MONAI_TRANSFORMS else None

        # Log dataset stats
        self._log_stats()

    def _log_stats(self):
        """Log dataset statistics"""
        logger.info(f"Loaded {len(self.df)} samples")

        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts().sort_index()

            # Build class names from 'group' column if available
            if 'group' in self.df.columns:
                # Map label -> group name(s)
                class_names = {}
                for label in label_counts.index:
                    groups = self.df[self.df['label'] == label]['group'].unique()
                    class_names[label] = '+'.join(sorted(groups))
            else:
                class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}

            for label, count in label_counts.items():
                name = class_names.get(label, f'Class {label}')
                logger.info(f"  {name}: {count} ({100*count/len(self.df):.1f}%)")

    def _build_transforms(self):
        """Build MONAI augmentation transforms"""
        if not MONAI_TRANSFORMS:
            return None

        return Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3),
            RandGaussianNoise(prob=0.2, std=0.05),
            RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape (1, D, H, W)
            label: int class label
        """
        row = self.df.iloc[idx]
        scan_path = row['scan_path']
        label = int(row['label'])

        # Load NIfTI
        try:
            nifti = nib.load(scan_path)
            image = nifti.get_fdata().astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {scan_path}: {e}")
            # Return zeros as fallback
            image = np.zeros(self.target_shape, dtype=np.float32)

        # Normalize intensity
        image = self._normalize(image)

        # Resize to target shape
        if image.shape != self.target_shape:
            image = self._resize(image)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = image[np.newaxis, ...]

        # Convert to tensor
        image = torch.from_numpy(image).float()

        # Apply augmentation
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity"""
        if self.normalize == 'percentile':
            # Robust percentile normalization (ignores background)
            brain_voxels = image[image > 0]
            if len(brain_voxels) > 0:
                p1, p99 = np.percentile(brain_voxels, (1, 99))
                image = np.clip(image, p1, p99)
                image = (image - p1) / (p99 - p1 + 1e-8)

        elif self.normalize == 'minmax':
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min())

        elif self.normalize == 'zscore':
            brain_voxels = image[image > 0]
            if len(brain_voxels) > 0:
                mean = brain_voxels.mean()
                std = brain_voxels.std()
                if std > 0:
                    image = (image - mean) / std

        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target shape"""
        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 4,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        batch_size: Batch size
        target_shape: Target volume shape
        num_workers: Number of workers
        pin_memory: Pin memory for faster GPU transfer
        augment: Whether to augment training data

    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info(f"Creating dataloaders (target_shape={target_shape})")

    # Create datasets
    train_dataset = MRIDataset(train_csv, target_shape=target_shape, augment=augment)
    val_dataset = MRIDataset(val_csv, target_shape=target_shape, augment=False)
    test_dataset = MRIDataset(test_csv, target_shape=target_shape, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


class ADNIDataset(MRIDataset):
    """Alias for backwards compatibility"""
    pass


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    dataset = MRIDataset(csv_path, target_shape=(96, 96, 96))

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    image, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label}")
