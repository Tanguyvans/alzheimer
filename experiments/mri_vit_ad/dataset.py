#!/usr/bin/env python3
"""
3D MRI Dataset for Alzheimer's Classification

Based on MICCAI 2024 paper preprocessing:
- 1.75mm isotropic voxel spacing
- RAS orientation
- 128x128x128 output resolution

Supports:
- Loading NIfTI files (.nii, .nii.gz)
- Data augmentation with MONAI transforms
- Proper voxel spacing resampling
"""

import torch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, affine_transform
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

# Paper preprocessing constants
TARGET_VOXEL_SPACING = 1.75  # mm (isotropic)
TARGET_OUTPUT_SIZE = 128  # 128x128x128


def resample_to_spacing(image: np.ndarray, current_spacing: np.ndarray, target_spacing: float = 1.75) -> np.ndarray:
    """
    Resample image to target isotropic voxel spacing.

    This matches the MICCAI 2024 paper preprocessing:
    - Resample to 1.75mm isotropic spacing

    Args:
        image: Input 3D array
        current_spacing: Current voxel spacing (D, H, W)
        target_spacing: Target isotropic spacing in mm (default 1.75)

    Returns:
        Resampled image array
    """
    # Calculate zoom factors to get to target spacing
    zoom_factors = current_spacing / target_spacing

    # Apply resampling
    resampled = zoom(image, zoom_factors, order=1, mode='nearest')

    return resampled


def convert_to_ras(image: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image to RAS (Right-Anterior-Superior) orientation.

    Args:
        image: Input 3D array
        affine: 4x4 affine transformation matrix

    Returns:
        Tuple of (reoriented image, new affine)
    """
    # Get current orientation
    current_ornt = nib.orientations.io_orientation(affine)

    # Target RAS orientation
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))

    # Get transformation between current and target
    transform = nib.orientations.ornt_transform(current_ornt, ras_ornt)

    # Apply the transform
    reoriented = nib.orientations.apply_orientation(image, transform)

    # Update affine
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform, image.shape)

    return reoriented, new_affine


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI scans

    Supports preprocessing matching MICCAI 2024 paper:
    - 1.75mm isotropic voxel spacing
    - RAS orientation
    - 128x128x128 output

    Expects CSV with columns:
    - scan_path: path to .nii.gz file
    - label: integer class label (0=CN, 1=MCI, 2=AD)
    """

    def __init__(
        self,
        csv_path: str,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        normalize: str = 'percentile',  # 'percentile', 'minmax', 'zscore'
        use_paper_preprocessing: bool = True,  # Use 1.75mm resampling + RAS
        target_spacing: float = 1.75,  # Target isotropic spacing in mm
    ):
        """
        Args:
            csv_path: Path to CSV file
            target_shape: Target volume shape (D, H, W)
            augment: Whether to apply data augmentation
            normalize: Normalization method
            use_paper_preprocessing: If True, resample to 1.75mm spacing and RAS orientation
            target_spacing: Target voxel spacing (default 1.75mm from paper)
        """
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.augment = augment
        self.normalize = normalize
        self.use_paper_preprocessing = use_paper_preprocessing
        self.target_spacing = target_spacing

        # Setup augmentation transforms
        self.transform = self._build_transforms() if augment and MONAI_TRANSFORMS else None

        # Log dataset stats
        self._log_stats()

        if use_paper_preprocessing:
            logger.info(f"Paper preprocessing enabled: {target_spacing}mm isotropic + RAS orientation")

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
            affine = nifti.affine

            if self.use_paper_preprocessing:
                # Paper preprocessing: RAS orientation + 1.75mm resampling

                # Step 1: Convert to RAS orientation
                image, affine = convert_to_ras(image, affine)

                # Step 2: Get current voxel spacing from affine
                current_spacing = np.abs(np.diag(affine)[:3])

                # Step 3: Resample to target spacing (1.75mm isotropic)
                image = resample_to_spacing(image, current_spacing, self.target_spacing)

        except Exception as e:
            logger.error(f"Error loading {scan_path}: {e}")
            # Return zeros as fallback
            image = np.zeros(self.target_shape, dtype=np.float32)

        # Normalize intensity
        image = self._normalize(image)

        # Resize to target shape (center crop or pad + resize)
        if image.shape != self.target_shape:
            image = self._resize_or_crop(image)

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

    def _resize_or_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Resize/crop image to target shape using center crop for larger dimensions
        and padding for smaller dimensions.

        This matches the paper's approach for handling variable-sized resampled images.
        """
        target = self.target_shape
        current = image.shape

        # For each dimension: crop if larger, pad if smaller
        result = image.copy()

        for dim in range(3):
            if current[dim] > target[dim]:
                # Center crop
                diff = current[dim] - target[dim]
                start = diff // 2
                end = start + target[dim]

                if dim == 0:
                    result = result[start:end, :, :]
                elif dim == 1:
                    result = result[:, start:end, :]
                else:
                    result = result[:, :, start:end]

            elif current[dim] < target[dim]:
                # Pad with zeros (background)
                diff = target[dim] - current[dim]
                pad_before = diff // 2
                pad_after = diff - pad_before

                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[dim] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='constant', constant_values=0)

            current = result.shape

        # Final resize if shapes still don't match exactly (shouldn't happen normally)
        if result.shape != target:
            zoom_factors = [t / s for t, s in zip(target, result.shape)]
            result = zoom(result, zoom_factors, order=1)

        return result

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Legacy resize method - use _resize_or_crop instead"""
        return self._resize_or_crop(image)


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 4,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    use_paper_preprocessing: bool = True,
    target_spacing: float = 1.75,
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
        use_paper_preprocessing: Enable MICCAI 2024 paper preprocessing
        target_spacing: Target voxel spacing in mm (default 1.75)

    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info(f"Creating dataloaders (target_shape={target_shape})")
    if use_paper_preprocessing:
        logger.info(f"Paper preprocessing: {target_spacing}mm isotropic + RAS orientation")

    # Create datasets
    train_dataset = MRIDataset(
        train_csv, target_shape=target_shape, augment=augment,
        use_paper_preprocessing=use_paper_preprocessing, target_spacing=target_spacing
    )
    val_dataset = MRIDataset(
        val_csv, target_shape=target_shape, augment=False,
        use_paper_preprocessing=use_paper_preprocessing, target_spacing=target_spacing
    )
    test_dataset = MRIDataset(
        test_csv, target_shape=target_shape, augment=False,
        use_paper_preprocessing=use_paper_preprocessing, target_spacing=target_spacing
    )

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
