#!/usr/bin/env python3
"""
3D HCCT Dataset Loader for ADNI Data

Adapted from: https://github.com/arindammajee/Alzheimer-Detection-with-3D-HCCT
Modified to work with existing ADNI preprocessed data (skull-stripped, registered MRI scans)
"""

import torch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADNIDataset(Dataset):
    """
    PyTorch Dataset for ADNI MRI scans

    Expects CSV files with columns:
    - scan_path: path to .nii.gz file
    - label: 0 for CN, 1 for MCI, 2 for AD
    """

    def __init__(
        self,
        csv_path: str,
        target_shape: Tuple[int, int, int] = (192, 192, 192),
        transform: Optional[callable] = None,
        augment: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV file with columns 'scan_path' and 'label'
            target_shape: Target shape for all volumes (D, H, W)
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.transform = transform
        self.augment = augment

        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"  CN: {len(self.df[self.df['label'] == 0])}")
        logger.info(f"  MCI: {len(self.df[self.df['label'] == 1])}")
        logger.info(f"  AD: {len(self.df[self.df['label'] == 2])}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape (1, D, H, W)
            label: int (0 for CN, 1 for MCI, 2 for AD)
        """
        # Get scan path and label
        scan_path = self.df.iloc[idx]['scan_path']
        label = int(self.df.iloc[idx]['label'])

        # Load NIfTI file
        nifti_img = nib.load(scan_path)
        image = nifti_img.get_fdata().astype(np.float32)

        # Normalize intensity to [0, 1]
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())

        # Resize to target shape
        image = self._resize_volume(image, self.target_shape)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = image[np.newaxis, ...]

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        # Apply augmentation if enabled
        if self.augment and self.transform:
            image = self.transform(image)

        return image, label

    def _resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize volume to target shape using scipy zoom

        Args:
            volume: numpy array of shape (D, H, W)
            target_shape: target shape (D, H, W)

        Returns:
            resized volume
        """
        current_shape = volume.shape

        # Calculate zoom factors for each dimension
        zoom_factors = [
            target_shape[i] / current_shape[i]
            for i in range(3)
        ]

        # Resize using scipy zoom (order=1 is bilinear interpolation)
        resized_volume = zoom(volume, zoom_factors, order=1)

        return resized_volume


class ADNIDatasetWithPadding(ADNIDataset):
    """
    Alternative dataset that pads instead of resizing
    This preserves aspect ratio but may use more memory
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor of shape (1, D, H, W)
            label: int (0 for CN, 1 for MCI, 2 for AD)
        """
        # Get scan path and label
        scan_path = self.df.iloc[idx]['scan_path']
        label = int(self.df.iloc[idx]['label'])

        # Load NIfTI file
        nifti_img = nib.load(scan_path)
        image = nifti_img.get_fdata().astype(np.float32)

        # Normalize intensity to [0, 1]
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())

        # Pad to target shape (center padding)
        image = self._pad_to_shape(image, self.target_shape)

        # Then resize if needed
        if image.shape != self.target_shape:
            image = self._resize_volume(image, self.target_shape)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = image[np.newaxis, ...]

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        return image, label

    def _pad_to_shape(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Pad volume to target shape (center padding with zeros)
        """
        current_shape = volume.shape

        padding = [
            ((target_shape[i] - current_shape[i]) // 2,
             (target_shape[i] - current_shape[i]) // 2 + (target_shape[i] - current_shape[i]) % 2)
            for i in range(3)
        ]

        padded_volume = np.pad(volume, padding, mode='constant', constant_values=0)

        return padded_volume


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 8,
    target_shape: Tuple[int, int, int] = (192, 192, 192),
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        batch_size: Batch size
        target_shape: Target volume shape
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ADNIDataset(train_csv, target_shape=target_shape, augment=False)
    val_dataset = ADNIDataset(val_csv, target_shape=target_shape, augment=False)
    test_dataset = ADNIDataset(test_csv, target_shape=target_shape, augment=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"\nDataLoader statistics:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    print(f"Testing dataset loading from {csv_path}")
    dataset = ADNIDataset(csv_path, target_shape=(192, 192, 192))

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    image, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label} ({'CN' if label == 0 else 'AD'})")
