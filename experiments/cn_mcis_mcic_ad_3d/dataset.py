#!/usr/bin/env python3
"""
4-Class MRI Dataset Loader

Classes:
- 0: CN (Cognitively Normal)
- 1: MCI_stable
- 2: MCI_to_AD (converter)
- 3: AD (Alzheimer's Disease)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ['CN', 'MCI_stable', 'MCI_to_AD', 'AD']


class MRI4ClassDataset(Dataset):
    """
    PyTorch Dataset for 4-class MRI classification

    Expects CSV with columns:
    - scan_path: path to .nii.gz file
    - label: 0=CN, 1=MCI_stable, 2=MCI_to_AD, 3=AD
    """

    def __init__(
        self,
        csv_path: str,
        target_shape: Tuple[int, int, int] = (192, 192, 192),
        augment: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.augment = augment

        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        for i, name in enumerate(CLASS_NAMES):
            count = len(self.df[self.df['label'] == i])
            logger.info(f"  {name}: {count}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        scan_path = self.df.iloc[idx]['scan_path']
        label = int(self.df.iloc[idx]['label'])

        # Load NIfTI
        nifti_img = nib.load(scan_path)
        image = nifti_img.get_fdata().astype(np.float32)

        # Percentile-based normalization (robust to outliers)
        brain_voxels = image[image > 0]
        if len(brain_voxels) > 0:
            p1, p99 = np.percentile(brain_voxels, (1, 99))
            image = np.clip(image, p1, p99)

        # Normalize to [0, 1]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # Resize to target shape
        image = self._resize_volume(image, self.target_shape)

        # Data augmentation
        if self.augment:
            image = self._augment(image)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = image[np.newaxis, ...]

        return torch.from_numpy(image).float(), label

    def _resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
        return zoom(volume, zoom_factors, order=1)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        # Random flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()

        # Random intensity shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            image = np.clip(image + shift, 0, 1)

        # Random intensity scale
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            image = np.clip(image * scale, 0, 1)

        return image

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced classes"""
        class_counts = [len(self.df[self.df['label'] == i]) for i in range(4)]
        total = sum(class_counts)
        weights = [total / (4 * c) if c > 0 else 0 for c in class_counts]
        return torch.FloatTensor(weights)


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 4,
    target_shape: Tuple[int, int, int] = (192, 192, 192),
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, val, test dataloaders

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_dataset = MRI4ClassDataset(train_csv, target_shape, augment=True)
    val_dataset = MRI4ClassDataset(val_csv, target_shape, augment=False)
    test_dataset = MRI4ClassDataset(test_csv, target_shape, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    class_weights = train_dataset.get_class_weights()

    logger.info(f"\nDataloaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info(f"  Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <csv_path>")
        sys.exit(1)

    dataset = MRI4ClassDataset(sys.argv[1])
    image, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Shape: {image.shape}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label} ({CLASS_NAMES[label]})")
