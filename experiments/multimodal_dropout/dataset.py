#!/usr/bin/env python3
"""
Multi-Modal Dataset with Modality Dropout

Handles:
- Samples with both MRI and tabular data
- Samples with only MRI (tabular filled with zeros)
- Random modality dropout during training for robustness

The modality dropout makes the model robust to missing data at inference time.
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MRI Preprocessing Functions (copied from mri_vit_ad for independence)
# ============================================================================

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

# ============================================================================


class ModalityDropoutDataset(Dataset):
    """
    Dataset with modality dropout for robust multi-modal learning.

    Features:
    - Handles samples with or without tabular data
    - Random modality dropout during training
    - Returns modality availability mask for the model

    CSV must have:
    - scan_path: path to .nii.gz file
    - label: integer class label
    - has_tabular: boolean indicating if tabular data is available
    - Tabular feature columns (can be NaN if has_tabular=False)
    """

    def __init__(
        self,
        csv_path: str,
        tabular_features: List[str],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        is_training: bool = True,
        mri_dropout_rate: float = 0.0,
        tabular_dropout_rate: float = 0.0,
        scaler: Optional[StandardScaler] = None,
        use_paper_preprocessing: bool = True,
        target_spacing: float = 1.75,
    ):
        """
        Args:
            csv_path: Path to CSV with MRI paths and tabular features
            tabular_features: List of column names for tabular features
            target_shape: Target MRI volume shape
            augment: Whether to apply MRI augmentation
            is_training: If True, apply modality dropout
            mri_dropout_rate: Probability of dropping MRI (zeroing out)
            tabular_dropout_rate: Probability of dropping tabular features
            scaler: Pre-fitted scaler (for val/test sets)
            use_paper_preprocessing: Use 1.75mm resampling + RAS
            target_spacing: Target voxel spacing
        """
        self.df = pd.read_csv(csv_path)
        self.tabular_features = tabular_features
        self.num_tabular_features = len(tabular_features)
        self.target_shape = target_shape
        self.augment = augment
        self.is_training = is_training
        self.mri_dropout_rate = mri_dropout_rate
        self.tabular_dropout_rate = tabular_dropout_rate
        self.use_paper_preprocessing = use_paper_preprocessing
        self.target_spacing = target_spacing

        # Check for has_tabular column, default to True if not present
        if 'has_tabular' not in self.df.columns:
            self.df['has_tabular'] = True

        # Fill missing tabular values with 0 (will be masked anyway)
        for col in tabular_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
            else:
                self.df[col] = 0

        # Fit or use provided scaler (only on samples with tabular data)
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            tabular_mask = self.df['has_tabular'] == True
            if tabular_mask.sum() > 0:
                tabular_data = self.df.loc[tabular_mask, tabular_features].values.astype(np.float32)
                self.scaler.fit(tabular_data)
            else:
                # No tabular data, fit on zeros
                self.scaler.fit(np.zeros((1, len(tabular_features))))

        self._log_stats()

    def _log_stats(self):
        """Log dataset statistics"""
        total = len(self.df)
        with_tabular = (self.df['has_tabular'] == True).sum()
        mri_only = total - with_tabular

        logger.info(f"Loaded {total} samples")
        logger.info(f"  With tabular: {with_tabular} ({100*with_tabular/total:.1f}%)")
        logger.info(f"  MRI only: {mri_only} ({100*mri_only/total:.1f}%)")

        if self.is_training:
            logger.info(f"  MRI dropout rate: {self.mri_dropout_rate}")
            logger.info(f"  Tabular dropout rate: {self.tabular_dropout_rate}")

        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                logger.info(f"  Class {label}: {count} ({100*count/total:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            mri: torch.Tensor (1, D, H, W) - MRI volume (may be zeros if dropped)
            tabular: torch.Tensor (num_features,) - Tabular features (may be zeros)
            modality_mask: torch.Tensor (2,) - [mri_available, tabular_available]
            label: int
        """
        row = self.df.iloc[idx]
        label = int(row['label'])
        has_tabular = bool(row['has_tabular'])

        # Load MRI
        mri = self._load_mri(row['scan_path'])

        # Load tabular features
        tabular = self._load_tabular(row, has_tabular)

        # Initialize modality mask (1 = available, 0 = dropped/missing)
        mri_available = 1.0
        tabular_available = 1.0 if has_tabular else 0.0

        # Apply modality dropout during training
        if self.is_training:
            # Dropout MRI
            if random.random() < self.mri_dropout_rate:
                mri = torch.zeros_like(mri)
                mri_available = 0.0

            # Dropout tabular (only if originally available)
            if has_tabular and random.random() < self.tabular_dropout_rate:
                tabular = torch.zeros_like(tabular)
                tabular_available = 0.0

        modality_mask = torch.tensor([mri_available, tabular_available], dtype=torch.float32)

        return mri, tabular, modality_mask, label

    def _load_mri(self, scan_path: str) -> torch.Tensor:
        """Load and preprocess MRI volume"""
        try:
            nifti = nib.load(scan_path)
            image = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

            if self.use_paper_preprocessing:
                image, affine = convert_to_ras(image, affine)
                current_spacing = np.abs(np.diag(affine)[:3])
                image = resample_to_spacing(image, current_spacing, self.target_spacing)

        except Exception as e:
            logger.error(f"Error loading {scan_path}: {e}")
            image = np.zeros(self.target_shape, dtype=np.float32)

        # Normalize intensity
        image = self._normalize_mri(image)

        # Resize/crop to target shape
        if image.shape != self.target_shape:
            image = self._resize_or_crop(image)

        # Add channel dimension
        image = image[np.newaxis, ...]
        image = torch.from_numpy(image).float()

        return image

    def _normalize_mri(self, image: np.ndarray) -> np.ndarray:
        """Percentile normalization for MRI"""
        brain_voxels = image[image > 0]
        if len(brain_voxels) > 0:
            p1, p99 = np.percentile(brain_voxels, (1, 99))
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)
        return image

    def _resize_or_crop(self, image: np.ndarray) -> np.ndarray:
        """Resize/crop to target shape"""
        target = self.target_shape
        current = list(image.shape)
        result = image.copy()

        for dim in range(3):
            if current[dim] > target[dim]:
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
                diff = target[dim] - current[dim]
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[dim] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='constant', constant_values=0)
            current = list(result.shape)

        if result.shape != target:
            zoom_factors = [t / s for t, s in zip(target, result.shape)]
            result = zoom(result, zoom_factors, order=1)

        return result

    def _load_tabular(self, row: pd.Series, has_tabular: bool) -> torch.Tensor:
        """Load tabular features, return zeros if not available"""
        if not has_tabular:
            return torch.zeros(self.num_tabular_features, dtype=torch.float32)

        features = []
        for col in self.tabular_features:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            features.append(float(val))

        features = np.array(features, dtype=np.float32)

        # Normalize
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        return torch.from_numpy(features).float()

    def get_scaler(self) -> StandardScaler:
        """Return the fitted scaler for use with val/test sets"""
        return self.scaler


def get_dropout_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    tabular_features: List[str],
    batch_size: int = 4,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 4,
    pin_memory: bool = True,
    mri_dropout_rate: float = 0.2,
    tabular_dropout_rate: float = 0.2,
    use_paper_preprocessing: bool = True,
    target_spacing: float = 1.75,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create dataloaders with modality dropout for training.

    Args:
        train_csv, val_csv, test_csv: Paths to data CSVs
        tabular_features: List of tabular feature column names
        batch_size: Batch size
        mri_dropout_rate: Probability of dropping MRI during training
        tabular_dropout_rate: Probability of dropping tabular during training

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    logger.info("Creating dataloaders with modality dropout")
    logger.info(f"  MRI dropout: {mri_dropout_rate}")
    logger.info(f"  Tabular dropout: {tabular_dropout_rate}")

    # Train dataset (fits scaler, applies dropout)
    train_dataset = ModalityDropoutDataset(
        train_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=True,
        is_training=True,
        mri_dropout_rate=mri_dropout_rate,
        tabular_dropout_rate=tabular_dropout_rate,
        scaler=None,
        use_paper_preprocessing=use_paper_preprocessing,
        target_spacing=target_spacing
    )

    scaler = train_dataset.get_scaler()

    # Val/test datasets (no dropout, use train scaler)
    val_dataset = ModalityDropoutDataset(
        val_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        is_training=False,
        mri_dropout_rate=0.0,
        tabular_dropout_rate=0.0,
        scaler=scaler,
        use_paper_preprocessing=use_paper_preprocessing,
        target_spacing=target_spacing
    )

    test_dataset = ModalityDropoutDataset(
        test_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        is_training=False,
        mri_dropout_rate=0.0,
        tabular_dropout_rate=0.0,
        scaler=scaler,
        use_paper_preprocessing=use_paper_preprocessing,
        target_spacing=target_spacing
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
    logger.info(f"Val: {len(val_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # Test dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    features = ['AGE', 'MMSCORE', 'CDGLOBAL']

    dataset = ModalityDropoutDataset(
        csv_path,
        tabular_features=features,
        target_shape=(128, 128, 128),
        is_training=True,
        mri_dropout_rate=0.2,
        tabular_dropout_rate=0.2
    )

    print(f"\nDataset size: {len(dataset)}")

    mri, tabular, mask, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  MRI shape: {mri.shape}")
    print(f"  Tabular shape: {tabular.shape}")
    print(f"  Modality mask: {mask}")
    print(f"  Label: {label}")
