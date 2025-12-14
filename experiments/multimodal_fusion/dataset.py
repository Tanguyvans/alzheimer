#!/usr/bin/env python3
"""
Multi-Modal Dataset: MRI + Tabular Clinical Features

Loads paired MRI scans and clinical data for multi-modal fusion.
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
import warnings

# Import MRI preprocessing functions using importlib to avoid circular import
import sys
import importlib.util
mri_vit_dataset_path = Path(__file__).parent.parent / "mri_vit_ad" / "dataset.py"
spec = importlib.util.spec_from_file_location("mri_vit_dataset", mri_vit_dataset_path)
mri_vit_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mri_vit_dataset)
convert_to_ras = mri_vit_dataset.convert_to_ras
resample_to_spacing = mri_vit_dataset.resample_to_spacing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Dataset that loads both MRI and tabular features

    Expects CSV with:
    - scan_path: path to .nii.gz file
    - label: integer class label
    - Tabular feature columns (e.g., AGE, MMSCORE, etc.)
    """

    def __init__(
        self,
        csv_path: str,
        tabular_features: List[str],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        normalize_tabular: bool = True,
        scaler: Optional[StandardScaler] = None,
        use_paper_preprocessing: bool = True,
        target_spacing: float = 1.75,
        handle_missing: str = 'median'
    ):
        """
        Args:
            csv_path: Path to CSV with MRI paths and tabular features
            tabular_features: List of column names for tabular features
            target_shape: Target MRI volume shape
            augment: Whether to apply MRI augmentation
            normalize_tabular: Whether to standardize tabular features
            scaler: Pre-fitted scaler (for val/test sets)
            use_paper_preprocessing: Use 1.75mm resampling + RAS
            target_spacing: Target voxel spacing
            handle_missing: How to handle missing values ('median', 'mean', 'drop')
        """
        self.df = pd.read_csv(csv_path)
        self.tabular_features = tabular_features
        self.target_shape = target_shape
        self.augment = augment
        self.use_paper_preprocessing = use_paper_preprocessing
        self.target_spacing = target_spacing
        self.handle_missing = handle_missing

        # Handle missing tabular values
        self._handle_missing_values()

        # Normalize tabular features
        self.normalize_tabular = normalize_tabular
        if normalize_tabular:
            if scaler is not None:
                self.scaler = scaler
            else:
                self.scaler = StandardScaler()
                tabular_data = self.df[tabular_features].values.astype(np.float32)
                self.scaler.fit(tabular_data)
        else:
            self.scaler = None

        self._log_stats()

    def _handle_missing_values(self):
        """Handle missing values in tabular features"""
        for col in self.tabular_features:
            if col not in self.df.columns:
                logger.warning(f"Feature '{col}' not found in CSV, filling with 0")
                self.df[col] = 0
                continue

            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                if self.handle_missing == 'median':
                    fill_value = self.df[col].median()
                elif self.handle_missing == 'mean':
                    fill_value = self.df[col].mean()
                else:
                    fill_value = 0

                self.df[col] = self.df[col].fillna(fill_value)
                logger.info(f"Filled {missing_count} missing values in '{col}' with {fill_value:.2f}")

    def _log_stats(self):
        """Log dataset statistics"""
        logger.info(f"Loaded {len(self.df)} samples")
        logger.info(f"Tabular features ({len(self.tabular_features)}): {self.tabular_features}")

        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                logger.info(f"  Class {label}: {count} ({100*count/len(self.df):.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            mri: torch.Tensor (1, D, H, W)
            tabular: torch.Tensor (num_features,)
            label: int
        """
        row = self.df.iloc[idx]
        label = int(row['label'])

        # Load MRI
        mri = self._load_mri(row['scan_path'])

        # Load tabular features
        tabular = self._load_tabular(row)

        return mri, tabular, label

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

    def _load_tabular(self, row: pd.Series) -> torch.Tensor:
        """Load and preprocess tabular features"""
        features = []
        for col in self.tabular_features:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            features.append(float(val))

        features = np.array(features, dtype=np.float32)

        # Normalize if scaler is set
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        return torch.from_numpy(features).float()

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the fitted scaler for use with val/test sets"""
        return self.scaler


def get_multimodal_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    tabular_features: List[str],
    batch_size: int = 4,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    use_paper_preprocessing: bool = True,
    target_spacing: float = 1.75,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train, val, and test dataloaders for multi-modal data

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    logger.info(f"Creating multi-modal dataloaders")
    logger.info(f"Tabular features: {tabular_features}")

    # Train dataset (fits the scaler)
    train_dataset = MultiModalDataset(
        train_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=augment,
        normalize_tabular=True,
        scaler=None,  # Will fit new scaler
        use_paper_preprocessing=use_paper_preprocessing,
        target_spacing=target_spacing
    )

    # Get fitted scaler
    scaler = train_dataset.get_scaler()

    # Val/test datasets use the same scaler
    val_dataset = MultiModalDataset(
        val_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=use_paper_preprocessing,
        target_spacing=target_spacing
    )

    test_dataset = MultiModalDataset(
        test_csv,
        tabular_features=tabular_features,
        target_shape=target_shape,
        augment=False,
        normalize_tabular=True,
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

    logger.info(f"Train: {len(train_dataset)} samples")
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

    dataset = MultiModalDataset(
        csv_path,
        tabular_features=features,
        target_shape=(128, 128, 128)
    )

    print(f"\nDataset size: {len(dataset)}")

    mri, tabular, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  MRI shape: {mri.shape}")
    print(f"  Tabular shape: {tabular.shape}")
    print(f"  Label: {label}")
