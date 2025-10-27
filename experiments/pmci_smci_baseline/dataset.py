"""
PyTorch Dataset for MCI Classification (pMCI vs sMCI)

Loads single baseline MRI scans (.nii.gz) for binary classification.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MCISingleMRIDataset(Dataset):
    """
    Dataset for loading single MRI scans for MCI classification

    Each sample consists of:
    - MRI scan (.nii.gz file)
    - Binary label (0=sMCI, 1=pMCI)
    - Patient metadata
    """

    def __init__(
        self,
        csv_path: str,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV file (train.csv, val.csv, or test.csv)
            target_shape: Target shape for resizing MRI volumes
            normalize: Whether to normalize intensity values
            augment: Whether to apply data augmentation (for training only)
        """
        self.csv_path = Path(csv_path)
        self.target_shape = target_shape
        self.normalize = normalize
        self.augment = augment

        # Load CSV
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"  pMCI: {(self.df['label'] == 1).sum()}")
        logger.info(f"  sMCI: {(self.df['label'] == 0).sum()}")

        # Verify all scan files exist
        self._verify_files()

    def _verify_files(self):
        """Verify that all scan files exist"""
        missing = []
        for idx, row in self.df.iterrows():
            scan_path = Path(row['scan_path'])
            if not scan_path.exists():
                missing.append(str(scan_path))

        if missing:
            logger.warning(f"Found {len(missing)} missing scan files")
            logger.warning(f"Sample missing files: {missing[:5]}")
            # Filter out missing files
            self.df = self.df[self.df['scan_path'].apply(lambda x: Path(x).exists())]
            logger.info(f"Filtered to {len(self.df)} samples with existing files")

    def __len__(self) -> int:
        return len(self.df)

    def _load_nifti(self, nifti_path: str) -> np.ndarray:
        """Load NIfTI file and return numpy array"""
        try:
            nii = nib.load(nifti_path)
            data = nii.get_fdata()
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {nifti_path}: {e}")
            raise

    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to target shape using trilinear interpolation"""
        import torch.nn.functional as F

        # Convert to torch tensor
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

        # Resize using trilinear interpolation
        resized = F.interpolate(
            volume_tensor,
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        )

        # Convert back to numpy
        return resized.squeeze().numpy()

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to zero mean and unit variance"""
        # Clip outliers (optional, helps with stability)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)

        # Z-score normalization
        mean = volume.mean()
        std = volume.std()

        if std > 0:
            volume = (volume - mean) / std
        else:
            volume = volume - mean

        return volume

    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply data augmentation (for training only)"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()

        # Random rotation (small angle)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            # Simple rotation around one axis (can be extended)
            pass  # Implement rotation if needed

        # Random intensity scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            volume = volume * scale

        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = volume + noise

        return volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        row = self.df.iloc[idx]

        # Load MRI scan
        scan_path = row['scan_path']
        volume = self._load_nifti(scan_path)

        # Resize to target shape
        if volume.shape != self.target_shape:
            volume = self._resize_volume(volume)

        # Normalize
        if self.normalize:
            volume = self._normalize_volume(volume)

        # Augment (training only)
        if self.augment:
            volume = self._augment_volume(volume)

        # Convert to tensor and add channel dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # [1, D, H, W]

        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)

        # Return as dictionary (compatible with existing training code)
        return {
            'image': volume_tensor,
            'label': label,
            'PTID': row['PTID'],
            'scan_path': scan_path
        }


def get_mci_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 8,
    num_workers: int = 4,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
    use_weighted_sampling: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for MCI classification

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_shape: Target shape for MRI volumes
        use_weighted_sampling: Whether to use weighted sampling for training (balances classes)

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    # Create datasets
    train_dataset = MCISingleMRIDataset(
        csv_path=train_csv,
        target_shape=target_shape,
        normalize=True,
        augment=True  # Augmentation for training
    )

    val_dataset = MCISingleMRIDataset(
        csv_path=val_csv,
        target_shape=target_shape,
        normalize=True,
        augment=False
    )

    test_dataset = MCISingleMRIDataset(
        csv_path=test_csv,
        target_shape=target_shape,
        normalize=True,
        augment=False
    )

    # Create weighted sampler for training (optional, helps with class imbalance)
    if use_weighted_sampling:
        train_df = pd.read_csv(train_csv)
        labels = train_df['label'].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # Val and test loaders (no shuffling, no sampling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
