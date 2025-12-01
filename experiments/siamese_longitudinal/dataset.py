#!/usr/bin/env python3
"""
Dataset for Siamese Network with paired MRI scans.

Supports both NIfTI (.nii, .nii.gz) and NumPy (.npy) files.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairedMRIDataset(Dataset):
    """
    Dataset for paired (baseline, followup) MRI scans.

    Args:
        pairs_csv: Path to CSV with columns [baseline_path, followup_path, is_converter, days_between]
        transform: Optional transforms to apply to both images
        normalize: Whether to normalize images to [0, 1]
        target_shape: Target shape for resizing (D, H, W)
    """

    def __init__(
        self,
        pairs_csv: str,
        transform=None,
        normalize: bool = True,
        target_shape: Optional[Tuple[int, int, int]] = None
    ):
        self.pairs_df = pd.read_csv(pairs_csv)
        self.transform = transform
        self.normalize = normalize
        self.target_shape = target_shape

        # Validate paths exist
        valid_pairs = []
        for idx, row in self.pairs_df.iterrows():
            if Path(row['baseline_path']).exists() and Path(row['followup_path']).exists():
                valid_pairs.append(idx)
            else:
                logger.debug(f"Missing files for pair {idx}")

        self.pairs_df = self.pairs_df.loc[valid_pairs].reset_index(drop=True)
        logger.info(f"Loaded {len(self.pairs_df)} valid pairs")

        # Log class distribution
        if 'label' in self.pairs_df.columns:
            # New 3-class format (CN=0, MCI=1, AD=2)
            label_counts = self.pairs_df['label'].value_counts().sort_index()
            class_names = ['CN', 'MCI', 'AD']
            for label, count in label_counts.items():
                name = class_names[int(label)] if int(label) < len(class_names) else f"Class{label}"
                logger.info(f"  {name}: {count}")
        elif 'is_converter' in self.pairs_df.columns:
            # Legacy binary format
            converter_counts = self.pairs_df['is_converter'].value_counts()
            logger.info(f"  Converters: {converter_counts.get(1, 0)}")
            logger.info(f"  Non-converters: {converter_counts.get(0, 0)}")

    def __len__(self):
        return len(self.pairs_df)

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """Load and preprocess a single MRI scan (supports NIfTI and NPY)."""
        path = str(path)

        # Load based on file extension
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            if not HAS_NIBABEL:
                raise ImportError("nibabel required for NIfTI files: pip install nibabel")
            nifti_img = nib.load(path)
            data = nifti_img.get_fdata().astype(np.float32)
        elif path.endswith('.npy'):
            data = np.load(path).astype(np.float32)
        else:
            # Try numpy first, then nifti
            try:
                data = np.load(path).astype(np.float32)
            except:
                if HAS_NIBABEL:
                    nifti_img = nib.load(path)
                    data = nifti_img.get_fdata().astype(np.float32)
                else:
                    raise ValueError(f"Unknown file format: {path}")

        # Handle different array shapes
        if data.ndim == 4:
            data = data[..., 0] if data.shape[-1] == 1 else data[0]
        if data.ndim == 2:
            data = np.expand_dims(data, 0)

        # Resize if needed
        if self.target_shape is not None:
            from scipy.ndimage import zoom
            factors = [t / s for t, s in zip(self.target_shape, data.shape)]
            data = zoom(data, factors, order=1)

        # Normalize to [0, 1]
        if self.normalize:
            # Percentile normalization for brain MRI
            brain_voxels = data[data > 0]
            if len(brain_voxels) > 0:
                p1, p99 = np.percentile(brain_voxels, (1, 99))
                data = np.clip(data, p1, p99)
            if data.max() > data.min():
                data = (data - data.min()) / (data.max() - data.min())

        return data

    def __getitem__(self, idx) -> dict:
        row = self.pairs_df.iloc[idx]

        try:
            # Load baseline and followup
            baseline = self._load_and_preprocess(row['baseline_path'])
            followup = self._load_and_preprocess(row['followup_path'])
        except Exception as e:
            # If file is corrupted/empty, return a random valid sample instead
            logger.warning(f"Error loading pair {idx}: {e}. Returning random sample.")
            return self.__getitem__(np.random.randint(0, len(self)))

        # Add channel dimension
        baseline = np.expand_dims(baseline, 0)  # (1, D, H, W)
        followup = np.expand_dims(followup, 0)

        # Apply transforms
        if self.transform:
            baseline = self.transform(baseline)
            followup = self.transform(followup)

        # Get label (supports both 'label' and legacy 'is_converter' columns)
        if 'label' in row:
            label = int(row['label'])
        else:
            label = int(row.get('is_converter', 0))

        # Get time delta in years
        days_between = row.get('days_between', 365)
        if pd.isna(days_between):
            days_between = 365
        time_delta = days_between / 365.25

        return {
            'baseline': torch.from_numpy(baseline).float(),
            'followup': torch.from_numpy(followup).float(),
            'label': torch.tensor(label).long(),
            'time_delta': torch.tensor(time_delta).float(),
            'ptid': row.get('ptid', ''),
            'trajectory': row.get('trajectory', 'unknown')
        }


class RandomAugmentation3D:
    """Simple 3D augmentation for MRI"""

    def __init__(self, flip_prob=0.5, noise_std=0.01):
        self.flip_prob = flip_prob
        self.noise_std = noise_std

    def __call__(self, x):
        # Random flip along each axis
        if np.random.random() < self.flip_prob:
            x = np.flip(x, axis=-1).copy()  # Left-right flip

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
            x = np.clip(x, 0, 1)

        return x


def create_dataloaders(
    pairs_csv: str,
    batch_size: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    target_shape: Optional[Tuple[int, int, int]] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified split.

    Args:
        pairs_csv: Path to pairs CSV
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of data loading workers
        target_shape: Target shape for resizing
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)

    # Stratified split by label
    labels = pairs_df['is_converter'].values if 'is_converter' in pairs_df.columns else np.zeros(len(pairs_df))

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(pairs_df)),
        test_size=test_split,
        stratify=labels,
        random_state=seed
    )

    # Second split: train vs val
    val_ratio = val_split / (1 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=labels[train_val_idx],
        random_state=seed
    )

    # Create split CSVs
    output_dir = Path(pairs_csv).parent
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = pairs_df.iloc[idx]
        split_df.to_csv(output_dir / f'{name}_pairs.csv', index=False)
        logger.info(f"{name}: {len(split_df)} pairs")

    # Create datasets
    train_dataset = PairedMRIDataset(
        output_dir / 'train_pairs.csv',
        transform=RandomAugmentation3D(),
        target_shape=target_shape
    )

    val_dataset = PairedMRIDataset(
        output_dir / 'val_pairs.csv',
        transform=None,
        target_shape=target_shape
    )

    test_dataset = PairedMRIDataset(
        output_dir / 'test_pairs.csv',
        transform=None,
        target_shape=target_shape
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

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

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    import sys

    if len(sys.argv) > 1:
        pairs_csv = sys.argv[1]
        dataset = PairedMRIDataset(pairs_csv)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Baseline shape: {sample['baseline'].shape}")
            print(f"Followup shape: {sample['followup'].shape}")
            print(f"Label: {sample['label']}")
            print(f"Time delta: {sample['time_delta']:.2f} years")
