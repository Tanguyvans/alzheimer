#!/usr/bin/env python3
"""
Multi-View Dataset for MedTransformer

Extracts 2D slices from 3 orthogonal views:
- Axial (horizontal): volume[:, :, z]
- Coronal (frontal): volume[:, y, :]
- Sagittal (side): volume[x, :, :]

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ['CN', 'MCI_stable', 'MCI_to_AD', 'AD']


def get_transforms(config: dict, training: bool = True) -> transforms.Compose:
    """Get image transforms"""
    image_size = config['model']['image_size']

    transform_list = [transforms.Resize((image_size, image_size))]

    if training and config.get('augmentation', {}).get('enabled', True):
        aug = config.get('augmentation', {})
        if aug.get('random_horizontal_flip', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug['random_horizontal_flip']))
        if aug.get('random_rotation', 0) > 0:
            transform_list.append(transforms.RandomRotation(aug['random_rotation']))
        if aug.get('color_jitter', False):
            transform_list.append(transforms.ColorJitter(
                brightness=aug.get('brightness', 0.1),
                contrast=aug.get('contrast', 0.1)
            ))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get('augmentation', {}).get('mean', [0.485, 0.456, 0.406]),
            std=config.get('augmentation', {}).get('std', [0.229, 0.224, 0.225])
        )
    ])

    return transforms.Compose(transform_list)


class MultiViewMRIDataset(Dataset):
    """
    Dataset for MedTransformer - extracts slices from 3 views

    Returns:
        axial_slices: (num_slices, C, H, W)
        coronal_slices: (num_slices, C, H, W)
        sagittal_slices: (num_slices, C, H, W)
        label: int
    """

    def __init__(
        self,
        csv_path: str,
        config: dict,
        training: bool = True
    ):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.training = training

        # Slice extraction settings
        self.num_slices = config['model']['num_slices']
        self.slice_range = config.get('slice_extraction', {}).get('range', [0.25, 0.75])

        # Image transforms
        self.transform = get_transforms(config, training)

        # Log info
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"Extracting {self.num_slices} slices per view from {self.slice_range}")
        for i, name in enumerate(CLASS_NAMES):
            count = len(self.df[self.df['label'] == i])
            logger.info(f"  {name}: {count}")

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to 0-1 range"""
        brain_voxels = volume[volume > 0]
        if len(brain_voxels) > 0:
            p1, p99 = np.percentile(brain_voxels, (1, 99))
            volume = np.clip(volume, p1, p99)

        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume

    def _extract_slices(self, volume: np.ndarray, axis: int) -> List[np.ndarray]:
        """
        Extract evenly-spaced slices from specified axis

        Args:
            volume: 3D numpy array
            axis: 0=sagittal, 1=coronal, 2=axial
        """
        depth = volume.shape[axis]
        start = int(depth * self.slice_range[0])
        end = int(depth * self.slice_range[1])

        # Get evenly spaced slice indices
        indices = np.linspace(start, end - 1, self.num_slices, dtype=int)

        slices = []
        for idx in indices:
            if axis == 0:  # Sagittal
                slice_2d = volume[idx, :, :]
            elif axis == 1:  # Coronal
                slice_2d = volume[:, idx, :]
            else:  # Axial
                slice_2d = volume[:, :, idx]

            # Convert to uint8
            slice_2d = (slice_2d * 255).astype(np.uint8)
            slices.append(slice_2d)

        return slices

    def _process_slices(self, slices: List[np.ndarray]) -> torch.Tensor:
        """Process list of 2D slices to tensor"""
        processed = []
        for slice_2d in slices:
            # Convert to PIL Image (RGB)
            pil_img = Image.fromarray(slice_2d).convert('RGB')
            # Apply transforms
            tensor = self.transform(pil_img)
            processed.append(tensor)

        return torch.stack(processed, dim=0)  # (num_slices, C, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        scan_path = row['scan_path']
        label = int(row['label'])

        try:
            # Load 3D volume
            nifti_img = nib.load(scan_path)
            volume = nifti_img.get_fdata().astype(np.float32)

            # Normalize
            volume = self._normalize_volume(volume)

            # Extract slices from each view
            axial_slices = self._extract_slices(volume, axis=2)
            coronal_slices = self._extract_slices(volume, axis=1)
            sagittal_slices = self._extract_slices(volume, axis=0)

            # Process to tensors
            axial_tensor = self._process_slices(axial_slices)
            coronal_tensor = self._process_slices(coronal_slices)
            sagittal_tensor = self._process_slices(sagittal_slices)

            return axial_tensor, coronal_tensor, sagittal_tensor, label

        except Exception as e:
            logger.warning(f"Error loading {scan_path}: {e}")
            # Return dummy data
            dummy = torch.zeros(self.num_slices, 3, self.config['model']['image_size'],
                              self.config['model']['image_size'])
            return dummy, dummy, dummy, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced classes"""
        class_counts = [len(self.df[self.df['label'] == i]) for i in range(4)]
        total = sum(class_counts)
        weights = [total / (4 * c) if c > 0 else 0 for c in class_counts]
        return torch.FloatTensor(weights)


class SingleViewMRIDataset(Dataset):
    """
    Single-view dataset for simpler baseline comparison

    Extracts only axial slices (hippocampus region)
    """

    def __init__(
        self,
        csv_path: str,
        config: dict,
        training: bool = True,
        view: str = 'axial'
    ):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.training = training
        self.view = view

        self.num_slices = config['model']['num_slices']
        self.slice_range = config.get('slice_extraction', {}).get('range', [0.40, 0.60])
        self.transform = get_transforms(config, training)

        logger.info(f"SingleView ({view}): {len(self.df)} samples, {self.num_slices} slices")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = row['scan_path']
        label = int(row['label'])

        try:
            nifti_img = nib.load(scan_path)
            volume = nifti_img.get_fdata().astype(np.float32)

            # Normalize
            brain_voxels = volume[volume > 0]
            if len(brain_voxels) > 0:
                p1, p99 = np.percentile(brain_voxels, (1, 99))
                volume = np.clip(volume, p1, p99)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

            # Get axis
            axis = {'sagittal': 0, 'coronal': 1, 'axial': 2}[self.view]
            depth = volume.shape[axis]
            start = int(depth * self.slice_range[0])
            end = int(depth * self.slice_range[1])
            indices = np.linspace(start, end - 1, self.num_slices, dtype=int)

            slices = []
            for idx in indices:
                if axis == 0:
                    s = volume[idx, :, :]
                elif axis == 1:
                    s = volume[:, idx, :]
                else:
                    s = volume[:, :, idx]

                s = (s * 255).astype(np.uint8)
                pil_img = Image.fromarray(s).convert('RGB')
                slices.append(self.transform(pil_img))

            return torch.stack(slices, dim=0), label

        except Exception as e:
            logger.warning(f"Error: {e}")
            dummy = torch.zeros(self.num_slices, 3, self.config['model']['image_size'],
                              self.config['model']['image_size'])
            return dummy, label


def collate_multiview(batch):
    """Custom collate function for multi-view dataset"""
    axial = torch.stack([item[0] for item in batch])
    coronal = torch.stack([item[1] for item in batch])
    sagittal = torch.stack([item[2] for item in batch])
    labels = torch.tensor([item[3] for item in batch])
    return axial, coronal, sagittal, labels


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train, val, test dataloaders for MedTransformer"""

    train_dataset = MultiViewMRIDataset(train_csv, config, training=True)
    val_dataset = MultiViewMRIDataset(val_csv, config, training=False)
    test_dataset = MultiViewMRIDataset(test_csv, config, training=False)

    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multiview
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multiview
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multiview
    )

    class_weights = train_dataset.get_class_weights()

    logger.info(f"\nDataloaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info(f"  Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    import yaml
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <config.yaml>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['train_csv']
    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    dataset = MultiViewMRIDataset(csv_path, config)
    axial, coronal, sagittal, label = dataset[0]

    print(f"\nFirst sample:")
    print(f"  Axial shape: {axial.shape}")
    print(f"  Coronal shape: {coronal.shape}")
    print(f"  Sagittal shape: {sagittal.shape}")
    print(f"  Label: {label} ({CLASS_NAMES[label]})")
