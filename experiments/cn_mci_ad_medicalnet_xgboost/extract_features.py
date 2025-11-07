#!/usr/bin/env python3
"""
Extract features from MRI scans using pretrained MedicalNet ResNet-18

This script:
1. Loads pretrained ResNet-18 from MedicalNet
2. Removes the final classification layer (use as feature extractor)
3. Extracts 512-dimensional features from each MRI scan
4. Saves features to CSV for XGBoost training
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import nibabel as nib
from scipy.ndimage import zoom

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent / 'cn_mci_ad_medicalnet'))
from model_resnet3d import resnet18, load_pretrained_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from MRI scans using pretrained ResNet"""

    def __init__(self, pretrained_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load pretrained model
        logger.info("Loading pretrained ResNet-18...")
        self.model = resnet18(num_classes=3, in_channels=1)

        if Path(pretrained_path).exists():
            self.model = load_pretrained_weights(self.model, pretrained_path, num_classes=3)
        else:
            logger.warning(f"Pretrained weights not found at {pretrained_path}")
            logger.warning("Using random initialization...")

        # Remove final FC layer to get features
        self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Feature extractor ready on {self.device}")

    @torch.no_grad()
    def extract_from_file(self, nifti_path: str, target_shape: tuple = (192, 192, 192)) -> np.ndarray:
        """
        Extract features from a single NIfTI file

        Args:
            nifti_path: Path to .nii.gz file
            target_shape: Target volume shape

        Returns:
            features: 512-dimensional feature vector
        """
        # Load NIfTI
        nifti_img = nib.load(nifti_path)
        volume = nifti_img.get_fdata().astype(np.float32)

        # Normalize
        brain_voxels = volume[volume > 0]
        if len(brain_voxels) > 0:
            p1, p99 = np.percentile(brain_voxels, (1, 99))
            volume = np.clip(volume, p1, p99)

        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Resize to target shape
        volume = self._resize_volume(volume, target_shape)

        # Convert to tensor: (1, 1, D, H, W)
        volume = torch.from_numpy(volume).float()
        volume = volume.unsqueeze(0).unsqueeze(0)
        volume = volume.to(self.device)

        # Extract features
        features = self.model(volume)

        # Return as numpy array
        return features.cpu().numpy().flatten()

    def _resize_volume(self, volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize volume using scipy zoom"""
        current_shape = volume.shape
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        resized = zoom(volume, zoom_factors, order=1)
        return resized


def extract_features_from_csv(
    csv_path: str,
    pretrained_path: str,
    output_path: str,
    device: str = 'cuda'
):
    """
    Extract features from all scans listed in CSV

    Args:
        csv_path: Path to CSV with columns: path, label
        pretrained_path: Path to pretrained weights
        output_path: Where to save features CSV
        device: cuda or cpu
    """
    logger.info("="*80)
    logger.info("EXTRACTING FEATURES FROM MRI SCANS")
    logger.info("="*80)
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output CSV: {output_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} scans")

    # Check column names (could be 'path' or 'scan_path')
    path_col = 'scan_path' if 'scan_path' in df.columns else 'path'

    # Initialize feature extractor
    extractor = FeatureExtractor(pretrained_path, device)

    # Extract features
    features_list = []
    labels_list = []
    paths_list = []
    failed_scans = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        scan_path = row[path_col]
        label = row['label']

        try:
            features = extractor.extract_from_file(scan_path)
            features_list.append(features)
            labels_list.append(label)
            paths_list.append(scan_path)
        except Exception as e:
            logger.error(f"Failed to extract features from {scan_path}: {e}")
            failed_scans.append(scan_path)

    logger.info(f"\nSuccessfully extracted features from {len(features_list)}/{len(df)} scans")

    if failed_scans:
        logger.warning(f"Failed scans: {len(failed_scans)}")
        for scan in failed_scans[:5]:
            logger.warning(f"  {scan}")

    # Create features DataFrame
    features_array = np.array(features_list)
    logger.info(f"Features shape: {features_array.shape}")

    # Create column names: feature_0, feature_1, ..., feature_511
    feature_cols = [f'feature_{i}' for i in range(features_array.shape[1])]

    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df['label'] = labels_list
    features_df['scan_path'] = paths_list

    # Save to CSV
    features_df.to_csv(output_path, index=False)
    logger.info(f"\nFeatures saved to {output_path}")

    # Print statistics
    logger.info("\nFeature statistics:")
    logger.info(f"  Mean: {features_array.mean():.4f}")
    logger.info(f"  Std: {features_array.std():.4f}")
    logger.info(f"  Min: {features_array.min():.4f}")
    logger.info(f"  Max: {features_array.max():.4f}")

    # Class distribution
    logger.info("\nClass distribution:")
    label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    for label in sorted(set(labels_list)):
        count = labels_list.count(label)
        logger.info(f"  {label_names[label]}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Extract features using MedicalNet ResNet-18')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV with scan paths and labels')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained ResNet-18 weights')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path for features')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')

    args = parser.parse_args()

    extract_features_from_csv(
        csv_path=args.csv,
        pretrained_path=args.pretrained,
        output_path=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
