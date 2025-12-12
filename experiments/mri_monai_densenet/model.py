#!/usr/bin/env python3
"""
3D DenseNet with MONAI and MedicalNet pretrained weights support

Supports:
- MONAI's native DenseNet implementations
- Loading MedicalNet pretrained weights (https://github.com/Tencent/MedicalNet)

Download pretrained weights:
- MedicalNet: https://github.com/Tencent/MedicalNet
  - densenet_121_23dataset.pth
  - densenet_169_23dataset.pth
  - densenet_201_23dataset.pth
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_densenet_model(
    architecture: str = 'densenet121',
    num_classes: int = 3,
    in_channels: int = 1,
    pretrained_path: Optional[str] = None,
    freeze_backbone: bool = False,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    Create a 3D DenseNet model with optional pretrained weights.

    Args:
        architecture: One of 'densenet121', 'densenet169', 'densenet201', 'densenet264'
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale MRI)
        pretrained_path: Path to MedicalNet pretrained weights (.pth file)
        freeze_backbone: Whether to freeze backbone layers (only train classifier)
        spatial_dims: 2 or 3 for 2D/3D models

    Returns:
        DenseNet model
    """
    model_dict = {
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'densenet264': DenseNet264,
    }

    if architecture not in model_dict:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(model_dict.keys())}")

    # Create MONAI DenseNet
    model = model_dict[architecture](
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=num_classes,
    )

    logger.info(f"Created {architecture} (spatial_dims={spatial_dims}, in_channels={in_channels})")

    # Load pretrained weights if specified
    if pretrained_path and Path(pretrained_path).exists():
        model = load_medicalnet_weights(model, pretrained_path, num_classes)

    # Freeze backbone if requested
    if freeze_backbone:
        freeze_backbone_layers(model)

    return model


def load_medicalnet_weights(
    model: nn.Module,
    pretrained_path: str,
    num_classes: int
) -> nn.Module:
    """
    Load MedicalNet pretrained weights into MONAI DenseNet.

    MedicalNet weights are trained on 23 medical imaging datasets.

    Args:
        model: MONAI DenseNet model
        pretrained_path: Path to MedicalNet .pth file
        num_classes: Number of output classes

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading MedicalNet weights from {pretrained_path}")

    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    # Remove 'module.' prefix (from DataParallel training)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Get model state dict
    model_dict = model.state_dict()

    # Map MedicalNet keys to MONAI keys
    # MedicalNet uses: features.*, classifier.*
    # MONAI uses: features.*, class_layers.*
    loaded_count = 0
    skipped_count = 0

    for key, value in pretrained_dict.items():
        # Skip final classification layer
        if 'classifier' in key or 'class_layers' in key:
            logger.debug(f"  Skipping {key} (classifier layer)")
            skipped_count += 1
            continue

        # Try direct match first
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                logger.warning(f"  Shape mismatch for {key}: {value.shape} vs {model_dict[key].shape}")
                skipped_count += 1
        else:
            # Try with 'features.' prefix for MONAI compatibility
            monai_key = key
            if not key.startswith('features.') and 'features.' + key in model_dict:
                monai_key = 'features.' + key

            if monai_key in model_dict:
                if value.shape == model_dict[monai_key].shape:
                    model_dict[monai_key] = value
                    loaded_count += 1
                else:
                    logger.warning(f"  Shape mismatch for {monai_key}: {value.shape} vs {model_dict[monai_key].shape}")
                    skipped_count += 1
            else:
                logger.debug(f"  Key not found: {key}")
                skipped_count += 1

    # Load updated state dict
    model.load_state_dict(model_dict, strict=False)

    logger.info(f"Loaded {loaded_count} layers, skipped {skipped_count}")
    logger.info(f"Classifier layer randomly initialized for {num_classes} classes")

    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    """
    Freeze all layers except the final classifier.

    Args:
        model: DenseNet model
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if 'class_layers' in name or 'classifier' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"Backbone frozen: {frozen_count:,} params frozen, {trainable_count:,} trainable")


class DenseNet3DClassifier(nn.Module):
    """
    Wrapper around MONAI DenseNet with additional features:
    - Dropout before classifier
    - Feature extraction mode
    """

    def __init__(
        self,
        architecture: str = 'densenet121',
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Create base model (initially with placeholder num_classes)
        self.base_model = get_densenet_model(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
        )

        # Get feature dimension and replace classifier
        # MONAI DenseNet121 has 1024 features before classifier
        feature_dims = {
            'densenet121': 1024,
            'densenet169': 1664,
            'densenet201': 1920,
            'densenet264': 2688,
        }
        in_features = feature_dims.get(architecture, 1024)

        # Replace the classifier with dropout + linear
        if hasattr(self.base_model, 'class_layers'):
            self.base_model.class_layers = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        # If base_model still returns logits, use them directly
        if features.shape[-1] == self.num_classes:
            return features
        # Otherwise, pass through our classifier
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.base_model.features(x)


if __name__ == '__main__':
    # Test model creation
    print("Testing MONAI DenseNet models...")

    for arch in ['densenet121', 'densenet169']:
        model = get_densenet_model(
            architecture=arch,
            num_classes=3,
            in_channels=1
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{arch}:")
        print(f"  Parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 1, 96, 96, 96)
        out = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Test wrapper class
    print("\n\nTesting DenseNet3DClassifier wrapper...")
    model = DenseNet3DClassifier(
        architecture='densenet121',
        num_classes=3,
        dropout=0.5
    )
    x = torch.randn(1, 1, 96, 96, 96)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
