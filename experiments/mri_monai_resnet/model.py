#!/usr/bin/env python3
"""
3D ResNet with MONAI and MedicalNet pretrained weights support

Supports:
- MONAI's native ResNet implementations
- Loading MedicalNet pretrained weights (https://github.com/Tencent/MedicalNet)
- Loading weights from HuggingFace

Download pretrained weights:
- MedicalNet: https://github.com/Tencent/MedicalNet
- HuggingFace: https://huggingface.co/TencentMedicalNet
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# MedicalNet architecture mapping (for weight compatibility)
MEDICALNET_CONFIGS = {
    'resnet10': {'model_depth': 10, 'n_seg_classes': 2},
    'resnet18': {'model_depth': 18, 'n_seg_classes': 2},
    'resnet34': {'model_depth': 34, 'n_seg_classes': 2},
    'resnet50': {'model_depth': 50, 'n_seg_classes': 2},
}


def get_resnet_model(
    architecture: str = 'resnet50',
    num_classes: int = 3,
    in_channels: int = 1,
    pretrained_path: Optional[str] = None,
    freeze_backbone: bool = False,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    Create a 3D ResNet model with optional pretrained weights.

    Args:
        architecture: One of 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale MRI)
        pretrained_path: Path to MedicalNet pretrained weights (.pth file)
        freeze_backbone: Whether to freeze backbone layers (only train classifier)
        spatial_dims: 2 or 3 for 2D/3D models

    Returns:
        ResNet model
    """
    model_dict = {
        'resnet10': resnet10,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }

    if architecture not in model_dict:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(model_dict.keys())}")

    # Create MONAI ResNet
    model = model_dict[architecture](
        spatial_dims=spatial_dims,
        n_input_channels=in_channels,
        num_classes=num_classes,
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
    Load MedicalNet pretrained weights into MONAI ResNet.

    MedicalNet weights are trained on 23 medical imaging datasets.

    Args:
        model: MONAI ResNet model
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

    # Map MedicalNet keys to MONAI keys (they should be compatible)
    loaded_count = 0
    skipped_count = 0

    for key, value in pretrained_dict.items():
        # Skip final classification layer
        if 'fc' in key or 'classifier' in key:
            logger.debug(f"  Skipping {key} (classifier layer)")
            skipped_count += 1
            continue

        # Check if key exists in model
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                logger.warning(f"  Shape mismatch for {key}: {value.shape} vs {model_dict[key].shape}")
                skipped_count += 1
        else:
            logger.debug(f"  Key not found: {key}")
            skipped_count += 1

    # Load updated state dict
    model.load_state_dict(model_dict)

    logger.info(f"Loaded {loaded_count} layers, skipped {skipped_count}")
    logger.info(f"Classifier layer randomly initialized for {num_classes} classes")

    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    """
    Freeze all layers except the final classifier.

    Args:
        model: ResNet model
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"Backbone frozen: {frozen_count:,} params frozen, {trainable_count:,} trainable")


class ResNet3DClassifier(nn.Module):
    """
    Wrapper around MONAI ResNet with additional features:
    - Dropout before classifier
    - Feature extraction mode
    - Gradient checkpointing
    """

    def __init__(
        self,
        architecture: str = 'resnet50',
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Create base model
        self.base_model = get_resnet_model(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
        )

        # Get feature dimension from base model
        if hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            in_features = 512 if 'resnet10' in architecture or 'resnet18' in architecture else 2048

        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.base_model(x)


if __name__ == '__main__':
    # Test model creation
    print("Testing MONAI ResNet models...")

    for arch in ['resnet10', 'resnet18', 'resnet50']:
        model = get_resnet_model(
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
    print("\n\nTesting ResNet3DClassifier wrapper...")
    model = ResNet3DClassifier(
        architecture='resnet50',
        num_classes=3,
        dropout=0.5
    )
    x = torch.randn(1, 1, 96, 96, 96)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
