#!/usr/bin/env python3
"""
3D Convolutional Neural Network for Alzheimer's Disease Classification

Architectures:
- cnn3d_small:  4 conv blocks, 32->64->128->256 channels (~2.5M params)
- cnn3d_base:   5 conv blocks, 32->64->128->256->512 channels (~11M params)
- cnn3d_large:  5 conv blocks, 64->128->256->512->512 channels (~25M params)

Input: (B, 1, 128, 128, 128) MRI volumes
Output: (B, num_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


CNN3D_CONFIGS = {
    'cnn3d_small': {
        'channels': [32, 64, 128, 256],
        'use_residual': False,
    },
    'cnn3d_base': {
        'channels': [32, 64, 128, 256, 512],
        'use_residual': True,
    },
    'cnn3d_large': {
        'channels': [64, 128, 256, 512, 512],
        'use_residual': True,
    },
}


class ConvBlock3D(nn.Module):
    """3D convolution block: Conv3d -> BatchNorm -> ReLU -> Conv3d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        elif use_residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            out = out + self.shortcut(identity)

        return self.relu(out)


class CNN3DClassifier(nn.Module):
    """
    3D CNN for volumetric MRI classification.

    Architecture: ConvBlock3D stages with MaxPool3d downsampling,
    followed by global average pooling and a classification head.
    """

    def __init__(
        self,
        architecture: str = 'cnn3d_base',
        num_classes: int = 2,
        in_channels: int = 1,
        dropout: float = 0.1,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()

        config = CNN3D_CONFIGS.get(architecture, CNN3D_CONFIGS['cnn3d_base'])
        channels = config['channels']
        use_residual = config['use_residual']

        self.architecture = architecture
        self.num_classes = num_classes

        # Build encoder stages
        stages = []
        prev_channels = in_channels
        for ch in channels:
            stages.append(ConvBlock3D(prev_channels, ch, use_residual=use_residual))
            stages.append(nn.MaxPool3d(kernel_size=2, stride=2))
            stages.append(nn.Dropout3d(p=dropout))
            prev_channels = ch

        self.encoder = nn.Sequential(*stages)
        self.feature_dim = channels[-1]

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"Created CNN3DClassifier ({architecture})")
        logger.info(f"  Channels: {channels}, Residual: {use_residual}")
        logger.info(f"  Feature dim: {self.feature_dim}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        x = self.encoder(x)
        x = self.global_pool(x)       # (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)     # (B, C)
        return self.head(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        x = self.encoder(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)


def get_cnn3d_model(
    architecture: str = 'cnn3d_base',
    num_classes: int = 2,
    in_channels: int = 1,
    dropout: float = 0.1,
    classifier_dropout: float = 0.5,
) -> nn.Module:
    """Convenience function to create a CNN3D model"""
    return CNN3DClassifier(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=dropout,
        classifier_dropout=classifier_dropout,
    )


if __name__ == '__main__':
    print("Testing CNN3D models...")

    for arch in ['cnn3d_small', 'cnn3d_base', 'cnn3d_large']:
        print(f"\n{arch}:")

        model = CNN3DClassifier(
            architecture=arch,
            num_classes=2,
            in_channels=1,
            dropout=0.1,
            classifier_dropout=0.5,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        x = torch.randn(1, 1, 128, 128, 128)
        out = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

        features = model.extract_features(x)
        print(f"  Features: {features.shape}")
