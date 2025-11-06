#!/usr/bin/env python3
"""
VECNN: Vision Transformer-equipped Convolutional Neural Network
for Alzheimer's Disease Classification

Based on: "Vision transformer-equipped Convolutional Neural Networks for
automated Alzheimer's disease diagnosis using 3D MRI scans"
PMC11682981

Architecture:
- 3D ResNet-50 backbone with modified block distribution [3,3,9,3]
- Non-overlapping stem convolution (4×4×4, stride 4)
- Spatial separable convolution in residual blocks
- GELU activation instead of ReLU
- Vision Transformer-inspired design elements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialSeparableConv3d(nn.Module):
    """
    Spatial separable convolution: separates channel and spatial mixing
    to reduce computational demands
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        # Depthwise convolution (spatial mixing)
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with GELU activation and spatial separable convolution
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()

        # Use spatial separable convolution
        self.conv1 = SpatialSeparableConv3d(in_channels, out_channels, kernel_size=3,
                                            stride=stride, padding=1)
        self.conv2 = SpatialSeparableConv3d(out_channels, out_channels, kernel_size=3,
                                            stride=1, padding=1)

        self.gelu = nn.GELU()  # GELU instead of ReLU
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.gelu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class Bottleneck3D(nn.Module):
    """
    3D Bottleneck block for deeper ResNet variants
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()

        # 1x1x1 conv
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # 3x3x3 conv with spatial separable
        self.conv2 = SpatialSeparableConv3d(out_channels, out_channels,
                                            kernel_size=3, stride=stride, padding=1)

        # 1x1x1 conv
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)

        self.gelu = nn.GELU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class VECNN(nn.Module):
    """
    Vision Transformer-equipped Convolutional Neural Network (VECNN)

    Key modifications from standard ResNet-50:
    1. Non-overlapping stem convolution (4×4×4, stride 4)
    2. Modified block distribution: [3,3,9,3] instead of [3,4,6,3]
    3. Spatial separable convolutions
    4. GELU activation
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 1):
        super().__init__()

        self.in_channels = 64

        # Non-overlapping stem convolution (4×4×4, stride 4)
        # This replaces the standard 7×7×7 conv with stride 2 + maxpool
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm3d(64),
            nn.GELU()
        )

        # Modified block distribution: [3,3,9,3] (aligned with Swin Transformer's 1:1:3:1)
        self.layer1 = self._make_layer(Bottleneck3D, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck3D, 128, 3, stride=2)
        self.layer3 = self._make_layer(Bottleneck3D, 256, 9, stride=2)  # 9 blocks (increased)
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classification head
        self.fc = nn.Linear(512 * Bottleneck3D.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple residual blocks"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, D, H, W)

        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract feature maps from different layers for visualization
        """
        x = self.stem(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return f1, f2, f3, f4


def vecnn_small(num_classes: int = 3, in_channels: int = 1) -> VECNN:
    """Create VECNN model (standard configuration)"""
    return VECNN(num_classes=num_classes, in_channels=in_channels)


if __name__ == '__main__':
    # Test the model
    model = vecnn_small(num_classes=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"VECNN Model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 1, 192, 192, 192)  # Batch of 2, 192^3 volumes
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0]}")
