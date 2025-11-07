#!/usr/bin/env python3
"""
3D ResNet for Alzheimer's Disease Classification with MedicalNet Pretrained Weights

Based on MedicalNet: https://github.com/Tencent/MedicalNet
Pretrained on 23 medical imaging datasets

Supports:
- ResNet-10
- ResNet-18
- ResNet-34
- ResNet-50

Download pretrained weights from:
https://github.com/Tencent/MedicalNet
or
https://huggingface.co/TencentMedicalNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """Basic 3D ResNet block (for ResNet-10, ResNet-18)"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck 3D ResNet block (for ResNet-34, ResNet-50)"""
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """3D ResNet for medical image classification"""

    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 3,
        in_channels: int = 1,
        shortcut_type: str = 'B'
    ):
        super().__init__()

        self.in_planes = 64
        self.shortcut_type = shortcut_type

        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a layer with multiple blocks"""
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = lambda x: F.pad(
                    x[:, :, ::stride, ::stride, ::stride],
                    (0, 0, 0, 0, 0, 0, planes * block.expansion // 4, planes * block.expansion // 4),
                    "constant", 0
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

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
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """Construct a ResNet-10 model"""
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Construct a ResNet-18 model"""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Construct a ResNet-34 model"""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Construct a ResNet-50 model"""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def load_pretrained_weights(model: nn.Module, pretrained_path: str, num_classes: int = 3):
    """
    Load MedicalNet pretrained weights

    Args:
        model: ResNet3D model
        pretrained_path: Path to pretrained .pth file
        num_classes: Number of output classes (3 for CN/MCI/AD)
    """
    logger.info(f"Loading pretrained weights from {pretrained_path}")

    # Load pretrained state dict
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    # Remove 'module.' prefix if present (from DataParallel)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Get model state dict
    model_dict = model.state_dict()

    # Filter out keys that don't match (especially fc layer)
    pretrained_dict_filtered = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            # Skip fc layer (different number of classes)
            if k.startswith('fc'):
                logger.info(f"  Skipping layer {k} (output size mismatch)")
                continue

            # Check shape match
            if v.shape == model_dict[k].shape:
                pretrained_dict_filtered[k] = v
            else:
                logger.warning(f"  Skipping layer {k} due to shape mismatch: "
                             f"{v.shape} vs {model_dict[k].shape}")

    # Update model dict
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)

    logger.info(f"Loaded {len(pretrained_dict_filtered)}/{len(model_dict)} layers from pretrained weights")
    logger.info(f"FC layer randomly initialized for {num_classes} classes")

    return model


if __name__ == '__main__':
    # Test the models
    for model_fn, name in [(resnet10, 'ResNet-10'),
                           (resnet18, 'ResNet-18'),
                           (resnet34, 'ResNet-34'),
                           (resnet50, 'ResNet-50')]:
        model = model_fn(num_classes=3, in_channels=1)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test forward pass
        x = torch.randn(1, 1, 192, 192, 192)
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
