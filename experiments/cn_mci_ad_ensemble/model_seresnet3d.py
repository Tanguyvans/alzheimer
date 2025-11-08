#!/usr/bin/env python3
"""
3D SEResNet (Squeeze-and-Excitation ResNet) for Alzheimer's Classification

Based on:
- "Deep CNN ResNet-18 based model with attention and transfer learning
   for Alzheimer's disease detection" (2024)
- Achieves 93.26% accuracy on ADNI dataset

SE blocks add channel attention with minimal parameter overhead:
- Recalibrates channel-wise feature responses
- Emphasizes important features, suppresses irrelevant ones
- Only ~2M extra parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    1. Global Average Pooling (Squeeze): Aggregate spatial information
    2. FC layers (Excitation): Learn channel-wise attention weights
    3. Sigmoid activation: Scale attention weights to [0, 1]
    4. Multiply: Recalibrate feature maps
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck (default: 16)
        """
        super().__init__()

        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool3d(1)

        # Excitation: Two FC layers with ReLU
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Recalibrated tensor of same shape
        """
        b, c, _, _, _ = x.size()

        # Squeeze: (B, C, D, H, W) -> (B, C, 1, 1, 1) -> (B, C)
        squeeze = self.squeeze(x).view(b, c)

        # Excitation: (B, C) -> (B, C)
        excitation = self.excitation(squeeze).view(b, c, 1, 1, 1)

        # Scale: Multiply input by attention weights
        return x * excitation.expand_as(x)


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,  # groups>1 enables depth-wise convolution
        bias=False
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEBasicBlock(nn.Module):
    """
    Basic 3D ResNet block with SE attention

    Architecture:
    - 3x3x3 conv -> BN -> ReLU
    - 3x3x3 conv -> BN
    - SE Block (channel attention)
    - Add residual -> ReLU
    """
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        # SE block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes, reduction=se_reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    """
    Bottleneck 3D ResNet block with SE attention

    For deeper networks (ResNet-34, ResNet-50)
    """
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        # SE block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes * self.expansion, reduction=se_reduction)

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

        # Apply SE attention
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet3D(nn.Module):
    """3D SEResNet for medical image classification"""

    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 3,
        in_channels: int = 1,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        self.in_planes = 64
        self.use_se = use_se

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

        # ResNet layers with SE blocks
        self.layer1 = self._make_layer(block, 64, layers[0], use_se=use_se, se_reduction=se_reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se, se_reduction=se_reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se, se_reduction=se_reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se, se_reduction=se_reduction)

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
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 16
    ) -> nn.Sequential:
        """Create a layer with multiple blocks"""
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.in_planes, planes, stride, downsample, use_se=use_se, se_reduction=se_reduction)
        )
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, use_se=use_se, se_reduction=se_reduction)
            )

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
                if m.bias is not None:
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


def seresnet10(**kwargs):
    """Construct a SEResNet-10 model"""
    return SEResNet3D(SEBasicBlock, [1, 1, 1, 1], **kwargs)


def seresnet18(**kwargs):
    """Construct a SEResNet-18 model"""
    return SEResNet3D(SEBasicBlock, [2, 2, 2, 2], **kwargs)


def seresnet34(**kwargs):
    """Construct a SEResNet-34 model"""
    return SEResNet3D(SEBasicBlock, [3, 4, 6, 3], **kwargs)


def seresnet50(**kwargs):
    """Construct a SEResNet-50 model"""
    return SEResNet3D(SEBottleneck, [3, 4, 6, 3], **kwargs)


def load_pretrained_weights(model: nn.Module, pretrained_path: str, num_classes: int = 3):
    """
    Load MedicalNet pretrained weights into SEResNet

    Note: Pretrained weights don't have SE blocks, so we only load conv/bn layers
    """
    logger.info(f"Loading pretrained weights from {pretrained_path}")

    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    # Remove 'module.' prefix
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Get model state dict
    model_dict = model.state_dict()

    # Filter weights (skip SE blocks and fc layer)
    pretrained_dict_filtered = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            # Skip fc layer and SE blocks (not in pretrained)
            if k.startswith('fc') or 'se.' in k:
                logger.info(f"  Skipping layer {k}")
                continue

            # Check shape match
            if v.shape == model_dict[k].shape:
                pretrained_dict_filtered[k] = v
            else:
                logger.warning(f"  Skipping layer {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")

    # Update model dict
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)

    logger.info(f"Loaded {len(pretrained_dict_filtered)}/{len(model_dict)} layers from pretrained weights")
    logger.info(f"SE blocks and FC layer randomly initialized")

    return model


if __name__ == '__main__':
    # Test the models
    for model_fn, name in [(seresnet10, 'SEResNet-10'),
                           (seresnet18, 'SEResNet-18'),
                           (seresnet34, 'SEResNet-34'),
                           (seresnet50, 'SEResNet-50')]:
        model = model_fn(num_classes=3, in_channels=1, use_se=True)

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
