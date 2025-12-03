#!/usr/bin/env python3
"""
Longitudinal MRI Models for Alzheimer's Disease Classification.

Models for comparing baseline and follow-up MRI scans to predict
cognitive status (CN/MCI/AD).

Available models:
- SiameseNetwork: Shared encoder with difference features
- WeightedSiameseNetwork: Siamese with temporal attention
- Longitudinal3DCNN: 2-channel input (baseline + followup)
- DifferenceNet3D: 3-channel input (baseline + followup + difference)
- LongitudinalMedicalNet: MedicalNet encoder + LSTM temporal modeling
- LongitudinalAttentionNet: MedicalNet encoder + cross-attention
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetBlock3D(nn.Module):
    """Basic 3D ResNet block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# =============================================================================
# Encoders
# =============================================================================

class Encoder3D(nn.Module):
    """Lightweight 3D CNN encoder for MRI feature extraction."""

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.MaxPool3d(2),
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
            nn.MaxPool3d(2),
            ConvBlock3D(base_channels * 2, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
            nn.MaxPool3d(2),
            ConvBlock3D(base_channels * 4, base_channels * 8),
            ConvBlock3D(base_channels * 8, base_channels * 8),
            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.fc(self.encoder(x))


class MedicalNetEncoder(nn.Module):
    """
    3D ResNet encoder compatible with MedicalNet pretrained weights.

    Download weights from: https://github.com/Tencent/MedicalNet
    """

    BLOCK_CONFIGS = {
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
    }

    def __init__(self, in_channels=1, resnet_depth=10):
        super().__init__()

        if resnet_depth not in self.BLOCK_CONFIGS:
            raise ValueError(f"Unsupported depth {resnet_depth}. Choose from {list(self.BLOCK_CONFIGS.keys())}")

        layers = self.BLOCK_CONFIGS[resnet_depth]

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = 512

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = [ResNetBlock3D(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


# =============================================================================
# Siamese Models (Shared Encoder)
# =============================================================================

class SiameseNetwork(nn.Module):
    """
    Siamese Network for comparing baseline and follow-up MRI.

    Uses shared encoder for both scans, then combines embeddings
    with difference features for classification.
    """

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256,
                 num_classes=3, dropout=0.5):
        super().__init__()

        self.encoder = Encoder3D(in_channels, base_channels, embedding_dim)

        classifier_input_dim = embedding_dim * 3 + 1  # baseline, followup, diff, time

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, num_classes)
        )

    def forward(self, baseline, followup, time_delta=None):
        emb_baseline = self.encoder(baseline)
        emb_followup = self.encoder(followup)
        diff = emb_followup - emb_baseline

        if time_delta is None:
            time_delta = torch.ones(baseline.size(0), 1, device=baseline.device)
        elif time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(1)

        combined = torch.cat([emb_baseline, emb_followup, diff, time_delta], dim=1)
        return self.classifier(combined)

    def get_embeddings(self, baseline, followup):
        return self.encoder(baseline), self.encoder(followup)


class WeightedSiameseNetwork(SiameseNetwork):
    """Siamese Network with temporal attention weighting."""

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256,
                 num_classes=3, dropout=0.5):
        super().__init__(in_channels, base_channels, embedding_dim, num_classes, dropout)

        self.temporal_attention = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, baseline, followup, time_delta=None):
        emb_baseline = self.encoder(baseline)
        emb_followup = self.encoder(followup)

        if time_delta is None:
            time_delta = torch.ones(baseline.size(0), 1, device=baseline.device)
        elif time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(1)

        attention = self.temporal_attention(time_delta)
        diff = (emb_followup - emb_baseline) * attention

        combined = torch.cat([emb_baseline, emb_followup, diff, time_delta], dim=1)
        return self.classifier(combined)


# =============================================================================
# Multi-Channel Models (No Shared Encoder)
# =============================================================================

class Longitudinal3DCNN(nn.Module):
    """3D CNN with 2-channel input (baseline + followup stacked)."""

    def __init__(self, in_channels=2, base_channels=32, embedding_dim=256,
                 num_classes=3, dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            ConvBlock3D(base_channels * 2, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.2),
            ConvBlock3D(base_channels * 4, base_channels * 8),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, baseline, followup, time_delta=None):
        x = torch.cat([baseline, followup], dim=1)
        return self.classifier(self.encoder(x))


class DifferenceNet3D(nn.Module):
    """3D CNN with 3-channel input (baseline + followup + difference)."""

    def __init__(self, in_channels=3, base_channels=32, embedding_dim=256,
                 num_classes=3, dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.1),
            ConvBlock3D(base_channels * 2, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.2),
            ConvBlock3D(base_channels * 4, base_channels * 8),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, baseline, followup, time_delta=None):
        diff = followup - baseline
        x = torch.cat([baseline, followup, diff], dim=1)
        return self.classifier(self.encoder(x))


# =============================================================================
# MedicalNet-based Models (Pretrained)
# =============================================================================

def load_medicalnet_weights(encoder, pretrained_path):
    """Load MedicalNet pretrained weights into encoder."""
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print("Training from scratch...")
        return 0

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load matching weights
    model_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    if pretrained_dict:
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained layers from MedicalNet")

    return len(pretrained_dict)


class LongitudinalMedicalNet(nn.Module):
    """
    MedicalNet encoder + LSTM temporal modeling.

    Architecture:
    1. Shared MedicalNet encoder extracts features from both scans
    2. LSTM processes the temporal sequence [baseline, followup]
    3. Classifier combines LSTM output with difference features
    """

    def __init__(self, in_channels=1, resnet_depth=10, hidden_dim=256,
                 num_classes=3, dropout=0.5, pretrained_path=None):
        super().__init__()

        self.encoder = MedicalNetEncoder(in_channels, resnet_depth)

        if pretrained_path:
            load_medicalnet_weights(self.encoder, pretrained_path)

        self.lstm = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        classifier_input = hidden_dim + self.encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, baseline, followup, time_delta=None):
        feat_baseline = self.encoder(baseline)
        feat_followup = self.encoder(followup)

        diff = feat_followup - feat_baseline
        sequence = torch.stack([feat_baseline, feat_followup], dim=1)

        lstm_out, _ = self.lstm(sequence)
        lstm_final = lstm_out[:, -1, :]

        combined = torch.cat([lstm_final, diff], dim=1)
        return self.classifier(combined)

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        print(f"Encoder weights {'frozen' if freeze else 'unfrozen'}")


class LongitudinalAttentionNet(nn.Module):
    """
    MedicalNet encoder + cross-attention temporal modeling.

    Architecture:
    1. Shared MedicalNet encoder extracts features from both scans
    2. Cross-attention: followup attends to baseline
    3. Classifier combines all features
    """

    def __init__(self, in_channels=1, resnet_depth=10, num_classes=3,
                 dropout=0.5, pretrained_path=None):
        super().__init__()

        self.encoder = MedicalNetEncoder(in_channels, resnet_depth)

        if pretrained_path:
            load_medicalnet_weights(self.encoder, pretrained_path)

        feature_dim = self.encoder.feature_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        classifier_input = feature_dim * 4  # baseline + followup + attended + diff

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, baseline, followup, time_delta=None):
        feat_baseline = self.encoder(baseline)
        feat_followup = self.encoder(followup)

        q = feat_followup.unsqueeze(1)
        k = feat_baseline.unsqueeze(1)
        v = feat_baseline.unsqueeze(1)

        attended, _ = self.attention(q, k, v)
        attended = attended.squeeze(1)

        diff = feat_followup - feat_baseline
        combined = torch.cat([feat_baseline, feat_followup, attended, diff], dim=1)

        return self.classifier(combined)

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        print(f"Encoder weights {'frozen' if freeze else 'unfrozen'}")


# =============================================================================
# Loss Functions
# =============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network pre-training."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


# =============================================================================
# Utilities
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing models...\n")

    batch_size = 2
    baseline = torch.randn(batch_size, 1, 64, 64, 64)
    followup = torch.randn(batch_size, 1, 64, 64, 64)
    time_delta = torch.tensor([1.5, 2.0])

    models = [
        ("SiameseNetwork", SiameseNetwork(num_classes=3)),
        ("Longitudinal3DCNN", Longitudinal3DCNN()),
        ("DifferenceNet3D", DifferenceNet3D()),
        ("LongitudinalMedicalNet", LongitudinalMedicalNet()),
        ("LongitudinalAttentionNet", LongitudinalAttentionNet()),
    ]

    for name, model in models:
        logits = model(baseline, followup, time_delta)
        print(f"{name}: {count_parameters(model):,} params, output shape: {logits.shape}")
