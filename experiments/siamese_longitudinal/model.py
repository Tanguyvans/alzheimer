#!/usr/bin/env python3
"""
Siamese Network for Longitudinal MRI Analysis.

Compares baseline and follow-up MRI scans to predict conversion probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder3D(nn.Module):
    """
    3D CNN encoder for MRI feature extraction.
    Lightweight design for efficiency.
    """

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256):
        super().__init__()

        # Encoder blocks with progressive downsampling
        self.encoder = nn.Sequential(
            # Block 1: (1, D, H, W) -> (32, D/2, H/2, W/2)
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.MaxPool3d(2),

            # Block 2: (32, D/2, H/2, W/2) -> (64, D/4, H/4, W/4)
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
            nn.MaxPool3d(2),

            # Block 3: (64, D/4, H/4, W/4) -> (128, D/8, H/8, W/8)
            ConvBlock3D(base_channels * 2, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
            nn.MaxPool3d(2),

            # Block 4: (128, D/8, H/8, W/8) -> (256, D/16, H/16, W/16)
            ConvBlock3D(base_channels * 4, base_channels * 8),
            ConvBlock3D(base_channels * 8, base_channels * 8),
            nn.AdaptiveAvgPool3d(1),
        )

        # Projection head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        features = self.encoder(x)
        embedding = self.fc(features)
        return embedding


class SiameseNetwork(nn.Module):
    """
    Siamese Network for comparing baseline and follow-up MRI.

    Architecture:
    - Shared encoder extracts features from both scans
    - Difference/similarity features computed
    - Classifier predicts conversion probability

    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        base_channels: Base number of channels in encoder
        embedding_dim: Dimension of embedding space
        num_classes: Number of output classes (2 for binary conversion)
    """

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256, num_classes=2):
        super().__init__()

        # Shared encoder
        self.encoder = Encoder3D(in_channels, base_channels, embedding_dim)

        # Classifier takes concatenated features + difference + time
        # [baseline_emb, followup_emb, diff, time_delta] -> prediction
        classifier_input_dim = embedding_dim * 3 + 1  # emb1, emb2, diff, time

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward_one(self, x):
        """Forward pass for one image"""
        return self.encoder(x)

    def forward(self, baseline, followup, time_delta=None):
        """
        Forward pass for Siamese network.

        Args:
            baseline: Baseline MRI tensor (B, 1, D, H, W)
            followup: Follow-up MRI tensor (B, 1, D, H, W)
            time_delta: Time between scans in years (B, 1)

        Returns:
            logits: Classification logits (B, num_classes)
            embeddings: Tuple of (baseline_emb, followup_emb) for visualization
        """
        # Get embeddings
        emb_baseline = self.forward_one(baseline)
        emb_followup = self.forward_one(followup)

        # Compute difference features
        diff = emb_followup - emb_baseline

        # Handle time delta
        if time_delta is None:
            time_delta = torch.ones(baseline.size(0), 1, device=baseline.device)
        elif time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(1)

        # Concatenate all features
        combined = torch.cat([emb_baseline, emb_followup, diff, time_delta], dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits, (emb_baseline, emb_followup)


class WeightedSiameseNetwork(SiameseNetwork):
    """
    Weighted Siamese Network with attention to temporal changes.

    Adds attention mechanism to weight the importance of different
    feature dimensions based on the time delta.
    """

    def __init__(self, in_channels=1, base_channels=32, embedding_dim=256, num_classes=2):
        super().__init__(in_channels, base_channels, embedding_dim, num_classes)

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, baseline, followup, time_delta=None):
        # Get embeddings
        emb_baseline = self.forward_one(baseline)
        emb_followup = self.forward_one(followup)

        # Handle time delta
        if time_delta is None:
            time_delta = torch.ones(baseline.size(0), 1, device=baseline.device)
        elif time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(1)

        # Compute temporal attention weights
        attention = self.temporal_attention(time_delta)

        # Weighted difference
        diff = (emb_followup - emb_baseline) * attention

        # Concatenate
        combined = torch.cat([emb_baseline, emb_followup, diff, time_delta], dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits, (emb_baseline, emb_followup)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese network pre-training.

    Pulls together embeddings of same-class pairs and pushes apart different-class pairs.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1: First embeddings (B, D)
            emb2: Second embeddings (B, D)
            label: 1 if same class (should be similar), 0 if different (should be dissimilar)
        """
        distance = F.pairwise_distance(emb1, emb2)

        # Same class: minimize distance
        # Different class: maximize distance (up to margin)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = SiameseNetwork(in_channels=1, base_channels=32, embedding_dim=256, num_classes=2)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 2
    baseline = torch.randn(batch_size, 1, 96, 112, 96)  # Example MRI size
    followup = torch.randn(batch_size, 1, 96, 112, 96)
    time_delta = torch.tensor([1.5, 2.0])

    logits, (emb1, emb2) = model(baseline, followup, time_delta)
    print(f"Logits shape: {logits.shape}")
    print(f"Embedding shape: {emb1.shape}")
