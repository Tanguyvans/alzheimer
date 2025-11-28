#!/usr/bin/env python3
"""
MedTransformer: Vision Transformer for Alzheimer's Classification

Based on: "MedTransformer: Accurate Alzheimer's Disease Diagnosis for 3D MRI
Images through 2D Vision Transformers" (arXiv 2024)

Key innovations:
1. Extracts 2D slices from 3 views (axial, coronal, sagittal)
2. Shared ViT encoder across views
3. Cross-attention mechanism between views
4. Dimension-specific encoders for each view

Reference: https://arxiv.org/html/2401.06349v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try to import timm for pretrained ViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttention(nn.Module):
    """Cross-attention module for fusing features from different views"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: (batch, seq_q, embed_dim)
        # key, value: (batch, seq_kv, embed_dim)
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))


class SliceEncoder(nn.Module):
    """Encode a single 2D slice using ViT or CNN backbone"""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embed_dim: int = 768
    ):
        super().__init__()

        if TIMM_AVAILABLE and 'vit' in model_name:
            # Use pretrained ViT from timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                in_chans=3
            )
            self.embed_dim = self.backbone.embed_dim
        else:
            # Fallback to CNN encoder
            self.backbone = self._build_cnn_encoder(embed_dim)
            self.embed_dim = embed_dim

    def _build_cnn_encoder(self, embed_dim):
        """Simple CNN encoder as fallback"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        # x: (batch, C, H, W)
        return self.backbone(x)


class DimensionEncoder(nn.Module):
    """Encode sequence of slices from one dimension (axial/coronal/sagittal)"""

    def __init__(
        self,
        slice_encoder: SliceEncoder,
        embed_dim: int = 768,
        num_slices: int = 20,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.slice_encoder = slice_encoder
        self.embed_dim = embed_dim

        # Positional encoding for slice sequence
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_slices, dropout=dropout)

        # Self-attention layers for intra-dimension attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for aggregating slice features
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, slices):
        """
        Args:
            slices: (batch, num_slices, C, H, W)
        Returns:
            features: (batch, embed_dim)
            slice_features: (batch, num_slices, embed_dim)
        """
        batch_size, num_slices = slices.shape[:2]

        # Encode each slice
        slice_features = []
        for i in range(num_slices):
            feat = self.slice_encoder(slices[:, i])  # (batch, embed_dim)
            slice_features.append(feat)

        slice_features = torch.stack(slice_features, dim=1)  # (batch, num_slices, embed_dim)

        # Add positional encoding
        slice_features = self.pos_encoding(slice_features)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, slice_features], dim=1)  # (batch, 1+num_slices, embed_dim)

        # Self-attention
        features = self.transformer(features)

        # Return CLS token as aggregated feature and slice features
        return features[:, 0], features[:, 1:]


class MedTransformer(nn.Module):
    """
    MedTransformer: Multi-view Vision Transformer for Alzheimer's Classification

    Architecture:
    1. Extract slices from 3 views (axial, coronal, sagittal)
    2. Shared ViT encoder for all slices
    3. Dimension-specific transformer encoders
    4. Cross-attention fusion between views
    5. Classification head
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_slices: int = 20,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_slices = num_slices
        self.use_cross_attention = use_cross_attention

        # Shared slice encoder
        self.slice_encoder = SliceEncoder(
            model_name=backbone,
            pretrained=pretrained,
            embed_dim=embed_dim
        )
        self.embed_dim = self.slice_encoder.embed_dim

        # Dimension-specific encoders
        self.axial_encoder = DimensionEncoder(
            self.slice_encoder, self.embed_dim, num_slices, num_heads, num_layers, dropout
        )
        self.coronal_encoder = DimensionEncoder(
            self.slice_encoder, self.embed_dim, num_slices, num_heads, num_layers, dropout
        )
        self.sagittal_encoder = DimensionEncoder(
            self.slice_encoder, self.embed_dim, num_slices, num_heads, num_layers, dropout
        )

        # Cross-attention between views
        if use_cross_attention:
            self.cross_attn_ax_co = CrossAttention(self.embed_dim, num_heads, dropout)
            self.cross_attn_ax_sa = CrossAttention(self.embed_dim, num_heads, dropout)
            self.cross_attn_co_ax = CrossAttention(self.embed_dim, num_heads, dropout)
            self.cross_attn_co_sa = CrossAttention(self.embed_dim, num_heads, dropout)
            self.cross_attn_sa_ax = CrossAttention(self.embed_dim, num_heads, dropout)
            self.cross_attn_sa_co = CrossAttention(self.embed_dim, num_heads, dropout)

        # Fusion and classification
        fusion_dim = self.embed_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, num_classes)
        )

    def forward(self, axial_slices, coronal_slices, sagittal_slices):
        """
        Args:
            axial_slices: (batch, num_slices, C, H, W)
            coronal_slices: (batch, num_slices, C, H, W)
            sagittal_slices: (batch, num_slices, C, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        # Encode each view
        ax_feat, ax_slices = self.axial_encoder(axial_slices)
        co_feat, co_slices = self.coronal_encoder(coronal_slices)
        sa_feat, sa_slices = self.sagittal_encoder(sagittal_slices)

        # Cross-attention fusion
        if self.use_cross_attention:
            # Axial attends to coronal and sagittal
            ax_feat_co = self.cross_attn_ax_co(ax_feat.unsqueeze(1), co_slices, co_slices).squeeze(1)
            ax_feat_sa = self.cross_attn_ax_sa(ax_feat.unsqueeze(1), sa_slices, sa_slices).squeeze(1)
            ax_feat = ax_feat + ax_feat_co + ax_feat_sa

            # Coronal attends to axial and sagittal
            co_feat_ax = self.cross_attn_co_ax(co_feat.unsqueeze(1), ax_slices, ax_slices).squeeze(1)
            co_feat_sa = self.cross_attn_co_sa(co_feat.unsqueeze(1), sa_slices, sa_slices).squeeze(1)
            co_feat = co_feat + co_feat_ax + co_feat_sa

            # Sagittal attends to axial and coronal
            sa_feat_ax = self.cross_attn_sa_ax(sa_feat.unsqueeze(1), ax_slices, ax_slices).squeeze(1)
            sa_feat_co = self.cross_attn_sa_co(sa_feat.unsqueeze(1), co_slices, co_slices).squeeze(1)
            sa_feat = sa_feat + sa_feat_ax + sa_feat_co

        # Concatenate and fuse
        fused = torch.cat([ax_feat, co_feat, sa_feat], dim=-1)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)

        return logits


class MedTransformerLite(nn.Module):
    """
    Lightweight version using single-view or fewer slices
    Good for faster experimentation
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_slices: int = 5,
        embed_dim: int = 384,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        # Single shared encoder
        self.slice_encoder = SliceEncoder(backbone, pretrained, embed_dim)
        self.embed_dim = self.slice_encoder.embed_dim

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.embed_dim, max_len=num_slices * 3)

        # Single transformer for all views
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # View embeddings to distinguish axial/coronal/sagittal
        self.view_embedding = nn.Embedding(3, self.embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, num_classes)
        )

    def forward(self, axial_slices, coronal_slices, sagittal_slices):
        batch_size = axial_slices.shape[0]
        num_slices = axial_slices.shape[1]

        all_features = []

        # Encode all slices from all views
        for view_idx, slices in enumerate([axial_slices, coronal_slices, sagittal_slices]):
            for i in range(num_slices):
                feat = self.slice_encoder(slices[:, i])  # (batch, embed_dim)
                # Add view embedding
                feat = feat + self.view_embedding.weight[view_idx]
                all_features.append(feat)

        # Stack: (batch, num_slices*3, embed_dim)
        features = torch.stack(all_features, dim=1)

        # Add positional encoding
        features = self.pos_encoding(features)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)

        # Transformer
        features = self.transformer(features)

        # Classify using CLS token
        return self.classifier(features[:, 0])


def get_model(config: dict) -> nn.Module:
    """Factory function to create model from config"""

    model_name = config['model'].get('name', 'MedTransformer')

    if model_name == 'MedTransformer':
        model = MedTransformer(
            num_classes=config['model']['num_classes'],
            num_slices=config['model']['num_slices'],
            embed_dim=config['model'].get('embed_dim', 768),
            num_heads=config['model'].get('num_heads', 8),
            num_layers=config['model'].get('num_layers', 2),
            dropout=config['model'].get('dropout', 0.1),
            backbone=config['model'].get('backbone', 'vit_base_patch16_224'),
            pretrained=config['model'].get('pretrained', True),
            use_cross_attention=config['model'].get('use_cross_attention', True)
        )
    elif model_name == 'MedTransformerLite':
        model = MedTransformerLite(
            num_classes=config['model']['num_classes'],
            num_slices=config['model']['num_slices'],
            embed_dim=config['model'].get('embed_dim', 384),
            backbone=config['model'].get('backbone', 'vit_small_patch16_224'),
            pretrained=config['model'].get('pretrained', True),
            dropout=config['model'].get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


if __name__ == "__main__":
    # Test model
    print("Testing MedTransformer...")

    model = MedTransformer(
        num_classes=4,
        num_slices=5,
        embed_dim=384,
        backbone="vit_small_patch16_224" if TIMM_AVAILABLE else "cnn",
        pretrained=True
    )

    # Dummy input: 3 views, each with 5 slices
    batch_size = 2
    num_slices = 5
    axial = torch.randn(batch_size, num_slices, 3, 224, 224)
    coronal = torch.randn(batch_size, num_slices, 3, 224, 224)
    sagittal = torch.randn(batch_size, num_slices, 3, 224, 224)

    output = model(axial, coronal, sagittal)
    print(f"Input shapes: axial={axial.shape}, coronal={coronal.shape}, sagittal={sagittal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test lite version
    print("\nTesting MedTransformerLite...")
    model_lite = MedTransformerLite(num_classes=4, num_slices=5)
    output_lite = model_lite(axial, coronal, sagittal)
    print(f"Output shape: {output_lite.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
