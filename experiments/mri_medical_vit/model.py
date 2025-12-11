#!/usr/bin/env python3
"""
3D Vision Transformer (ViT) for Medical Image Classification

Based on:
- MONAI's ViT implementation
- "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- "Vision Transformer for 3D Medical Image Registration" (Chen et al.)

Supports:
- Various ViT configurations (Base, Small, Tiny)
- Loading pretrained 2D ImageNet weights (adapted to 3D)
- Custom patch sizes for medical imaging
"""

import torch
import torch.nn as nn
from monai.networks.nets import ViT
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


# ViT configuration presets
VIT_CONFIGS = {
    'vit_tiny': {
        'hidden_size': 192,
        'mlp_dim': 768,
        'num_layers': 12,
        'num_heads': 3,
    },
    'vit_small': {
        'hidden_size': 384,
        'mlp_dim': 1536,
        'num_layers': 12,
        'num_heads': 6,
    },
    'vit_base': {
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_layers': 12,
        'num_heads': 12,
    },
    'vit_large': {
        'hidden_size': 1024,
        'mlp_dim': 4096,
        'num_layers': 24,
        'num_heads': 16,
    },
}


def get_vit_model(
    architecture: str = 'vit_base',
    num_classes: int = 3,
    in_channels: int = 1,
    image_size: int = 96,
    patch_size: int = 16,
    dropout: float = 0.1,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    """
    Create a 3D Vision Transformer model.

    Args:
        architecture: One of 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'
        num_classes: Number of output classes
        in_channels: Number of input channels
        image_size: Input image size (assumes cubic)
        patch_size: Size of each patch (assumes cubic)
        dropout: Dropout rate
        pretrained_path: Path to pretrained weights

    Returns:
        ViT model
    """
    if architecture not in VIT_CONFIGS:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(VIT_CONFIGS.keys())}")

    config = VIT_CONFIGS[architecture]

    model = ViT(
        in_channels=in_channels,
        img_size=(image_size, image_size, image_size),
        patch_size=(patch_size, patch_size, patch_size),
        hidden_size=config['hidden_size'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        proj_type='conv',
        pos_embed_type='learnable',
        classification=True,
        num_classes=num_classes,
        dropout_rate=dropout,
        spatial_dims=3,
    )

    # Calculate number of patches
    num_patches = (image_size // patch_size) ** 3
    logger.info(f"Created {architecture}")
    logger.info(f"  Image size: {image_size}x{image_size}x{image_size}")
    logger.info(f"  Patch size: {patch_size}x{patch_size}x{patch_size}")
    logger.info(f"  Number of patches: {num_patches}")
    logger.info(f"  Hidden size: {config['hidden_size']}")
    logger.info(f"  Layers: {config['num_layers']}, Heads: {config['num_heads']}")

    # Load pretrained weights if provided
    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path, num_classes)

    return model


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    num_classes: int
) -> nn.Module:
    """
    Load pretrained weights into ViT model.

    Supports:
    - Full 3D ViT checkpoints
    - Partial loading (skip mismatched layers)

    Args:
        model: ViT model
        pretrained_path: Path to checkpoint
        num_classes: Number of output classes

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading pretrained weights from {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint

    # Remove 'module.' prefix if present
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()
    loaded_count = 0
    skipped_count = 0

    for key, value in pretrained_dict.items():
        # Skip classification head
        if 'classification_head' in key or 'head' in key:
            logger.debug(f"  Skipping {key} (classification head)")
            skipped_count += 1
            continue

        if key in model_dict:
            if value.shape == model_dict[key].shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                logger.warning(f"  Shape mismatch for {key}: {value.shape} vs {model_dict[key].shape}")
                skipped_count += 1
        else:
            skipped_count += 1

    model.load_state_dict(model_dict)
    logger.info(f"Loaded {loaded_count} layers, skipped {skipped_count}")

    return model


class ViT3DClassifier(nn.Module):
    """
    Vision Transformer wrapper with additional features:
    - Multiple classification heads
    - Feature extraction mode
    - Attention visualization
    """

    def __init__(
        self,
        architecture: str = 'vit_base',
        num_classes: int = 3,
        in_channels: int = 1,
        image_size: int = 96,
        patch_size: int = 16,
        dropout: float = 0.1,
        pretrained_path: Optional[str] = None,
        pool: str = 'cls',  # 'cls' or 'mean'
    ):
        super().__init__()

        self.pool = pool
        config = VIT_CONFIGS[architecture]

        # Create base ViT (without built-in classifier)
        self.vit = ViT(
            in_channels=in_channels,
            img_size=(image_size, image_size, image_size),
            patch_size=(patch_size, patch_size, patch_size),
            hidden_size=config['hidden_size'],
            mlp_dim=config['mlp_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            proj_type='conv',
            pos_embed_type='learnable',
            classification=False,  # We'll add our own
            dropout_rate=dropout,
            spatial_dims=3,
            save_attn=True,  # Save attention maps
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config['hidden_size']),
            nn.Dropout(p=dropout),
            nn.Linear(config['hidden_size'], num_classes)
        )

        # Load pretrained weights
        if pretrained_path:
            self._load_backbone_weights(pretrained_path)

    def _load_backbone_weights(self, pretrained_path: str):
        """Load pretrained weights for backbone only"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        # Filter to vit backbone only
        vit_dict = self.vit.state_dict()
        loaded = 0

        for key, value in pretrained_dict.items():
            # Remove prefix if needed
            clean_key = key.replace('module.', '').replace('vit.', '')

            if clean_key in vit_dict and value.shape == vit_dict[clean_key].shape:
                vit_dict[clean_key] = value
                loaded += 1

        self.vit.load_state_dict(vit_dict)
        logger.info(f"Loaded {loaded} layers into ViT backbone")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        # Get ViT features
        features, hidden_states = self.vit(x)

        # Pool features
        if self.pool == 'cls':
            # Use CLS token (first token)
            pooled = features[:, 0]
        else:
            # Mean pooling over all patches
            pooled = features.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features, _ = self.vit(x)
        if self.pool == 'cls':
            return features[:, 0]
        return features.mean(dim=1)

    def get_attention_maps(self) -> list:
        """Get attention maps from all layers"""
        # MONAI ViT stores attention in self.vit.blocks
        attention_maps = []
        for block in self.vit.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'att_mat'):
                attention_maps.append(block.attn.att_mat)
        return attention_maps


class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer with CNN backbone for feature extraction.

    Uses a 3D CNN to extract initial features and reduce spatial resolution
    before feeding to the transformer. This makes the transformer feasible
    for 3D medical images.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        image_size: int = 96,
        vit_config: str = 'vit_small',
        dropout: float = 0.1,
        patch_size: int = 8,  # Further downsampling with strided conv
    ):
        super().__init__()

        config = VIT_CONFIGS[vit_config]

        # CNN backbone: 96 -> 48 -> 24 -> 12 (with additional stride)
        self.cnn_backbone = nn.Sequential(
            # Stage 1: 96 -> 48
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # 48 -> 24

            # Stage 2: 24 -> 24
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Stage 3: 24 -> 12 (additional downsampling)
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # Stage 4: 12 -> 6 (more downsampling for manageable tokens)
            nn.Conv3d(256, config['hidden_size'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(config['hidden_size']),
            nn.ReLU(inplace=True),
        )

        # Final spatial size: 96 / 16 = 6
        final_size = image_size // 16
        num_patches = final_size ** 3  # 6^3 = 216 tokens

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config['hidden_size'])
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))

        # ViT encoder blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['mlp_dim'],
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=config['num_layers']
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config['hidden_size']),
            nn.Dropout(p=dropout),
            nn.Linear(config['hidden_size'], num_classes)
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        logger.info(f"HybridViT created:")
        logger.info(f"  Input: {image_size}x{image_size}x{image_size}")
        logger.info(f"  CNN output: {final_size}x{final_size}x{final_size}")
        logger.info(f"  Tokens: {num_patches} + 1 (CLS)")
        logger.info(f"  Hidden size: {config['hidden_size']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN feature extraction with spatial reduction
        x = self.cnn_backbone(x)

        # Reshape to sequence: (B, C, D, H, W) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Classify using CLS token
        x = self.classifier(x[:, 0])

        return x


if __name__ == '__main__':
    print("Testing 3D ViT models...")

    # Test basic ViT
    for arch in ['vit_tiny', 'vit_small', 'vit_base']:
        model = get_vit_model(
            architecture=arch,
            num_classes=3,
            image_size=96,
            patch_size=16
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{arch}:")
        print(f"  Parameters: {total_params:,}")

        x = torch.randn(1, 1, 96, 96, 96)
        out, _ = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Test wrapper class
    print("\n\nTesting ViT3DClassifier wrapper...")
    model = ViT3DClassifier(
        architecture='vit_small',
        num_classes=3,
        image_size=96,
        patch_size=16
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    x = torch.randn(1, 1, 96, 96, 96)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Test Hybrid ViT
    print("\n\nTesting HybridViT...")
    model = HybridViT(
        num_classes=3,
        image_size=96,
        vit_config='vit_small'
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    x = torch.randn(1, 1, 96, 96, 96)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
