#!/usr/bin/env python3
"""
3D Vision Transformer (ViT) for Alzheimer's Disease Classification

Based on: "Training ViT with Limited Data for Alzheimer's Disease Classification" (MICCAI 2024)
GitHub: https://github.com/qasymjomart/ViT_recipe_for_AD

This implementation matches the paper's custom ViT3D architecture:
- 128x128x128 input with 16x16x16 patches -> 512 patches
- cls_token prepended to patch embeddings
- Standard transformer blocks (no cross-attention like MONAI)
- When loading MAE weights, only transformer blocks are loaded (pos_embed, patch_embed reinitialized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


# ViT configurations matching the paper
VIT_CONFIGS = {
    'vit_base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'patch_size': 16,
    },
    'vit_small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'patch_size': 16,
    },
    'vit_tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'patch_size': 16,
    },
}


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using Conv3D"""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block with GELU activation"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    """Transformer block with attention + MLP"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT3DClassifier(nn.Module):
    """
    3D Vision Transformer matching the paper's architecture.

    Key differences from MONAI's ViT:
    - Uses cls_token (prepended to patch embeddings)
    - Standard transformer blocks (no cross-attention)
    - Learnable position embeddings
    """

    def __init__(
        self,
        architecture: str = 'vit_base',
        num_classes: int = 2,
        in_channels: int = 1,
        image_size: int = 128,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.1,
        classifier_dropout: float = 0.5,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        config = VIT_CONFIGS.get(architecture, VIT_CONFIGS['vit_base'])
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config['mlp_ratio']
        self.patch_size = config['patch_size']
        self.image_size = image_size
        self.num_classes = num_classes

        # Calculate number of patches
        self.num_patches = (image_size // self.patch_size) ** 3

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=image_size,
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=self.embed_dim,
        )

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                drop_path=dpr[i],
            )
            for i in range(self.depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.embed_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"Created ViT3DClassifier with {architecture}")
        logger.info(f"  Image size: {image_size}x{image_size}x{image_size}")
        logger.info(f"  Patch size: {self.patch_size}")
        logger.info(f"  Num patches: {self.num_patches}")
        logger.info(f"  Embed dim: {self.embed_dim}, Depth: {self.depth}, Heads: {self.num_heads}")

        # Load pretrained weights
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained(pretrained_path)

    def freeze_backbone(self):
        """Freeze all layers except classification head"""
        for name, param in self.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        logger.info("Backbone frozen - only training classification head")

    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen - training all layers")

    def _init_weights(self):
        """Initialize weights"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other layers
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, pretrained_path: str):
        """
        Load MAE pre-trained weights.

        Since we use same image size (128³) as pretraining, we can load:
        - All transformer blocks
        - cls_token (same shape)
        - pos_embed (if shapes match after removing MAE's extra cls_token)
        - patch_embed (if input channels match)

        Only skip: head (different num_classes) and decoder (not used)
        """
        logger.info(f"Loading MAE pre-trained weights from {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Extract state dict
        if 'net' in checkpoint:
            pretrained_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        # Clean keys (remove 'module.' prefix from DataParallel)
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

        # Get model state dict for comparison
        model_dict = self.state_dict()

        # Process each key
        loadable_dict = {}
        skipped_keys = []

        for k, v in pretrained_dict.items():
            # Skip decoder layers and mask token (MAE-specific)
            if 'decoder' in k or 'mask_token' in k:
                skipped_keys.append(k)
                continue

            # Skip classification head (different num_classes)
            if k in ['head.weight', 'head.bias']:
                skipped_keys.append(k)
                continue

            # Handle pos_embed - MAE checkpoint has (1, 513, 768) = 512 patches + 1 cls_token
            if k == 'pos_embed':
                target_shape = model_dict[k].shape
                if v.shape == target_shape:
                    loadable_dict[k] = v
                    logger.info(f"  Loaded pos_embed directly (shapes match: {v.shape})")
                elif v.shape[1] == target_shape[1] + 1:
                    # Remove first token (cls_token position) if MAE has extra
                    v = v[:, 1:, :]
                    if v.shape[1] == target_shape[1] - 1:
                        # Our model expects cls_token in pos_embed, checkpoint doesn't have it there
                        # Skip and let it be randomly initialized
                        logger.info(f"  pos_embed shape mismatch after cls removal, reinitializing")
                        skipped_keys.append(k)
                    else:
                        loadable_dict[k] = v
                        logger.info(f"  Loaded pos_embed after removing cls position: {v.shape}")
                else:
                    logger.warning(f"  pos_embed shape mismatch: {v.shape} vs {target_shape}, reinitializing")
                    skipped_keys.append(k)
                continue

            # Try to load if key exists and shapes match
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    loadable_dict[k] = v
                else:
                    logger.warning(f"  Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
                    skipped_keys.append(k)
            else:
                logger.debug(f"  Key not in model: {k}")
                skipped_keys.append(k)

        # Load the weights
        missing, unexpected = self.load_state_dict(loadable_dict, strict=False)

        logger.info(f"Loaded {len(loadable_dict)} pretrained weights (including pos_embed, cls_token, patch_embed if matched)")
        logger.info(f"Missing keys: {len(missing)}")
        if missing:
            logger.info(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        if len(loadable_dict) < 50:
            logger.warning("Very few weights loaded! Check pretrained checkpoint format.")
            logger.info(f"Loaded keys: {list(loadable_dict.keys())[:10]}...")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding: (B, C, D, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend cls_token: (B, num_patches, embed_dim) -> (B, num_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Use cls_token output for classification
        cls_output = x[:, 0]

        return self.head(cls_output)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


# Convenience function for backward compatibility
def get_vit_model(
    architecture: str = 'vit_base',
    num_classes: int = 2,
    in_channels: int = 1,
    image_size: Tuple[int, int, int] = (128, 128, 128),
    pretrained_path: Optional[str] = None,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a ViT3D model"""
    return ViT3DClassifier(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size[0] if isinstance(image_size, tuple) else image_size,
        pretrained_path=pretrained_path,
        dropout=dropout,
    )


if __name__ == '__main__':
    # Test model creation
    print("Testing custom ViT3D models...")

    for arch in ['vit_tiny', 'vit_small', 'vit_base']:
        print(f"\n{arch}:")

        model = ViT3DClassifier(
            architecture=arch,
            num_classes=2,
            in_channels=1,
            image_size=128,  # Paper uses 128
            dropout=0.1,
            classifier_dropout=0.5,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 1, 128, 128, 128)
        out = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

        # Test feature extraction
        features = model.extract_features(x)
        print(f"  Features: {features.shape}")
