#!/usr/bin/env python3
"""
Multi-Modal Fusion Model with Modality Dropout Support

Key features:
- Handles missing modalities gracefully
- Optional modality availability embeddings
- Works with MRI-only, tabular-only, or both modalities

This module is fully self-contained with the ViT3DClassifier included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

# ============================================================================
# ViT3D Components (copied from mri_vit_ad for independence)
# Based on: "Training ViT with Limited Data for Alzheimer's Disease Classification" (MICCAI 2024)
# ============================================================================

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
    3D Vision Transformer matching the MICCAI 2024 paper's architecture.

    Key features:
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

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
        """Load MAE pre-trained weights."""
        logger.info(f"Loading MAE pre-trained weights from {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

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

            # Handle pos_embed
            if k == 'pos_embed':
                target_shape = model_dict[k].shape
                if v.shape == target_shape:
                    loadable_dict[k] = v
                    logger.info(f"  Loaded pos_embed directly (shapes match: {v.shape})")
                elif v.shape[1] == target_shape[1] + 1:
                    v = v[:, 1:, :]
                    if v.shape[1] == target_shape[1] - 1:
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

        logger.info(f"Loaded {len(loadable_dict)} pretrained weights")
        logger.info(f"Missing keys: {len(missing)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Prepend cls_token
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

# ============================================================================
# End of ViT3D Components
# ============================================================================


class TabularEncoder(nn.Module):
    """MLP encoder for tabular clinical features"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.encoder(x)


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities"""

    def __init__(self, img_dim: int, tab_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_img = nn.Linear(img_dim + tab_dim, img_dim)
        self.gate_tab = nn.Linear(img_dim + tab_dim, tab_dim)
        self.fc = nn.Linear(img_dim + tab_dim, hidden_dim)

    def forward(self, img_feat, tab_feat):
        combined = torch.cat([img_feat, tab_feat], dim=1)
        gate_i = torch.sigmoid(self.gate_img(combined))
        gate_t = torch.sigmoid(self.gate_tab(combined))
        gated = torch.cat([gate_i * img_feat, gate_t * tab_feat], dim=1)
        return self.fc(gated)


class ModalityDropoutFusion(nn.Module):
    """
    Multi-Modal Fusion Model with Modality Dropout Support

    Features:
    - Handles missing modalities at inference time
    - Modality availability embeddings help the model know what's available
    - Works with MRI-only, tabular-only, or both
    """

    def __init__(
        self,
        num_tabular_features: int,
        num_classes: int = 2,
        backbone_config: dict = None,
        tabular_config: dict = None,
        fusion_config: dict = None,
        pretrained_path: str = None,
        freeze_backbone: bool = False,
        use_modality_embeddings: bool = True,
    ):
        super().__init__()

        # Default configs
        backbone_config = backbone_config or {'type': 'vit', 'architecture': 'vit_base', 'image_size': 128}
        tabular_config = tabular_config or {'hidden_dims': [128, 64], 'dropout': 0.3}
        fusion_config = fusion_config or {'method': 'concat', 'hidden_dim': 256, 'dropout': 0.5}

        self.use_modality_embeddings = use_modality_embeddings
        self.num_tabular_features = num_tabular_features

        # MRI backbone (ViT)
        self.mri_backbone = ViT3DClassifier(
            architecture=backbone_config.get('architecture', 'vit_base'),
            num_classes=num_classes,
            in_channels=1,
            image_size=backbone_config.get('image_size', 128),
            pretrained_path=pretrained_path,
            dropout=0.1,
            classifier_dropout=0.0
        )
        self.mri_feature_dim = backbone_config.get('feature_dim', 768)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.mri_backbone.parameters():
                param.requires_grad = False
            logger.info("MRI backbone frozen")

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=num_tabular_features,
            hidden_dims=tabular_config.get('hidden_dims', [128, 64]),
            dropout=tabular_config.get('dropout', 0.3)
        )
        self.tabular_feature_dim = self.tabular_encoder.output_dim

        # Modality availability embeddings (learned)
        # These help the model understand which modalities are present
        if use_modality_embeddings:
            # 2 modalities x 2 states (available/missing) = 4 embeddings
            self.mri_available_embed = nn.Parameter(torch.zeros(1, 16))
            self.mri_missing_embed = nn.Parameter(torch.zeros(1, 16))
            self.tab_available_embed = nn.Parameter(torch.zeros(1, 16))
            self.tab_missing_embed = nn.Parameter(torch.zeros(1, 16))
            nn.init.normal_(self.mri_available_embed, std=0.02)
            nn.init.normal_(self.mri_missing_embed, std=0.02)
            nn.init.normal_(self.tab_available_embed, std=0.02)
            nn.init.normal_(self.tab_missing_embed, std=0.02)
            modality_embed_dim = 32  # 16 + 16
        else:
            modality_embed_dim = 0

        # Fusion layer
        fusion_method = fusion_config.get('method', 'concat')
        fusion_hidden = fusion_config.get('hidden_dim', 256)
        fusion_dropout = fusion_config.get('dropout', 0.5)

        total_feature_dim = self.mri_feature_dim + self.tabular_feature_dim + modality_embed_dim

        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(total_feature_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            )
        elif fusion_method == 'gated':
            # For gated fusion, we first project to same dim, then apply gating
            self.mri_proj = nn.Linear(self.mri_feature_dim, fusion_hidden // 2)
            self.tab_proj = nn.Linear(self.tabular_feature_dim + modality_embed_dim, fusion_hidden // 2)
            self.fusion = GatedFusion(fusion_hidden // 2, fusion_hidden // 2, fusion_hidden)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_method = fusion_method

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(128, num_classes)
        )

        logger.info(f"ModalityDropoutFusion initialized:")
        logger.info(f"  MRI features: {self.mri_feature_dim}")
        logger.info(f"  Tabular features: {num_tabular_features} -> {self.tabular_feature_dim}")
        logger.info(f"  Modality embeddings: {use_modality_embeddings}")
        logger.info(f"  Fusion method: {fusion_method}")
        logger.info(f"  Total fusion input: {total_feature_dim}")

    def extract_mri_features(self, x):
        """Extract features from MRI using ViT backbone"""
        x = self.mri_backbone.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.mri_backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.mri_backbone.pos_embed
        x = self.mri_backbone.pos_drop(x)

        for block in self.mri_backbone.blocks:
            x = block(x)

        x = self.mri_backbone.norm(x)

        # Return cls token features
        return x[:, 0]

    def forward(self, mri, tabular, modality_mask=None):
        """
        Forward pass with optional modality masking.

        Args:
            mri: (B, 1, D, H, W) - 3D MRI volume
            tabular: (B, num_features) - Tabular features
            modality_mask: (B, 2) - [mri_available, tabular_available]
                          If None, assumes both modalities are available

        Returns:
            logits: (B, num_classes)
        """
        B = mri.shape[0]
        device = mri.device

        # Default mask: both available
        if modality_mask is None:
            modality_mask = torch.ones(B, 2, device=device)

        mri_available = modality_mask[:, 0:1]  # (B, 1)
        tab_available = modality_mask[:, 1:2]  # (B, 1)

        # Extract MRI features (masked by availability)
        mri_features = self.extract_mri_features(mri)
        mri_features = mri_features * mri_available  # Zero out if not available

        # Encode tabular features (masked by availability)
        tab_features = self.tabular_encoder(tabular)
        tab_features = tab_features * tab_available  # Zero out if not available

        # Get modality embeddings
        if self.use_modality_embeddings:
            # Select appropriate embedding based on availability
            mri_embed = torch.where(
                mri_available.bool(),
                self.mri_available_embed.expand(B, -1),
                self.mri_missing_embed.expand(B, -1)
            )
            tab_embed = torch.where(
                tab_available.bool(),
                self.tab_available_embed.expand(B, -1),
                self.tab_missing_embed.expand(B, -1)
            )
            modality_embed = torch.cat([mri_embed, tab_embed], dim=1)
        else:
            modality_embed = None

        # Fusion
        if self.fusion_method == 'concat':
            if modality_embed is not None:
                combined = torch.cat([mri_features, tab_features, modality_embed], dim=1)
            else:
                combined = torch.cat([mri_features, tab_features], dim=1)
            fused = self.fusion(combined)
        elif self.fusion_method == 'gated':
            mri_proj = self.mri_proj(mri_features)
            if modality_embed is not None:
                tab_with_embed = torch.cat([tab_features, modality_embed], dim=1)
            else:
                tab_with_embed = tab_features
            tab_proj = self.tab_proj(tab_with_embed)
            fused = self.fusion(mri_proj, tab_proj)

        # Classification
        logits = self.classifier(fused)

        return logits

    def unfreeze_backbone(self, num_layers: int = -1):
        """Unfreeze MRI backbone for fine-tuning"""
        if num_layers == -1:
            for param in self.mri_backbone.parameters():
                param.requires_grad = True
            logger.info("All MRI backbone layers unfrozen")
        else:
            # Unfreeze last N blocks
            num_blocks = len(self.mri_backbone.blocks)
            for i, block in enumerate(self.mri_backbone.blocks):
                if i >= num_blocks - num_layers:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.mri_backbone.norm.parameters():
                param.requires_grad = True
            logger.info(f"Last {num_layers} backbone blocks unfrozen")


def build_model(config: dict, num_tabular_features: int) -> ModalityDropoutFusion:
    """Build model from config"""
    model_cfg = config['model']

    backbone_config = model_cfg.get('backbone', {
        'type': 'vit',
        'architecture': 'vit_base',
        'image_size': 128,
        'feature_dim': 768
    })

    model = ModalityDropoutFusion(
        num_tabular_features=num_tabular_features,
        num_classes=model_cfg.get('num_classes', 2),
        backbone_config=backbone_config,
        tabular_config=model_cfg.get('tabular', {}),
        fusion_config=model_cfg.get('fusion', {}),
        pretrained_path=backbone_config.get('pretrained_path'),
        freeze_backbone=backbone_config.get('freeze', False),
        use_modality_embeddings=model_cfg.get('use_modality_embeddings', True),
    )

    return model


if __name__ == '__main__':
    # Test model
    model = ModalityDropoutFusion(
        num_tabular_features=19,
        num_classes=2,
        backbone_config={'architecture': 'vit_base', 'image_size': 128, 'feature_dim': 768},
        tabular_config={'hidden_dims': [128, 64], 'dropout': 0.3},
        fusion_config={'method': 'concat', 'hidden_dim': 256, 'dropout': 0.5},
        pretrained_path=None,
        freeze_backbone=True,
        use_modality_embeddings=True
    )

    # Test forward with both modalities
    mri = torch.randn(2, 1, 128, 128, 128)
    tabular = torch.randn(2, 19)
    mask = torch.ones(2, 2)  # Both available

    with torch.no_grad():
        output = model(mri, tabular, mask)
        print(f"Output shape (both available): {output.shape}")

    # Test with MRI only
    mask_mri_only = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with torch.no_grad():
        output = model(mri, tabular, mask_mri_only)
        print(f"Output shape (MRI only): {output.shape}")

    # Test with tabular only
    mask_tab_only = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    with torch.no_grad():
        output = model(mri, tabular, mask_tab_only)
        print(f"Output shape (tabular only): {output.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
