#!/usr/bin/env python3
"""
3D Vision Transformer (ViT) for Alzheimer's Disease Classification

Based on: "Training ViT with Limited Data for Alzheimer's Disease Classification" (MICCAI 2024)
GitHub: https://github.com/qasymjomart/ViT_recipe_for_AD

Supports:
- MONAI's native ViT implementation
- Loading MAE pre-trained weights from the paper
- Custom classification head

Pre-trained weights:
- Download from paper's Google Drive (link in their GitHub)
- Place in pretrained/vit_b_75mask.pth
"""

import torch
import torch.nn as nn
from monai.networks.nets import ViT
from typing import Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ViT-B configuration (matches the paper)
VIT_CONFIGS = {
    'vit_base': {
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'patch_size': (16, 16, 16),
    },
    'vit_small': {
        'hidden_size': 384,
        'mlp_dim': 1536,
        'num_layers': 12,
        'num_heads': 6,
        'patch_size': (16, 16, 16),
    },
    'vit_tiny': {
        'hidden_size': 192,
        'mlp_dim': 768,
        'num_layers': 12,
        'num_heads': 3,
        'patch_size': (16, 16, 16),
    },
}


def get_vit_model(
    architecture: str = 'vit_base',
    num_classes: int = 2,
    in_channels: int = 1,
    image_size: Tuple[int, int, int] = (96, 96, 96),
    pretrained_path: Optional[str] = None,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Create a 3D ViT model with optional MAE pre-trained weights.

    Args:
        architecture: One of 'vit_base', 'vit_small', 'vit_tiny'
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale MRI)
        image_size: Input image size (D, H, W)
        pretrained_path: Path to MAE pre-trained weights (.pth file)
        dropout: Dropout rate

    Returns:
        ViT model
    """
    if architecture not in VIT_CONFIGS:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(VIT_CONFIGS.keys())}")

    config = VIT_CONFIGS[architecture]

    # Create MONAI ViT
    model = ViT(
        in_channels=in_channels,
        img_size=image_size,
        patch_size=config['patch_size'],
        hidden_size=config['hidden_size'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        classification=True,
        num_classes=num_classes,
        dropout_rate=dropout,
        spatial_dims=3,
    )

    logger.info(f"Created {architecture} (image_size={image_size}, patch_size={config['patch_size']})")
    logger.info(f"  Hidden: {config['hidden_size']}, MLP: {config['mlp_dim']}, Layers: {config['num_layers']}, Heads: {config['num_heads']}")

    # Load pretrained weights if specified
    if pretrained_path and Path(pretrained_path).exists():
        model = load_mae_weights(model, pretrained_path)

    return model


def load_mae_weights(
    model: nn.Module,
    pretrained_path: str,
) -> nn.Module:
    """
    Load MAE pre-trained weights into MONAI ViT.

    The MAE weights are from self-supervised pre-training on BraTS, IXI, OASIS3.

    Args:
        model: MONAI ViT model
        pretrained_path: Path to MAE checkpoint (.pth file)

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading MAE pre-trained weights from {pretrained_path}")

    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'net' in checkpoint:
        pretrained_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint

    # Remove 'module.' prefix (from DataParallel training)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Get model state dict
    model_dict = model.state_dict()

    # Try to match keys
    loaded_count = 0
    skipped_count = 0
    mismatched_keys = []

    for key, value in pretrained_dict.items():
        # Skip classification head (we train from scratch)
        if 'classification_head' in key or 'fc' in key or 'head' in key:
            logger.debug(f"  Skipping {key} (classification layer)")
            skipped_count += 1
            continue

        # Skip decoder layers (MAE decoder not used for classification)
        if 'decoder' in key:
            logger.debug(f"  Skipping {key} (decoder layer)")
            skipped_count += 1
            continue

        # Check if key exists in model
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                model_dict[key] = value
                loaded_count += 1
            else:
                mismatched_keys.append((key, value.shape, model_dict[key].shape))
                skipped_count += 1
        else:
            # Try common key mappings
            mapped_key = map_key(key)
            if mapped_key and mapped_key in model_dict:
                if value.shape == model_dict[mapped_key].shape:
                    model_dict[mapped_key] = value
                    loaded_count += 1
                else:
                    mismatched_keys.append((mapped_key, value.shape, model_dict[mapped_key].shape))
                    skipped_count += 1
            else:
                logger.debug(f"  Key not found: {key}")
                skipped_count += 1

    # Load updated state dict
    model.load_state_dict(model_dict, strict=False)

    logger.info(f"Loaded {loaded_count} layers, skipped {skipped_count}")
    if mismatched_keys:
        logger.warning(f"Shape mismatches: {len(mismatched_keys)}")
        for key, pretrained_shape, model_shape in mismatched_keys[:5]:
            logger.warning(f"  {key}: {pretrained_shape} vs {model_shape}")

    return model


def map_key(key: str) -> Optional[str]:
    """
    Map pretrained weight keys to MONAI ViT keys.

    Args:
        key: Key from pretrained checkpoint

    Returns:
        Mapped key for MONAI model, or None if no mapping
    """
    # Common mappings between different ViT implementations
    mappings = {
        'patch_embed': 'patch_embedding',
        'pos_embed': 'pos_embedding',
        'norm': 'ln',
        'mlp.fc1': 'mlp.linear1',
        'mlp.fc2': 'mlp.linear2',
        'attn.qkv': 'attn.qkv',
        'attn.proj': 'attn.out_proj',
    }

    for old, new in mappings.items():
        if old in key:
            return key.replace(old, new)

    return None


class ViT3DClassifier(nn.Module):
    """
    Wrapper around MONAI ViT with additional features:
    - Dropout before classifier
    - Feature extraction mode
    - Flexible classification head
    """

    def __init__(
        self,
        architecture: str = 'vit_base',
        num_classes: int = 2,
        in_channels: int = 1,
        image_size: Tuple[int, int, int] = (96, 96, 96),
        pretrained_path: Optional[str] = None,
        dropout: float = 0.1,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Get config
        config = VIT_CONFIGS.get(architecture, VIT_CONFIGS['vit_base'])
        hidden_size = config['hidden_size']

        # Create base ViT (without built-in classification)
        self.vit = ViT(
            in_channels=in_channels,
            img_size=image_size,
            patch_size=config['patch_size'],
            hidden_size=hidden_size,
            mlp_dim=config['mlp_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            classification=False,  # We add our own head
            dropout_rate=dropout,
            spatial_dims=3,
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_size, num_classes)
        )

        logger.info(f"Created ViT3DClassifier with {architecture}")
        logger.info(f"  Image size: {image_size}, Classes: {num_classes}")

        # Load pretrained weights
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights into ViT backbone with key mapping."""
        logger.info(f"Loading pre-trained weights from {pretrained_path}")

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

        # Clean keys
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

        # Map checkpoint keys to MONAI ViT keys
        mapped_dict = {}
        for k, v in pretrained_dict.items():
            # Skip decoder, mask token, and classification layers
            if 'decoder' in k or 'mask_token' in k or 'head' in k or 'fc' in k:
                continue

            new_key = k

            # Map position embedding
            if k == 'pos_embed':
                new_key = 'patch_embedding.position_embeddings'
                target_shape = self.vit.patch_embedding.position_embeddings.shape
                target_num_patches = target_shape[1]

                # Check if class token is included (num_patches + 1)
                # Common sizes: 513 = 512+1 (8x8x8 + cls), 217 = 216+1 (6x6x6 + cls)
                src_num_patches = v.shape[1]
                has_cls_token = False

                # Try to detect class token by checking if (n-1) is a perfect cube
                for offset in [0, 1]:
                    n = src_num_patches - offset
                    cube_root = round(n ** (1/3))
                    if cube_root ** 3 == n:
                        if offset == 1:
                            has_cls_token = True
                            v = v[:, 1:, :]  # Remove class token
                            logger.info(f"  Removed cls_token from pos_embed: {src_num_patches} -> {v.shape[1]}")
                        break

                # Interpolate if spatial dimensions don't match
                if v.shape[1] != target_num_patches:
                    import torch.nn.functional as F

                    src_size = round(v.shape[1] ** (1/3))
                    tgt_size = round(target_num_patches ** (1/3))

                    logger.info(f"  Interpolating pos_embed: {src_size}^3={v.shape[1]} -> {tgt_size}^3={target_num_patches}")

                    # Reshape to 3D grid: (1, n_patches, hidden) -> (1, hidden, d, h, w)
                    hidden_dim = v.shape[2]
                    v = v.permute(0, 2, 1).reshape(1, hidden_dim, src_size, src_size, src_size)

                    # Interpolate
                    v = F.interpolate(v, size=(tgt_size, tgt_size, tgt_size), mode='trilinear', align_corners=False)

                    # Reshape back: (1, hidden, d, h, w) -> (1, n_patches, hidden)
                    v = v.reshape(1, hidden_dim, -1).permute(0, 2, 1)
                    logger.info(f"  Interpolated pos_embed to {v.shape}")

            # Map patch embedding
            elif k == 'patch_embed.proj.weight':
                new_key = 'patch_embedding.patch_embeddings.weight'
            elif k == 'patch_embed.proj.bias':
                new_key = 'patch_embedding.patch_embeddings.bias'

            # Map transformer blocks
            elif 'blocks.' in k:
                # mlp.fc1 -> mlp.linear1
                new_key = k.replace('mlp.fc1', 'mlp.linear1')
                # mlp.fc2 -> mlp.linear2
                new_key = new_key.replace('mlp.fc2', 'mlp.linear2')
                # attn.proj -> attn.out_proj
                new_key = new_key.replace('attn.proj.', 'attn.out_proj.')

            # Skip cls_token (MONAI handles this differently)
            elif k == 'cls_token':
                continue

            # Skip norm layer at the end (encoder norm)
            elif k in ['norm.weight', 'norm.bias']:
                new_key = k.replace('norm.', 'norm.')  # Keep as is, MONAI should have this

            mapped_dict[new_key] = v

        # Get model state dict for shape checking
        model_dict = self.vit.state_dict()

        # Filter mapped_dict to only include keys that exist in model with matching shapes
        final_dict = {}
        for k, v in mapped_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    final_dict[k] = v
                else:
                    logger.warning(f"  Shape mismatch for {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
            else:
                logger.debug(f"  Key not in model: {k}")

        # Load weights
        missing, unexpected = self.vit.load_state_dict(final_dict, strict=False)

        logger.info(f"Successfully loaded {len(final_dict)} weights into MONAI ViT")
        logger.info(f"Missing keys: {len(missing)} (will be randomly initialized)")
        if len(missing) > 0:
            logger.debug(f"  Missing: {missing[:5]}...")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT returns (features, hidden_states) when classification=False
        features, _ = self.vit(x)

        # Global average pooling over patches
        # features shape: (B, num_patches, hidden_size)
        features = features.mean(dim=1)  # (B, hidden_size)

        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features, _ = self.vit(x)
        return features.mean(dim=1)


if __name__ == '__main__':
    # Test model creation
    print("Testing MONAI ViT models...")

    for arch in ['vit_tiny', 'vit_small', 'vit_base']:
        print(f"\n{arch}:")

        model = ViT3DClassifier(
            architecture=arch,
            num_classes=2,
            in_channels=1,
            image_size=(96, 96, 96),
            dropout=0.1,
            classifier_dropout=0.5,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 1, 96, 96, 96)
        out = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

        # Test feature extraction
        features = model.extract_features(x)
        print(f"  Features: {features.shape}")
