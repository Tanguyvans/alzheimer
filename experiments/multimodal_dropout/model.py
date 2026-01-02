#!/usr/bin/env python3
"""
Multi-Modal Fusion Model with Modality Dropout Support

Key features:
- Handles missing modalities gracefully
- Optional modality availability embeddings
- Works with MRI-only, tabular-only, or both modalities
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

# Import ViT model from mri_vit_ad
import importlib.util
mri_vit_model_path = Path(__file__).parent.parent / "mri_vit_ad" / "model.py"
spec = importlib.util.spec_from_file_location("mri_vit_model", mri_vit_model_path)
mri_vit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mri_vit_module)
ViT3DClassifier = mri_vit_module.ViT3DClassifier

logger = logging.getLogger(__name__)


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
