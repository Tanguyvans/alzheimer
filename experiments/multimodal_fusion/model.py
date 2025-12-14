#!/usr/bin/env python3
"""
Multi-Modal Fusion Model: ViT (MRI) + Tabular Features

Architecture:
- ViT backbone extracts 768-dim features from 3D MRI
- MLP processes tabular clinical data
- Late fusion combines both for classification
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import ViT model from mri_vit_ad using importlib
import importlib.util
mri_vit_model_path = Path(__file__).parent.parent / "mri_vit_ad" / "model.py"
spec = importlib.util.spec_from_file_location("mri_vit_model", mri_vit_model_path)
mri_vit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mri_vit_module)
ViT3DClassifier = mri_vit_module.ViT3DClassifier


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


class AttentionFusion(nn.Module):
    """Cross-attention fusion between modalities"""

    def __init__(self, img_dim: int, tab_dim: int, hidden_dim: int):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, img_feat, tab_feat):
        img_proj = self.img_proj(img_feat).unsqueeze(1)  # (B, 1, H)
        tab_proj = self.tab_proj(tab_feat).unsqueeze(1)  # (B, 1, H)

        # Cross attention
        tokens = torch.cat([img_proj, tab_proj], dim=1)  # (B, 2, H)
        attn_out, _ = self.attention(tokens, tokens, tokens)

        # Pool and project
        pooled = attn_out.mean(dim=1)  # (B, H)
        concat = torch.cat([pooled, img_proj.squeeze(1) + tab_proj.squeeze(1)], dim=1)
        return self.fc(concat)


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Fusion Model

    Combines:
    - 3D ViT for MRI feature extraction
    - MLP for tabular clinical features
    - Late fusion for classification
    """

    def __init__(
        self,
        num_tabular_features: int,
        num_classes: int = 2,
        vit_config: dict = None,
        tabular_config: dict = None,
        fusion_config: dict = None,
        pretrained_vit_path: str = None,
        freeze_vit: bool = True
    ):
        super().__init__()

        # Default configs
        vit_config = vit_config or {}
        tabular_config = tabular_config or {'hidden_dims': [128, 64], 'dropout': 0.3}
        fusion_config = fusion_config or {'method': 'concat', 'hidden_dim': 256, 'dropout': 0.5}

        # ViT backbone for MRI
        self.vit = ViT3DClassifier(
            architecture=vit_config.get('architecture', 'vit_base'),
            num_classes=num_classes,  # Will be replaced
            in_channels=1,
            image_size=vit_config.get('image_size', 128),
            pretrained_path=pretrained_vit_path,
            dropout=0.1,
            classifier_dropout=0.0  # We'll use our own classifier
        )

        # Remove ViT's classifier head - we'll use fusion instead
        self.vit_feature_dim = vit_config.get('feature_dim', 768)

        # Freeze ViT if specified
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT backbone frozen")

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=num_tabular_features,
            hidden_dims=tabular_config.get('hidden_dims', [128, 64]),
            dropout=tabular_config.get('dropout', 0.3)
        )
        self.tabular_feature_dim = self.tabular_encoder.output_dim

        # Fusion layer
        fusion_method = fusion_config.get('method', 'concat')
        fusion_hidden = fusion_config.get('hidden_dim', 256)
        fusion_dropout = fusion_config.get('dropout', 0.5)

        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(self.vit_feature_dim + self.tabular_feature_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            )
            classifier_input_dim = fusion_hidden

        elif fusion_method == 'gated':
            self.fusion = GatedFusion(
                self.vit_feature_dim,
                self.tabular_feature_dim,
                fusion_hidden
            )
            classifier_input_dim = fusion_hidden

        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(
                self.vit_feature_dim,
                self.tabular_feature_dim,
                fusion_hidden
            )
            classifier_input_dim = fusion_hidden

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_method = fusion_method

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(128, num_classes)
        )

        print(f"MultiModalFusion initialized:")
        print(f"  ViT features: {self.vit_feature_dim}")
        print(f"  Tabular features: {num_tabular_features} -> {self.tabular_feature_dim}")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Fusion hidden: {fusion_hidden}")

    def extract_vit_features(self, x):
        """Extract features from ViT (before classification head)"""
        # Forward through ViT up to the cls token
        x = self.vit.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for block in self.vit.blocks:
            x = block(x)

        x = self.vit.norm(x)

        # Return cls token features
        return x[:, 0]

    def forward(self, mri, tabular):
        """
        Forward pass

        Args:
            mri: (B, 1, D, H, W) - 3D MRI volume
            tabular: (B, num_features) - Tabular clinical features

        Returns:
            logits: (B, num_classes)
        """
        # Extract MRI features
        with torch.set_grad_enabled(not self._is_vit_frozen()):
            img_features = self.extract_vit_features(mri)

        # Encode tabular features
        tab_features = self.tabular_encoder(tabular)

        # Fusion
        if self.fusion_method == 'concat':
            combined = torch.cat([img_features, tab_features], dim=1)
            fused = self.fusion(combined)
        else:
            fused = self.fusion(img_features, tab_features)

        # Classification
        logits = self.classifier(fused)

        return logits

    def _is_vit_frozen(self):
        """Check if ViT is frozen"""
        return not next(self.vit.parameters()).requires_grad

    def unfreeze_vit(self, unfreeze_layers: int = -1):
        """
        Unfreeze ViT layers for fine-tuning

        Args:
            unfreeze_layers: Number of transformer blocks to unfreeze from the end
                            -1 = unfreeze all
        """
        if unfreeze_layers == -1:
            for param in self.vit.parameters():
                param.requires_grad = True
            print("All ViT layers unfrozen")
        else:
            # Unfreeze last N blocks + norm + head
            num_blocks = len(self.vit.blocks)
            for i, block in enumerate(self.vit.blocks):
                if i >= num_blocks - unfreeze_layers:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.vit.norm.parameters():
                param.requires_grad = True
            print(f"Last {unfreeze_layers} ViT blocks unfrozen")


def build_model(config: dict, num_tabular_features: int) -> MultiModalFusion:
    """Build multi-modal fusion model from config"""

    model_cfg = config['model']

    model = MultiModalFusion(
        num_tabular_features=num_tabular_features,
        num_classes=model_cfg['num_classes'],
        vit_config=model_cfg.get('vit', {}),
        tabular_config=model_cfg.get('tabular', {}),
        fusion_config=model_cfg.get('fusion', {}),
        pretrained_vit_path=model_cfg['vit'].get('pretrained_path'),
        freeze_vit=model_cfg['vit'].get('freeze_backbone', True)
    )

    return model


if __name__ == '__main__':
    # Test model
    model = MultiModalFusion(
        num_tabular_features=9,
        num_classes=2,
        vit_config={'architecture': 'vit_base', 'image_size': 128, 'feature_dim': 768},
        tabular_config={'hidden_dims': [128, 64], 'dropout': 0.3},
        fusion_config={'method': 'concat', 'hidden_dim': 256, 'dropout': 0.5},
        pretrained_vit_path=None,
        freeze_vit=True
    )

    # Test forward
    mri = torch.randn(2, 1, 128, 128, 128)
    tabular = torch.randn(2, 9)

    with torch.no_grad():
        output = model(mri, tabular)
        print(f"Output shape: {output.shape}")
