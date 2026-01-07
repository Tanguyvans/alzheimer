#!/usr/bin/env python3
"""
Multi-Modal Fusion Model: MRI backbone (ViT or ResNet3D) + Tabular Features

Architecture:
- MRI backbone extracts features from 3D MRI (ViT: 768-dim, ResNet: 512/2048-dim)
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

# Try to import MONAI for ResNet3D
try:
    from monai.networks.nets import ResNet
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available, ResNet3D backbone disabled")


class ResNet3DBackbone(nn.Module):
    """3D ResNet backbone for MRI feature extraction using MONAI"""

    # ResNet output dimensions by depth
    FEATURE_DIMS = {
        10: 512,
        18: 512,
        34: 512,
        50: 2048,
        101: 2048,
        152: 2048,
        200: 2048
    }

    def __init__(
        self,
        depth: int = 50,
        in_channels: int = 1,
        pretrained: bool = False,
        spatial_dims: int = 3
    ):
        super().__init__()
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for ResNet3D backbone")

        self.feature_dim = self.FEATURE_DIMS.get(depth, 2048)

        # Create ResNet without classification head
        self.resnet = ResNet(
            block='bottleneck' if depth >= 50 else 'basic',
            layers=self._get_layers(depth),
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=self.feature_dim,  # Will extract features before this
            feed_forward=False  # Don't use final FC layer
        )

    def _get_layers(self, depth: int):
        """Get layer configuration for ResNet depth"""
        configs = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }
        return configs.get(depth, [3, 4, 6, 3])

    def forward(self, x):
        """Extract features from 3D MRI volume"""
        features = self.resnet(x)
        # Global average pool if needed
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3, 4])
        return features


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


class FTTransformerEncoder(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer) for tabular data.

    Each numerical feature is embedded independently, then processed
    through transformer layers for feature interaction learning.

    Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Feature tokenizer: embed each feature independently
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(input_dim)
        ])

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding for features
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - tabular features
        Returns:
            (batch_size, output_dim) - encoded features
        """
        B = x.shape[0]

        # Tokenize each feature independently
        # x[:, i:i+1] has shape (B, 1)
        tokens = []
        for i, embed in enumerate(self.feature_embeddings):
            feat_token = embed(x[:, i:i+1])  # (B, embed_dim)
            tokens.append(feat_token)

        tokens = torch.stack(tokens, dim=1)  # (B, input_dim, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, input_dim+1, embed_dim)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer encoding
        tokens = self.transformer(tokens)

        # Extract [CLS] token output
        cls_output = tokens[:, 0]  # (B, embed_dim)

        # Project to output dimension
        output = self.output_proj(cls_output)

        return output


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


class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion

    Implements bidirectional attention where:
    - MRI features attend to tabular features
    - Tabular features attend to MRI features

    Based on cross-modal attention mechanisms from multimodal learning literature.
    """

    def __init__(
        self,
        img_dim: int,
        tab_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both modalities to same dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)

        # Bidirectional cross-attention
        # MRI attends to tabular
        self.mri_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Tabular attends to MRI
        self.tab_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.norm_mri = nn.LayerNorm(hidden_dim)
        self.norm_tab = nn.LayerNorm(hidden_dim)

        # Feed-forward networks
        self.ffn_mri = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_tab = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm_mri2 = nn.LayerNorm(hidden_dim)
        self.norm_tab2 = nn.LayerNorm(hidden_dim)

        # Final fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, img_feat, tab_feat):
        """
        Args:
            img_feat: (B, img_dim) - MRI features
            tab_feat: (B, tab_dim) - Tabular features
        Returns:
            fused: (B, hidden_dim) - Fused features
        """
        # Project to hidden dimension
        img_h = self.img_proj(img_feat).unsqueeze(1)  # (B, 1, H)
        tab_h = self.tab_proj(tab_feat).unsqueeze(1)  # (B, 1, H)

        # Cross-attention: MRI attends to tabular
        mri_attn_out, _ = self.mri_cross_attn(img_h, tab_h, tab_h)
        img_h = self.norm_mri(img_h + mri_attn_out)
        img_h = self.norm_mri2(img_h + self.ffn_mri(img_h))

        # Cross-attention: Tabular attends to MRI
        tab_attn_out, _ = self.tab_cross_attn(tab_h, img_h, img_h)
        tab_h = self.norm_tab(tab_h + tab_attn_out)
        tab_h = self.norm_tab2(tab_h + self.ffn_tab(tab_h))

        # Squeeze and concatenate
        img_out = img_h.squeeze(1)  # (B, H)
        tab_out = tab_h.squeeze(1)  # (B, H)

        # Fuse
        fused = torch.cat([img_out, tab_out], dim=1)  # (B, 2H)
        fused = self.fusion_fc(fused)  # (B, H)

        return fused


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Fusion Model

    Combines:
    - 3D ViT for MRI feature extraction
    - MLP for tabular clinical features
    - Late fusion for classification (concat, gated, attention, or cross_modal)
    """

    def __init__(
        self,
        num_tabular_features: int,
        num_classes: int = 2,
        backbone_config: dict = None,
        tabular_config: dict = None,
        fusion_config: dict = None,
        pretrained_path: str = None,
        freeze_backbone: bool = True
    ):
        super().__init__()

        # Default configs
        backbone_config = backbone_config or {'type': 'vit'}
        tabular_config = tabular_config or {'hidden_dims': [128, 64], 'dropout': 0.3}
        fusion_config = fusion_config or {'method': 'concat', 'hidden_dim': 256, 'dropout': 0.5}

        backbone_type = backbone_config.get('type', 'vit')
        self.backbone_type = backbone_type

        # Create MRI backbone based on type
        if backbone_type == 'resnet':
            # ResNet3D backbone
            depth = backbone_config.get('depth', 50)
            self.mri_backbone = ResNet3DBackbone(
                depth=depth,
                in_channels=1,
                pretrained=False
            )
            self.mri_feature_dim = self.mri_backbone.feature_dim
            print(f"ResNet3D-{depth} backbone: {self.mri_feature_dim}-dim features")

        else:
            # ViT backbone (default)
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
            print(f"ViT backbone: {self.mri_feature_dim}-dim features")

        # Alias for backwards compatibility
        self.vit = self.mri_backbone
        self.vit_feature_dim = self.mri_feature_dim

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.mri_backbone.parameters():
                param.requires_grad = False
            print(f"{backbone_type.upper()} backbone frozen")

        # Tabular encoder (MLP or FT-Transformer)
        tabular_type = tabular_config.get('type', 'mlp')
        self.tabular_type = tabular_type

        if tabular_type == 'ft_transformer':
            self.tabular_encoder = FTTransformerEncoder(
                input_dim=num_tabular_features,
                embed_dim=tabular_config.get('embed_dim', 64),
                num_heads=tabular_config.get('num_heads', 4),
                num_layers=tabular_config.get('num_layers', 3),
                dropout=tabular_config.get('dropout', 0.1),
                output_dim=tabular_config.get('output_dim', 64)
            )
            print(f"FT-Transformer tabular encoder: {num_tabular_features} -> {self.tabular_encoder.output_dim}")
        else:
            # Default MLP encoder
            self.tabular_encoder = TabularEncoder(
                input_dim=num_tabular_features,
                hidden_dims=tabular_config.get('hidden_dims', [128, 64]),
                dropout=tabular_config.get('dropout', 0.3)
            )
            print(f"MLP tabular encoder: {num_tabular_features} -> {self.tabular_encoder.output_dim}")

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

        elif fusion_method == 'cross_modal':
            num_heads = fusion_config.get('num_heads', 8)
            self.fusion = CrossModalAttention(
                self.vit_feature_dim,
                self.tabular_feature_dim,
                fusion_hidden,
                num_heads=num_heads,
                dropout=fusion_dropout
            )
            classifier_input_dim = fusion_hidden

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_method = fusion_method

        # Auxiliary losses support
        self.use_auxiliary_losses = fusion_config.get('auxiliary_losses', False)
        if self.use_auxiliary_losses:
            # Modality-specific classifiers for regularization
            self.mri_classifier = nn.Sequential(
                nn.Linear(self.mri_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(128, num_classes)
            )
            self.tab_classifier = nn.Sequential(
                nn.Linear(self.tabular_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(64, num_classes)
            )
            print(f"  Auxiliary losses: Enabled (MRI + Tabular classifiers)")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(128, num_classes)
        )

        print(f"MultiModalFusion initialized:")
        print(f"  MRI backbone: {backbone_type} ({self.mri_feature_dim}-dim)")
        print(f"  Tabular features: {num_tabular_features} -> {self.tabular_feature_dim}")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Fusion hidden: {fusion_hidden}")

    def extract_mri_features(self, x):
        """Extract features from MRI backbone (ViT or ResNet)"""
        if self.backbone_type == 'resnet':
            # ResNet3D forward
            return self.mri_backbone(x)
        else:
            # ViT forward - extract cls token features
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

    # Alias for backwards compatibility
    def extract_vit_features(self, x):
        return self.extract_mri_features(x)

    def forward(self, mri, tabular, return_auxiliary=False):
        """
        Forward pass

        Args:
            mri: (B, 1, D, H, W) - 3D MRI volume
            tabular: (B, num_features) - Tabular clinical features
            return_auxiliary: If True, return auxiliary logits for auxiliary losses

        Returns:
            logits: (B, num_classes)
            or dict with 'logits', 'mri_logits', 'tab_logits' if return_auxiliary=True
        """
        # Extract MRI features
        with torch.set_grad_enabled(not self._is_backbone_frozen()):
            img_features = self.extract_mri_features(mri)

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

        # Return auxiliary outputs if requested
        if return_auxiliary and self.use_auxiliary_losses:
            mri_logits = self.mri_classifier(img_features)
            tab_logits = self.tab_classifier(tab_features)
            return {
                'logits': logits,
                'mri_logits': mri_logits,
                'tab_logits': tab_logits
            }

        return logits

    def _is_backbone_frozen(self):
        """Check if MRI backbone is frozen"""
        return not next(self.mri_backbone.parameters()).requires_grad

    # Alias for backwards compatibility
    def _is_vit_frozen(self):
        return self._is_backbone_frozen()

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

    # Support both old 'vit' config and new 'backbone' config
    if 'backbone' in model_cfg:
        backbone_config = model_cfg['backbone']
        pretrained_path = backbone_config.get('pretrained_path')
        freeze_backbone = backbone_config.get('freeze', True)
    else:
        # Legacy: convert 'vit' section to backbone_config
        vit_cfg = model_cfg.get('vit', {})
        backbone_config = {
            'type': 'vit',
            'architecture': vit_cfg.get('architecture', 'vit_base'),
            'image_size': vit_cfg.get('image_size', 128),
            'feature_dim': vit_cfg.get('feature_dim', 768)
        }
        pretrained_path = vit_cfg.get('pretrained_path')
        freeze_backbone = vit_cfg.get('freeze_backbone', True)

    model = MultiModalFusion(
        num_tabular_features=num_tabular_features,
        num_classes=model_cfg['num_classes'],
        backbone_config=backbone_config,
        tabular_config=model_cfg.get('tabular', {}),
        fusion_config=model_cfg.get('fusion', {}),
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone
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
