#!/usr/bin/env python3
"""
ResNet50 3D Backbone (MedicalNet pretrained) + Early Fusion Model for Alzheimer's Classification.

- ResNet3DBackbone: MONAI ResNet50 (spatial_dims=3) with MedicalNet pretrained weights, outputs 2048-d features
- EarlyFusionModel: ResNet3D + Tabular MLP encoder -> concat -> MLP classifier
"""

import torch
import torch.nn as nn
from monai.networks.nets.resnet import resnet50


class ResNet3DBackbone(nn.Module):
    """
    ResNet50 3D backbone (MONAI + MedicalNet pretrained) for feature extraction.

    Pretrained on 23 medical imaging datasets (Med3D).
    Outputs 2048-d features (feed_forward=False removes the final FC layer).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # MedicalNet pretrained requires: n_input_channels=1, feed_forward=False,
        # shortcut_type='B', bias_downsample=False
        self.net = resnet50(
            pretrained=pretrained,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            shortcut_type='B',
            bias_downsample=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) MRI volume
        Returns:
            (B, 2048) feature vector
        """
        return self.net(x)


class EarlyFusionModel(nn.Module):
    """
    Early Fusion: ResNet50 3D (MRI) + Tabular MLP -> concat -> classifier.

    Architecture:
        MRI branch:     ResNet50 3D (MedicalNet pretrained) -> (B, 2048)
        Tabular branch: MLP [input_dim -> 64 -> 32] with LayerNorm+ReLU+Dropout
        Fusion:         concat (B, 2080) -> MLP [256, 128] -> (B, num_classes)
    """

    def __init__(
        self,
        pretrained: bool = True,
        tabular_input_dim: int = 16,
        tabular_hidden_dims: list = None,
        fusion_hidden_dims: list = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if tabular_hidden_dims is None:
            tabular_hidden_dims = [64, 32]
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256, 128]

        # MRI backbone
        self.backbone = ResNet3DBackbone(pretrained=pretrained)
        cnn_feat_dim = 2048

        # Tabular encoder: input_dim -> hidden_dims with LayerNorm + ReLU + Dropout
        tab_layers = []
        in_dim = tabular_input_dim
        for h_dim in tabular_hidden_dims:
            tab_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.tabular_encoder = nn.Sequential(*tab_layers)
        tab_out_dim = tabular_hidden_dims[-1]

        # Fusion classifier: concat -> MLP -> num_classes
        concat_dim = cnn_feat_dim + tab_out_dim
        fusion_layers = []
        in_dim = concat_dim
        for h_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        fusion_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*fusion_layers)

    def forward(self, mri: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mri: (B, 1, D, H, W)
            tabular: (B, tabular_input_dim)
        Returns:
            (B, num_classes) logits
        """
        cnn_feat = self.backbone(mri)
        tab_feat = self.tabular_encoder(tabular)
        fused = torch.cat([cnn_feat, tab_feat], dim=1)
        return self.classifier(fused)

    @torch.no_grad()
    def extract_cnn_features(self, mri: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features only (for late fusion with XGBoost).

        Args:
            mri: (B, 1, D, H, W)
        Returns:
            (B, 2048) CNN feature vectors
        """
        self.backbone.eval()
        return self.backbone(mri)


def build_early_fusion_model(config: dict) -> EarlyFusionModel:
    """Build EarlyFusionModel from config dict."""
    model_cfg = config['model']
    tab_cfg = model_cfg['tabular']
    fusion_cfg = model_cfg['early_fusion']
    resnet_cfg = model_cfg.get('resnet', {})

    return EarlyFusionModel(
        pretrained=resnet_cfg.get('pretrained', True),
        tabular_input_dim=tab_cfg['input_dim'],
        tabular_hidden_dims=tab_cfg['hidden_dims'],
        fusion_hidden_dims=fusion_cfg['hidden_dims'],
        num_classes=fusion_cfg['num_classes'],
        dropout=fusion_cfg['dropout'],
    )


if __name__ == '__main__':
    # Quick test
    print("Loading ResNet50 3D with MedicalNet pretrained weights...")
    model = EarlyFusionModel(pretrained=True, tabular_input_dim=16, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EarlyFusionModel total parameters: {total_params:,}")

    mri = torch.randn(2, 1, 128, 128, 128)
    tab = torch.randn(2, 16)
    logits = model(mri, tab)
    print(f"Input MRI: {mri.shape}, Tabular: {tab.shape}")
    print(f"Output logits: {logits.shape}")

    feats = model.extract_cnn_features(mri)
    print(f"CNN features: {feats.shape}")
