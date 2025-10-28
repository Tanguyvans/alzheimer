"""
ResNet3D with MedicalNet pretrained weights

This module provides a 3D ResNet encoder with pretrained weights from MedicalNet.
MedicalNet was trained on 23 medical imaging datasets including brain MRI.

Reference: https://github.com/Tencent/MedicalNet
"""
import torch
import torch.nn as nn
import logging
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, get_pretrained_resnet_medicalnet


depth_map = {
    10: resnet10,
    18: resnet18,
    34: resnet34,
    50: resnet50
}


class Resnet3D(nn.Module):
    """
    3D ResNet encoder with MedicalNet pretrained weights

    Args:
        depth: ResNet depth (10, 18, 34, or 50)
        freeze: Whether to freeze encoder weights
        include_head: Whether to include classification head (not used here)
        num_classes: Number of classes for classification head (not used here)
    """
    def __init__(self, depth, freeze=True, include_head=False, num_classes=None):
        super(Resnet3D, self).__init__()
        self.freeze = freeze
        self.include_head = include_head

        # Create ResNet architecture from MONAI
        net = depth_map[depth](
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        )

        # Load MedicalNet pretrained weights
        pretrained_state_dict = get_pretrained_resnet_medicalnet(resnet_depth=depth)
        net_dict = net.state_dict()

        # Clean up state dict keys
        pretrained_state_dict = {
            k.replace("module.", ""): v for k, v in pretrained_state_dict.items()
        }

        # Filter to only matching keys
        missing = tuple({k for k in net_dict if k not in pretrained_state_dict})
        inside = tuple({k for k in pretrained_state_dict if k in net_dict})
        unused = tuple({k for k in pretrained_state_dict if k not in net_dict})

        logging.debug(f"Loaded pretrained weights: {len(inside)} layers")
        logging.debug(f"Missing in pretrained: {len(missing)} layers")
        logging.debug(f"Unused from pretrained: {len(unused)} layers")

        assert len(inside) > len(missing), "More layers missing than loaded from pretrained weights"
        assert len(inside) > len(unused), "More unused layers than loaded from pretrained weights"

        pretrained_state_dict = {
            k: v for k, v in pretrained_state_dict.items() if k in net_dict
        }
        net.load_state_dict(pretrained_state_dict, strict=False)

        # Extract features (remove classification head)
        self.features_extractor = nn.Sequential(*list(net.children())[:-1])

        # Freeze if requested
        self.freeze_parameters()

        # Optional classification head (not used in our pipeline)
        self.classifier = None
        if include_head and num_classes:
            self.classifier = nn.LazyLinear(num_classes)

    def freeze_parameters(self):
        """Freeze encoder parameters if requested"""
        if self.freeze and self.features_extractor is not None:
            logging.info("Freezing ResNet3D encoder parameters")
            for param in self.features_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 1, D, H, W)

        Returns:
            Feature tensor of shape (B, 2048) for ResNet-50
        """
        x = self.features_extractor(x).squeeze()
        if self.classifier:
            x = self.classifier(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x
