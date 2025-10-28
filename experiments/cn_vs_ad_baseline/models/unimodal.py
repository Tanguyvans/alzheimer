"""
Unimodal model wrapper

Combines an encoder and classifier for single-modality inputs.
"""
import torch
import torch.nn as nn


class UnimodalModel(nn.Module):
    """
    Unimodal model combining encoder and classifier

    Args:
        encoders: Dictionary of encoders (e.g., {"image": resnet3d})
        classifier: Classifier module
    """
    def __init__(self, encoders: dict, classifier):
        super(UnimodalModel, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.classifier = classifier

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Dictionary with modality as key (e.g., {"image": tensor})

        Returns:
            Classification logits
        """
        # Get the only encoder key (we're unimodal)
        only_key = next(iter(self.encoders.keys()))
        # Encode the input
        encoded = self.encoders[only_key](x[only_key])
        # Classify
        out = self.classifier(encoded)
        return out
