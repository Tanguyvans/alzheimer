"""
Custom classifier with lazy initialization

This classifier automatically determines the input dimension on the first forward pass.
"""
import torch
import torch.nn as nn


class CustomClassifier(nn.Module):
    """
    Binary/Multi-class classifier with optional hidden layer

    Args:
        hidden_dim: Hidden layer dimension (0 = no hidden layer)
        activation_fun: Activation function (e.g., nn.ReLU())
        num_class: Number of output classes
        task: Task type ("binary" or "multiclass")
        dropout_rate: Dropout rate for regularization (None = no dropout)
    """
    def __init__(self, hidden_dim, activation_fun, num_class, task, dropout_rate=None):
        super(CustomClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_layer1 = activation_fun
        self.num_class = num_class
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.task = task

        # Lazy layer initialization
        self.fc1 = None  # First layer (optional)
        self.fc2 = None  # Output layer
        self.in_dim_initialized = False

    def set_input_dim(self, in_dim, device):
        """
        Initialize layers with the given input dimension.
        This is called automatically on the first forward pass.
        """
        if not self.in_dim_initialized:
            print(f"Initializing layers with input dimension: {in_dim}")
            if self.hidden_dim != 0:
                # Two-layer classifier
                self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(device)
                self.fc2 = nn.Linear(
                    self.hidden_dim,
                    1 if self.task == "binary" else self.num_class
                ).to(device)
            else:
                # Single-layer classifier
                self.fc2 = nn.Linear(
                    in_dim,
                    1 if self.task == "binary" else self.num_class
                ).to(device)
            self.in_dim_initialized = True

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features of shape (B, in_dim)

        Returns:
            Logits of shape (B, num_class) for multiclass or (B, 1) for binary
        """
        # One-time initialization of layers
        if not self.in_dim_initialized:
            self.set_input_dim(x.shape[-1], x.device)

        # Forward pass
        if self.hidden_dim != 0:
            x = self.fc1(x)
            x = self.activation_layer1(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.fc2(x)
        return x
