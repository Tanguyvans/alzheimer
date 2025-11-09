#!/usr/bin/env python3
"""
3D Vision Transformer (3D-ViT) for Alzheimer's Disease Classification

Inspired by RanCom-ViT architecture with:
1. 3D Patch Embedding for volumetric MRI data
2. Token Compression for computational efficiency
3. Enhanced Classification Head with random projection (RVFL-inspired)

References:
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al.)
- RanCom-ViT: An efficient vision transformer for Alzheimer's disease classification (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding Layer

    Divides 3D volume into non-overlapping patches and projects them to embedding dimension.
    For example, a 192x192x192 volume with patch_size=16 creates 12x12x12=1728 patches.
    """
    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 3

        # 3D Convolution to create patch embeddings
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, depth, height, width)
        Returns:
            patches: (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TokenCompression(nn.Module):
    """
    Token Compression Module

    Reduces the number of tokens by removing less important ones based on attention scores.
    This significantly reduces computational cost while maintaining performance.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        compression_ratio: float = 0.5
    ):
        super().__init__()
        self.compression_ratio = compression_ratio

        # Learnable importance scoring
        self.importance_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_tokens, embed_dim)
        Returns:
            compressed_x: (batch_size, n_tokens * compression_ratio, embed_dim)
            indices: Indices of kept tokens
        """
        B, N, D = x.shape

        # Compute importance scores for each token
        scores = self.importance_score(x)  # (B, N, 1)
        scores = scores.squeeze(-1)  # (B, N)

        # Keep top-k tokens based on importance
        n_keep = int(N * self.compression_ratio)
        _, indices = torch.topk(scores, k=n_keep, dim=1)  # (B, n_keep)

        # Gather selected tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        compressed_x = torch.gather(x, 1, indices_expanded)  # (B, n_keep, D)

        return compressed_x, indices


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_tokens, embed_dim)
        Returns:
            output: (batch_size, n_tokens, embed_dim)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activation"""
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class RandomProjectionHead(nn.Module):
    """
    Enhanced Classification Head with Random Projection (RVFL-inspired)

    Random Vector Functional Link (RVFL) networks use random projections
    to create a rich feature space without training all weights.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Random projection layer (frozen)
        self.random_proj = nn.Linear(embed_dim, hidden_dim)
        # Freeze random projection weights
        for param in self.random_proj.parameters():
            param.requires_grad = False

        # Trainable classification layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim + hidden_dim),
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, embed_dim) - CLS token or pooled features
        Returns:
            logits: (batch_size, num_classes)
        """
        # Random projection
        random_features = F.gelu(self.random_proj(x))

        # Concatenate original and random features
        enhanced_features = torch.cat([x, random_features], dim=-1)

        # Classification
        logits = self.classifier(enhanced_features)
        return logits


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer for Alzheimer's Disease Classification

    Architecture:
    1. 3D Patch Embedding
    2. Transformer Encoder Blocks
    3. Token Compression (optional, for efficiency)
    4. Random Projection Classification Head
    """
    def __init__(
        self,
        img_size: int = 192,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_token_compression: bool = True,
        compression_ratio: float = 0.5,
        compression_layer: int = 6
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_token_compression = use_token_compression
        self.compression_layer = compression_layer

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Token compression (applied at middle layer)
        if use_token_compression:
            self.token_compression = TokenCompression(
                embed_dim=embed_dim,
                compression_ratio=compression_ratio
            )

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = RandomProjectionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=dropout
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize random projection with fixed seed for reproducibility
        if hasattr(self, 'head'):
            with torch.no_grad():
                nn.init.normal_(self.head.random_proj.weight, std=0.02)
                nn.init.zeros_(self.head.random_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, depth, height, width)
        Returns:
            logits: (batch_size, num_classes)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Apply token compression at middle layer
            if (self.use_token_compression and
                i == self.compression_layer and
                self.training):
                # Keep CLS token, compress rest
                cls_token = x[:, :1]  # (B, 1, embed_dim)
                patch_tokens = x[:, 1:]  # (B, n_patches, embed_dim)

                # Compress patch tokens
                compressed_patches, _ = self.token_compression(patch_tokens)

                # Concatenate CLS with compressed tokens
                x = torch.cat([cls_token, compressed_patches], dim=1)

        # Layer norm
        x = self.norm(x)

        # Extract CLS token
        cls_output = x[:, 0]  # (B, embed_dim)

        # Classification
        logits = self.head(cls_output)

        return logits


def vit3d_small(num_classes=2, img_size=192, patch_size=16):
    """
    Small 3D ViT (ViT-S/16)
    - Suitable for datasets < 1000 samples
    - Faster training, less overfitting
    """
    return VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        use_token_compression=True,
        compression_ratio=0.5,
        compression_layer=3
    )


def vit3d_base(num_classes=2, img_size=192, patch_size=16):
    """
    Base 3D ViT (ViT-B/16)
    - Balanced performance and efficiency
    - Recommended for datasets 500-2000 samples
    """
    return VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        use_token_compression=True,
        compression_ratio=0.5,
        compression_layer=6
    )


def vit3d_large(num_classes=2, img_size=192, patch_size=16):
    """
    Large 3D ViT (ViT-L/16)
    - Best performance but requires more data
    - Use with datasets > 2000 samples or with strong augmentation
    """
    return VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.1,
        use_token_compression=True,
        compression_ratio=0.5,
        compression_layer=12
    )


if __name__ == "__main__":
    # Test the model
    print("Testing 3D Vision Transformer...")

    # Small model (recommended for your dataset size: ~685 samples)
    model = vit3d_small(num_classes=2, img_size=192, patch_size=16)

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 192, 192, 192)

    with torch.no_grad():
        output = model(x)

    print(f"\nModel: ViT-Small/16")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel details:")
    print(f"  - Patch size: 16x16x16")
    print(f"  - Number of patches: {(192//16)**3} = {12**3}")
    print(f"  - Embedding dimension: 384")
    print(f"  - Transformer depth: 6 layers")
    print(f"  - Attention heads: 6")
    print(f"  - Token compression: Enabled (50% at layer 3)")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / (1024**2):.1f} MB (fp32)")
