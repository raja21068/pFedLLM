"""
models/feature_compressor.py
─────────────────────────────
Client-side feature compressor  C_φc : I → h_c ∈ ℝ^d

Architecture (Section IV.C):
    3 convolutional blocks  →  Global Average Pooling  →  Linear projection

Each block: Conv2d → BatchNorm → ReLU → MaxPool
The final linear layer maps to the shared embedding dimension d (default 1024).
This compact encoder suppresses identifiable pixel-level information while
preserving clinically relevant features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Single convolutional block: Conv → BN → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class FeatureCompressor(nn.Module):
    """
    Client-side lightweight image encoder  C_φc.

    Input  : x ∈ ℝ^{B × C_in × H × W}   (chest X-ray image)
    Output : h_c ∈ ℝ^{B × d}             (compact latent embedding)

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for grayscale CXR, 3 for RGB).
    channels : List[int]
        Output channels of each convolutional block.
        Default [32, 64, 128] gives three blocks as described in the paper.
    embed_dim : int
        Dimensionality d of the output embedding (default 1024).
    dropout : float
        Dropout probability applied before the linear projection.
    use_vit : bool
        If True, replace CNN blocks with a tiny patch-based ViT encoder
        (alternative mentioned in Section IV.C).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        embed_dim: int = 1024,
        dropout: float = 0.1,
        use_vit: bool = False,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.use_vit = use_vit
        self.embed_dim = embed_dim

        if use_vit:
            self.encoder = TinyViTEncoder(in_channels, embed_dim)
        else:
            # ── 3 CNN blocks ──────────────────────────────────────────
            blocks = []
            prev = in_channels
            for out_ch in channels:
                blocks.append(ConvBlock(prev, out_ch, pool=True))
                prev = out_ch
            self.blocks = nn.Sequential(*blocks)

            # ── Global Average Pooling → Linear projection ────────────
            self.gap  = nn.AdaptiveAvgPool2d(1)   # → (B, channels[-1], 1, 1)
            self.drop = nn.Dropout(dropout)
            self.proj = nn.Linear(channels[-1], embed_dim)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, H, W)

        Returns
        -------
        h_c : (B, embed_dim)   — compressed feature sent to server
        """
        if self.use_vit:
            return self.encoder(x)

        feat = self.blocks(x)          # (B, 128, H/8, W/8)
        feat = self.gap(feat)          # (B, 128, 1, 1)
        feat = feat.flatten(1)         # (B, 128)
        feat = self.drop(feat)
        h_c  = self.proj(feat)         # (B, embed_dim)
        return h_c

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def communication_cost_bytes(self, batch_size: int = 1) -> int:
        """Bytes needed to transmit h_c for a given batch (float32)."""
        return batch_size * self.embed_dim * 4   # 4 bytes per float32


# ── Optional tiny ViT encoder ─────────────────────────────────────────────────

class TinyViTEncoder(nn.Module):
    """
    Minimal patch-based Vision Transformer (alternative compressor backbone).
    16×16 patches, 4 transformer layers, projects to embed_dim.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 1024,
                 patch_size: int = 16, num_heads: int = 4, depth: int = 4):
        super().__init__()
        hidden = 256
        self.patch_embed = nn.Conv2d(in_channels, hidden,
                                     kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden))
        self.proj        = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        patches = self.patch_embed(x)                   # (B, hidden, nP, nP)
        patches = patches.flatten(2).transpose(1, 2)    # (B, N, hidden)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, patches], dim=1)          # (B, N+1, hidden)
        out = self.transformer(seq)
        return self.proj(out[:, 0])                     # (B, embed_dim)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_compressor(cfg) -> FeatureCompressor:
    """Build a FeatureCompressor from a FederatedConfig object."""
    return FeatureCompressor(
        in_channels=cfg.image_channels,
        channels=cfg.compressor_channels,
        embed_dim=cfg.embedding_dim,
    )
