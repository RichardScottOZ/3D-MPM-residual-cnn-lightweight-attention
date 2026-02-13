"""3D Residual CNN with Lightweight Attention Module (RCNN-LAM).

Implements the DL4 architecture variant from the paper:
    1 initial Conv3d → *N* Residual Blocks → Lightweight Attention Module
    → Global Average Pooling → Dropout → Fully-Connected classifier.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualBlock3D(nn.Module):
    """A single 3D residual block: two 3×3×3 conv layers with batch-norm and
    a skip connection.

    When the number of input channels differs from *out_channels* a 1×1×1
    projection shortcut is used.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class LightweightAttentionModule(nn.Module):
    """Channel-attention module using global average pooling and a
    bottleneck MLP, as described in the paper.

    Architecture:
        GAP → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid → scale input
    """

    def __init__(self, channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        if reduction_ratio < 1:
            raise ValueError("reduction_ratio must be >= 1")
        mid = max(1, channels // reduction_ratio)
        self.fc1 = nn.Linear(channels, mid, bias=True)
        self.fc2 = nn.Linear(mid, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        b, c = x.shape[:2]
        # Global average pooling over spatial dims
        scale = x.view(b, c, -1).mean(dim=2)  # (B, C)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        # Reshape for broadcasting and rescale
        scale = scale.view(b, c, 1, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class RCNNLAM(nn.Module):
    """3D Residual CNN with Lightweight Attention Module.

    Parameters
    ----------
    config:
        A :class:`ModelConfig` instance describing the network shape.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # --- initial convolution ---
        self.init_conv = nn.Sequential(
            nn.Conv3d(config.evidence_channels, config.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        # --- residual blocks ---
        blocks: list[nn.Module] = []
        in_ch = config.base_channels
        for _ in range(config.residual_blocks):
            blocks.append(ResidualBlock3D(in_ch, config.residual_channels))
            in_ch = config.residual_channels
        self.res_blocks = nn.Sequential(*blocks)

        # --- lightweight attention ---
        self.lam = LightweightAttentionModule(config.residual_channels, config.lam_reduction_ratio)

        # --- classifier head ---
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.fc = nn.Linear(config.residual_channels, config.output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C_in, D, H, W)`` where ``C_in`` is
            ``evidence_channels``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, output_classes)``.
        """
        out = self.init_conv(x)
        out = self.res_blocks(out)
        out = self.lam(out)
        out = self.gap(out).flatten(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def build_model(config: ModelConfig) -> RCNNLAM:
    """Convenience constructor that returns an :class:`RCNNLAM` from a config."""
    return RCNNLAM(config)
