"""Generalisable 3D MPM RCNN-LAM framework.

This package provides a reference implementation of the DL4 architecture
variant from:

    *Three-dimensional mineral prospectivity mapping using a residual
    convolutional neural network with lightweight attention mechanisms*
    (Ore Geology Reviews, 2025).

Key components
--------------
- **Configuration** — :class:`ModelConfig`, :class:`ModelReference`,
  :func:`build_default_configuration`
- **Grid scaling** — :func:`scale_patch_size`
- **Model** — :class:`RCNNLAM`, :class:`ResidualBlock3D`,
  :class:`LightweightAttentionModule`, :func:`build_model`
- **Patch extraction** — :func:`extract_patches`
"""

from .config import ModelConfig, ModelReference, build_default_configuration
from .model import RCNNLAM, LightweightAttentionModule, ResidualBlock3D, build_model
from .patches import extract_patches
from .scaling import scale_patch_size

__all__ = [
    "ModelConfig",
    "ModelReference",
    "build_default_configuration",
    "scale_patch_size",
    "RCNNLAM",
    "ResidualBlock3D",
    "LightweightAttentionModule",
    "build_model",
    "extract_patches",
]
