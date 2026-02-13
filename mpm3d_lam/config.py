"""Typed configuration objects for the RCNN-LAM framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .scaling import scale_patch_size


@dataclass(frozen=True)
class ModelReference:
    """Reference setup from the paper (DL4 variant).

    Attributes
    ----------
    evidence_channels:
        Number of geoscience evidence layers (paper default: 7).
    reference_patch_size:
        Patch size used in the reference study.
    reference_cell_size:
        Grid cell spacing (in metres) of the reference study.
    """

    evidence_channels: int = 7
    reference_patch_size: Tuple[int, int, int] = (3, 3, 3)
    reference_cell_size: Tuple[float, float, float] = (50.0, 50.0, 50.0)


@dataclass(frozen=True)
class ModelConfig:
    """Generalised configuration for RCNN-LAM across arbitrary 3D grid scales.

    Attributes
    ----------
    evidence_channels:
        Number of input evidence channels.
    patch_size:
        Spatial patch dimensions ``(D, H, W)`` after grid-scale adjustment.
    cell_size:
        Target grid cell spacing in metres ``(dz, dy, dx)``.
    base_channels:
        Number of feature maps produced by the initial convolution layer.
    residual_channels:
        Number of feature maps inside each residual block.
    residual_blocks:
        Number of stacked residual blocks in the network.
    lam_reduction_ratio:
        Channel reduction ratio *r* used in the lightweight attention module
        (LAM).  The intermediate channel count is ``residual_channels // r``.
    output_classes:
        Number of output classes (paper default: 2 — prospective / barren).
    dropout_rate:
        Dropout probability applied before the final classifier.
    """

    evidence_channels: int
    patch_size: Tuple[int, int, int]
    cell_size: Tuple[float, float, float]
    base_channels: int = 64
    residual_channels: int = 128
    residual_blocks: int = 2
    lam_reduction_ratio: int = 8
    output_classes: int = 2
    dropout_rate: float = 0.5


def build_default_configuration(
    target_cell_size: Tuple[float, float, float],
    reference: Optional[ModelReference] = None,
    *,
    evidence_channels: Optional[int] = None,
    output_classes: int = 2,
) -> ModelConfig:
    """Build a DL4-style RCNN-LAM config for a target 3D grid spacing.

    Parameters
    ----------
    target_cell_size:
        Grid cell size (in metres) for the target dataset.
    reference:
        A :class:`ModelReference` to scale from.  ``None`` uses the paper
        defaults.
    evidence_channels:
        Override the number of input evidence channels.  When ``None`` the
        value from *reference* is used.
    output_classes:
        Number of output classes.

    Returns
    -------
    ModelConfig
    """
    if reference is None:
        reference = ModelReference()

    patch_size = scale_patch_size(
        reference_patch_size=reference.reference_patch_size,
        reference_cell_size=reference.reference_cell_size,
        target_cell_size=target_cell_size,
        minimum_patch_size=3,
    )

    return ModelConfig(
        evidence_channels=evidence_channels if evidence_channels is not None else reference.evidence_channels,
        patch_size=patch_size,
        cell_size=target_cell_size,
        output_classes=output_classes,
    )
