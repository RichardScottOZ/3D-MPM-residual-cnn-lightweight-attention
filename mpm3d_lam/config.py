from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .scaling import scale_patch_size


@dataclass(frozen=True)
class ModelReference:
    """Reference setup from the paper (DL4 variant)."""

    evidence_channels: int = 7
    reference_patch_size: Tuple[int, int, int] = (3, 3, 3)
    reference_cell_size: Tuple[float, float, float] = (50.0, 50.0, 50.0)


@dataclass(frozen=True)
class ModelConfig:
    """Generalised configuration for RCNN-LAM across arbitrary 3D grid scales."""

    evidence_channels: int
    patch_size: Tuple[int, int, int]
    cell_size: Tuple[float, float, float]
    base_channels: int = 64
    residual_channels: int = 128
    residual_blocks: int = 2
    lam_reduction_channels: int = 8
    output_classes: int = 2


def build_default_configuration(
    target_cell_size: Tuple[float, float, float],
    reference: ModelReference = ModelReference(),
) -> ModelConfig:
    """Build a DL4-style RCNN-LAM config for a target 3D grid spacing."""

    patch_size = scale_patch_size(
        reference_patch_size=reference.reference_patch_size,
        reference_cell_size=reference.reference_cell_size,
        target_cell_size=target_cell_size,
        minimum_patch_size=3,
    )

    return ModelConfig(
        evidence_channels=reference.evidence_channels,
        patch_size=patch_size,
        cell_size=target_cell_size,
    )
