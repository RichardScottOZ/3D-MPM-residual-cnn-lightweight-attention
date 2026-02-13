# 3D-MPM-residual-cnn-lightweight-attention

Generalised framework implementation of the paper:
**Three-dimensional mineral prospectivity mapping using a residual convolutional neural network with lightweight attention mechanisms** (Ore Geology Reviews, 2025).

## What is included

- `mpm3d_lam` Python package with:
  - **Model architecture** — 3D Residual CNN with Lightweight Attention Module
    (`RCNNLAM`), including `ResidualBlock3D` and `LightweightAttentionModule`
    building blocks.
  - **Configuration** — DL4-style reference configuration (`ModelReference`,
    `ModelConfig`) with `build_default_configuration()`.
  - **Grid-scale generalisation** — `scale_patch_size()` rescales patch
    dimensions from a reference cell size to a target cell size, preserving
    physical receptive field across resolutions.
  - **Patch extraction** — `extract_patches()` extracts 3D sub-volumes from
    gridded evidence data for model input.
- Comprehensive unit tests for scaling, model, and patch extraction.
- `pyproject.toml` for standard Python packaging.

## Installation

```bash
pip install -e .
```

## Quick usage

### Scale-aware configuration

```python
from mpm3d_lam import build_default_configuration

# Paper reference is 3x3x3 at 50 m cell size.
# Build a configuration for another grid spacing:
config = build_default_configuration(target_cell_size=(25.0, 25.0, 25.0))
print(config.patch_size)  # -> (7, 7, 7)
```

### Build and run the model

```python
import torch
from mpm3d_lam import build_default_configuration, build_model

config = build_default_configuration(target_cell_size=(50.0, 50.0, 50.0))
model = build_model(config)

# Dummy batch: 4 samples, 7 evidence channels, 3x3x3 patch
x = torch.randn(4, config.evidence_channels, *config.patch_size)
logits = model(x)
print(logits.shape)  # -> torch.Size([4, 2])
```

### Extract patches from a 3D evidence volume

```python
import numpy as np
from mpm3d_lam import extract_patches

volume = np.random.randn(7, 100, 100, 100).astype(np.float32)  # (C, D, H, W)
centres = np.array([[50, 50, 50], [30, 40, 60]])
patches = extract_patches(volume, patch_size=(3, 3, 3), centres=centres)
print(patches.shape)  # -> (2, 7, 3, 3, 3)
```

## Running tests

```bash
python -m pytest tests/ -v
```
