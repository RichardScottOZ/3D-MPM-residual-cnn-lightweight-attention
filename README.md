# 3D-MPM-residual-cnn-lightweight-attention

Minimal framework implementation of the paper:
**Three-dimensional mineral prospectivity mapping using a residual convolutional neural network with lightweight attention mechanisms** (Ore Geology Reviews, 2025).

## What is included

- `mpm3d_lam` Python package with:
  - DL4-style reference configuration (`1 conv + 2 ResBlocks + LAM`)
  - grid-scale generalisation utility that rescales patch sizes from a reference input size/cell size
- Focused unit tests validating scaling and configuration behaviour.

## Quick usage

```python
from mpm3d_lam import build_default_configuration

# Paper reference is 3x3x3 at 50m cell size.
# Build a configuration for another grid spacing:
config = build_default_configuration(target_cell_size=(25.0, 25.0, 25.0))
print(config.patch_size)  # -> (7, 7, 7)
```

This keeps approximately the same physical receptive field when grid resolution changes.
