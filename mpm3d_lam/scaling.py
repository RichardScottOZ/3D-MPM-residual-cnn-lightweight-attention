from __future__ import annotations

from typing import Iterable, Tuple

Int3 = Tuple[int, int, int]
Float3 = Tuple[float, float, float]


def _as_tuple3(values: Iterable[float], value_type: type) -> tuple:
    seq = tuple(values)
    if len(seq) != 3:
        raise ValueError("Expected exactly 3 values for a 3D shape/spacing.")
    return tuple(value_type(v) for v in seq)


def _to_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def scale_patch_size(
    reference_patch_size: Iterable[int],
    reference_cell_size: Iterable[float],
    target_cell_size: Iterable[float],
    minimum_patch_size: int = 3,
) -> Int3:
    """Scale a 3D patch size to preserve physical coverage across grid resolutions.

    The paper uses 3x3x3 samples at a 50x50x50 m grid. This function generalises
    that strategy to arbitrary grid spacing while keeping patch dimensions odd so
    each sample remains centered.
    """

    ref_patch = _as_tuple3(reference_patch_size, int)
    ref_cell = _as_tuple3(reference_cell_size, float)
    tgt_cell = _as_tuple3(target_cell_size, float)

    if minimum_patch_size < 1:
        raise ValueError("minimum_patch_size must be >= 1")

    scaled = []
    for patch, ref, tgt in zip(ref_patch, ref_cell, tgt_cell):
        if ref <= 0 or tgt <= 0:
            raise ValueError("Grid cell sizes must be positive.")
        estimate = int(round(patch * (ref / tgt)))
        estimate = max(minimum_patch_size, estimate)
        scaled.append(_to_odd(estimate))

    return tuple(scaled)  # type: ignore[return-value]
