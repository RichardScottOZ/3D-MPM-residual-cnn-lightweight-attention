"""3D patch extraction from gridded evidence data."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_patches(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    centres: np.ndarray,
) -> np.ndarray:
    """Extract fixed-size 3D patches centred on the given voxel coordinates.

    Parameters
    ----------
    volume:
        Evidence volume of shape ``(C, D, H, W)`` where *C* is the number of
        evidence channels.
    patch_size:
        ``(pD, pH, pW)`` — must be odd along each axis so the patch is
        symmetric around the centre voxel.
    centres:
        Integer voxel coordinates of shape ``(N, 3)`` giving ``(d, h, w)``
        for each sample.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, C, pD, pH, pW)``.

    Raises
    ------
    ValueError
        If any patch dimension is even, or if any centre is too close to the
        volume boundary to extract a full patch.
    """
    if volume.ndim != 4:
        raise ValueError(
            "volume must have shape (C, D, H, W), got %d dimensions" % volume.ndim
        )
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError(
            "centres must have shape (N, 3), got %s" % (centres.shape,)
        )

    pd, ph, pw = patch_size
    for dim_name, dim_val in zip(("pD", "pH", "pW"), patch_size):
        if dim_val < 1 or dim_val % 2 == 0:
            raise ValueError(
                "Patch dimension %s must be a positive odd integer, got %d"
                % (dim_name, dim_val)
            )

    c, vd, vh, vw = volume.shape
    half_d, half_h, half_w = pd // 2, ph // 2, pw // 2

    n = centres.shape[0]
    patches = np.empty((n, c, pd, ph, pw), dtype=volume.dtype)

    for i in range(n):
        d0, h0, w0 = int(centres[i, 0]), int(centres[i, 1]), int(centres[i, 2])

        d_lo, d_hi = d0 - half_d, d0 + half_d + 1
        h_lo, h_hi = h0 - half_h, h0 + half_h + 1
        w_lo, w_hi = w0 - half_w, w0 + half_w + 1

        if d_lo < 0 or h_lo < 0 or w_lo < 0:
            raise ValueError(
                "Centre %d at (%d, %d, %d) is too close to the lower volume "
                "boundary for patch_size %s" % (i, d0, h0, w0, patch_size)
            )
        if d_hi > vd or h_hi > vh or w_hi > vw:
            raise ValueError(
                "Centre %d at (%d, %d, %d) is too close to the upper volume "
                "boundary for patch_size %s" % (i, d0, h0, w0, patch_size)
            )

        patches[i] = volume[:, d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]

    return patches
