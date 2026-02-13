"""Tests for the patch extraction utility."""

import unittest

import numpy as np

from mpm3d_lam import extract_patches


class ExtractPatchesTests(unittest.TestCase):
    """Tests for extract_patches()."""

    def _make_volume(self, channels=3, depth=10, height=10, width=10):
        return np.arange(channels * depth * height * width, dtype=np.float32).reshape(
            channels, depth, height, width
        )

    def test_single_centre_patch(self):
        vol = self._make_volume()
        centres = np.array([[5, 5, 5]])
        patches = extract_patches(vol, (3, 3, 3), centres)
        self.assertEqual(patches.shape, (1, 3, 3, 3, 3))

    def test_multiple_centres(self):
        vol = self._make_volume()
        centres = np.array([[5, 5, 5], [3, 3, 3]])
        patches = extract_patches(vol, (3, 3, 3), centres)
        self.assertEqual(patches.shape, (2, 3, 3, 3, 3))

    def test_patch_content_matches_slice(self):
        vol = self._make_volume(channels=1, depth=10, height=10, width=10)
        centres = np.array([[5, 5, 5]])
        patches = extract_patches(vol, (3, 3, 3), centres)
        expected = vol[:, 4:7, 4:7, 4:7]
        np.testing.assert_array_equal(patches[0], expected)

    def test_rejects_even_patch_size(self):
        vol = self._make_volume()
        centres = np.array([[5, 5, 5]])
        with self.assertRaises(ValueError):
            extract_patches(vol, (2, 3, 3), centres)

    def test_rejects_boundary_violations_lower(self):
        vol = self._make_volume()
        centres = np.array([[0, 0, 0]])  # half-patch goes negative
        with self.assertRaises(ValueError):
            extract_patches(vol, (3, 3, 3), centres)

    def test_rejects_boundary_violations_upper(self):
        vol = self._make_volume(depth=5, height=5, width=5)
        centres = np.array([[4, 4, 4]])
        with self.assertRaises(ValueError):
            extract_patches(vol, (3, 3, 3), centres)

    def test_rejects_wrong_volume_ndim(self):
        vol = np.zeros((10, 10, 10))  # missing channel dim
        centres = np.array([[5, 5, 5]])
        with self.assertRaises(ValueError):
            extract_patches(vol, (3, 3, 3), centres)

    def test_rejects_wrong_centres_shape(self):
        vol = self._make_volume()
        centres = np.array([[5, 5]])  # only 2 coords
        with self.assertRaises(ValueError):
            extract_patches(vol, (3, 3, 3), centres)

    def test_larger_patch_size(self):
        vol = self._make_volume(depth=20, height=20, width=20)
        centres = np.array([[10, 10, 10]])
        patches = extract_patches(vol, (7, 7, 7), centres)
        self.assertEqual(patches.shape, (1, 3, 7, 7, 7))


if __name__ == "__main__":
    unittest.main()
