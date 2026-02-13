"""Tests for the scaling module."""

import unittest

from mpm3d_lam import ModelReference, build_default_configuration, scale_patch_size


class ScalePatchSizeTests(unittest.TestCase):
    """Tests for scale_patch_size()."""

    def test_keeps_reference_size_for_reference_grid(self):
        self.assertEqual(
            scale_patch_size((3, 3, 3), (50, 50, 50), (50, 50, 50)),
            (3, 3, 3),
        )

    def test_scales_up_for_finer_grid(self):
        self.assertEqual(
            scale_patch_size((3, 3, 3), (50, 50, 50), (25, 25, 25)),
            (7, 7, 7),
        )

    def test_preserves_odd_and_minimum_for_coarser_grid(self):
        self.assertEqual(
            scale_patch_size((3, 3, 3), (50, 50, 50), (100, 100, 100)),
            (3, 3, 3),
        )

    def test_anisotropic_scaling(self):
        result = scale_patch_size((3, 3, 3), (50, 50, 50), (25, 50, 100))
        self.assertEqual(result, (7, 3, 3))

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            scale_patch_size((3, 3), (50, 50, 50), (25, 25, 25))

    def test_rejects_zero_cell_size(self):
        with self.assertRaises(ValueError):
            scale_patch_size((3, 3, 3), (50, 50, 50), (0, 25, 25))

    def test_rejects_negative_cell_size(self):
        with self.assertRaises(ValueError):
            scale_patch_size((3, 3, 3), (50, 50, 50), (-10, 25, 25))

    def test_rejects_zero_minimum_patch_size(self):
        with self.assertRaises(ValueError):
            scale_patch_size((3, 3, 3), (50, 50, 50), (25, 25, 25), minimum_patch_size=0)


class ConfigBuilderTests(unittest.TestCase):
    """Tests for build_default_configuration()."""

    def test_builds_dl4_style_defaults(self):
        reference = ModelReference()
        config = build_default_configuration((25.0, 50.0, 100.0), reference=reference)

        self.assertEqual(config.evidence_channels, 7)
        self.assertEqual(config.patch_size, (7, 3, 3))
        self.assertEqual(config.residual_blocks, 2)
        self.assertEqual(config.base_channels, 64)
        self.assertEqual(config.residual_channels, 128)
        self.assertEqual(config.lam_reduction_ratio, 8)
        self.assertEqual(config.output_classes, 2)
        self.assertAlmostEqual(config.dropout_rate, 0.5)

    def test_uses_paper_defaults_without_explicit_reference(self):
        config = build_default_configuration((50.0, 50.0, 50.0))
        self.assertEqual(config.evidence_channels, 7)
        self.assertEqual(config.patch_size, (3, 3, 3))

    def test_evidence_channels_override(self):
        config = build_default_configuration((50.0, 50.0, 50.0), evidence_channels=12)
        self.assertEqual(config.evidence_channels, 12)

    def test_output_classes_override(self):
        config = build_default_configuration((50.0, 50.0, 50.0), output_classes=5)
        self.assertEqual(config.output_classes, 5)


if __name__ == "__main__":
    unittest.main()
