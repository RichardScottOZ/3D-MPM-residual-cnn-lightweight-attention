import unittest

from mpm3d_lam import ModelReference, build_default_configuration, scale_patch_size


class ScalePatchSizeTests(unittest.TestCase):
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


class ConfigBuilderTests(unittest.TestCase):
    def test_builds_dl4_style_defaults(self):
        reference = ModelReference()
        config = build_default_configuration((25.0, 50.0, 100.0), reference=reference)

        self.assertEqual(config.evidence_channels, 7)
        self.assertEqual(config.patch_size, (7, 3, 3))
        self.assertEqual(config.residual_blocks, 2)
        self.assertEqual(config.base_channels, 64)
        self.assertEqual(config.residual_channels, 128)
        self.assertEqual(config.lam_reduction_channels, 8)


if __name__ == "__main__":
    unittest.main()
