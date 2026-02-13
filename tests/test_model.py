"""Tests for the RCNN-LAM model components."""

import unittest

import torch

from mpm3d_lam import (
    RCNNLAM,
    LightweightAttentionModule,
    ModelConfig,
    ResidualBlock3D,
    build_model,
)


class ResidualBlock3DTests(unittest.TestCase):
    """Tests for ResidualBlock3D."""

    def test_same_channels_identity_shortcut(self):
        block = ResidualBlock3D(64, 64)
        x = torch.randn(1, 64, 3, 3, 3)
        out = block(x)
        self.assertEqual(out.shape, (1, 64, 3, 3, 3))

    def test_different_channels_projection_shortcut(self):
        block = ResidualBlock3D(64, 128)
        x = torch.randn(1, 64, 3, 3, 3)
        out = block(x)
        self.assertEqual(out.shape, (1, 128, 3, 3, 3))

    def test_output_is_finite(self):
        block = ResidualBlock3D(32, 64)
        x = torch.randn(2, 32, 5, 5, 5)
        out = block(x)
        self.assertTrue(torch.isfinite(out).all())


class LightweightAttentionModuleTests(unittest.TestCase):
    """Tests for LightweightAttentionModule."""

    def test_output_shape_matches_input(self):
        lam = LightweightAttentionModule(128, reduction_ratio=8)
        x = torch.randn(2, 128, 3, 3, 3)
        out = lam(x)
        self.assertEqual(out.shape, x.shape)

    def test_reduction_ratio_one(self):
        lam = LightweightAttentionModule(16, reduction_ratio=1)
        x = torch.randn(1, 16, 3, 3, 3)
        out = lam(x)
        self.assertEqual(out.shape, x.shape)

    def test_rejects_invalid_reduction_ratio(self):
        with self.assertRaises(ValueError):
            LightweightAttentionModule(128, reduction_ratio=0)

    def test_scale_values_between_zero_and_one(self):
        """Sigmoid output should keep scale factors in [0, 1]."""
        lam = LightweightAttentionModule(64, reduction_ratio=4)
        x = torch.ones(1, 64, 3, 3, 3)
        out = lam(x)
        # Since input is all ones, each output channel should be in [0, 1]
        # (scale * 1.0 = scale, and scale is sigmoid output).
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())


class RCNNLAMModelTests(unittest.TestCase):
    """Integration tests for the full RCNN-LAM model."""

    def _make_config(self, **overrides):
        defaults = dict(
            evidence_channels=7,
            patch_size=(3, 3, 3),
            cell_size=(50.0, 50.0, 50.0),
            base_channels=32,
            residual_channels=64,
            residual_blocks=2,
            lam_reduction_ratio=4,
            output_classes=2,
            dropout_rate=0.0,
        )
        defaults.update(overrides)
        return ModelConfig(**defaults)

    def test_forward_shape(self):
        cfg = self._make_config()
        model = RCNNLAM(cfg)
        model.eval()
        x = torch.randn(4, 7, 3, 3, 3)
        out = model(x)
        self.assertEqual(out.shape, (4, 2))

    def test_forward_different_patch_size(self):
        cfg = self._make_config(patch_size=(7, 7, 7))
        model = RCNNLAM(cfg)
        model.eval()
        x = torch.randn(2, 7, 7, 7, 7)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))

    def test_multiclass_output(self):
        cfg = self._make_config(output_classes=5)
        model = RCNNLAM(cfg)
        model.eval()
        x = torch.randn(1, 7, 3, 3, 3)
        out = model(x)
        self.assertEqual(out.shape, (1, 5))

    def test_build_model_convenience(self):
        cfg = self._make_config()
        model = build_model(cfg)
        self.assertIsInstance(model, RCNNLAM)

    def test_gradients_flow(self):
        cfg = self._make_config(dropout_rate=0.0)
        model = RCNNLAM(cfg)
        model.train()
        x = torch.randn(2, 7, 3, 3, 3)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, msg=f"No gradient for {name}")


if __name__ == "__main__":
    unittest.main()
