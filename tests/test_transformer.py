"""Tests for Causal Transformer base model."""

import torch
import pytest

from temporal.models.transformer import CausalTransformer


class TestCausalTransformer:
    @pytest.fixture
    def model(self):
        return CausalTransformer(
            obs_dim=637, num_actions=4, embed_dim=64,
            num_layers=3, num_heads=2, head_dim=32, mlp_dim=128,
        )

    def test_forward_shape(self, model):
        obs = torch.randn(2, 10, 637)
        result = model(obs)
        assert result["action_logits"].shape == (2, 10, 4)
        assert result["obs_pred"].shape == (2, 10, 637)

    def test_forward_with_residuals(self, model):
        obs = torch.randn(2, 10, 637)
        result = model(obs, return_residuals=True)
        assert "residuals" in result
        # Should have layers 0, 1, 2, 3 (num_layers + 1 including input)
        assert len(result["residuals"]) == 4  # 0..3
        for l in range(4):
            assert result["residuals"][l].shape == (2, 10, 64)

    def test_split_forward(self, model):
        """forward_up_to + forward_from should match full forward."""
        obs = torch.randn(2, 10, 637)

        # Full forward
        full = model(obs)

        # Split at layer 2
        e, _ = model.forward_up_to_layer(obs, layer=2)
        assert e.shape == (2, 10, 64)

        result, _ = model.forward_from_layer(e, layer=2)
        assert result["action_logits"].shape == (2, 10, 4)

        # Results should match
        torch.testing.assert_close(
            result["action_logits"], full["action_logits"], atol=1e-5, rtol=1e-5
        )

    def test_intervention(self, model):
        """forward_with_intervention should work."""
        obs = torch.randn(2, 10, 637)
        controller = torch.randn(2, 10, 64) * 0.01  # Small perturbation
        result = model.forward_with_intervention(obs, controller, layer=2)
        assert result["action_logits"].shape == (2, 10, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
