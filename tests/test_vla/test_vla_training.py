"""Tests for VLA metacontroller training with dummy residual data.

These tests verify the training pipeline without loading actual VLA models.
Uses DummyResidualDataset to bypass the VLA wrapper entirely.
"""

import pytest
import torch
import torch.nn as nn

from temporal.vla.models.metacontroller_vla import (
    VLAMetaController,
    VLAMetaControllerConfig,
)


class _DummyWrapper:
    """Minimal wrapper for training tests (no actual VLA model)."""

    def __init__(self, embed_dim=64, action_dim=7, action_horizon=4):
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self._residual = None

    def freeze_vlm(self):
        pass

    def extract_residual(self, batch):
        self._residual = batch["residual"]
        return self._residual

    def predict_with_controlled_residual(self, controlled, batch):
        # Simple projection to simulate action prediction
        B = controlled.shape[0]
        v_t = controlled[:, :self.action_horizon, :self.action_dim]
        u_t = batch["noise"][:, :self.action_horizon, :self.action_dim]
        loss = (u_t - v_t).pow(2)
        return loss, v_t

    @property
    def vlm_embed_dim(self):
        return self.embed_dim

    @property
    def controlled_layer(self):
        return 2

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32


class TestVLATrainingStep:
    """Test single training step mechanics."""

    @pytest.fixture
    def setup(self):
        config = VLAMetaControllerConfig(
            embed_dim=64,
            latent_dim=8,
            gru_dim=16,
            seq_embed_dim=16,
            encoder_hidden=32,
            decoder_hidden=16,
            controller_rank=8,
            batch_size=2,
            lr=1e-3,
        )
        mc = VLAMetaController(config)
        wrapper = _DummyWrapper(embed_dim=64, action_dim=7, action_horizon=4)
        optimizer = torch.optim.AdamW(mc.parameters(), lr=config.lr)
        return mc, wrapper, optimizer, config

    def test_loss_computes(self, setup):
        """Loss computation should succeed."""
        mc, wrapper, optimizer, config = setup
        batch = {
            "residual": torch.randn(2, 10, 64),
            "actions": torch.randn(2, 4, 7),
            "noise": torch.randn(2, 4, 7),
            "time": torch.rand(2),
        }

        residual = wrapper.extract_residual(batch)
        mc_out = mc(residual)
        action_loss, _ = wrapper.predict_with_controlled_residual(
            mc_out["e_controlled"], batch
        )

        total_loss = action_loss.mean() + config.kl_alpha * mc_out["kl_loss"]
        assert total_loss.dim() == 0
        assert not torch.isnan(total_loss)

    def test_backward_succeeds(self, setup):
        """Backward pass should produce gradients."""
        mc, wrapper, optimizer, config = setup
        batch = {
            "residual": torch.randn(2, 10, 64),
            "actions": torch.randn(2, 4, 7),
            "noise": torch.randn(2, 4, 7),
            "time": torch.rand(2),
        }

        residual = wrapper.extract_residual(batch)
        mc_out = mc(residual)
        action_loss, _ = wrapper.predict_with_controlled_residual(
            mc_out["e_controlled"], batch
        )

        total_loss = action_loss.mean() + config.kl_alpha * mc_out["kl_loss"]
        optimizer.zero_grad()
        total_loss.backward()

        grad_params = [p for p in mc.parameters() if p.grad is not None]
        assert len(grad_params) > 0, "No gradients produced"

    def test_optimizer_step(self, setup):
        """Optimizer step should update parameters."""
        mc, wrapper, optimizer, config = setup
        batch = {
            "residual": torch.randn(2, 10, 64),
            "actions": torch.randn(2, 4, 7),
            "noise": torch.randn(2, 4, 7),
            "time": torch.rand(2),
        }

        # Save initial params
        initial_params = {
            name: p.clone() for name, p in mc.named_parameters()
        }

        residual = wrapper.extract_residual(batch)
        mc_out = mc(residual)
        action_loss, _ = wrapper.predict_with_controlled_residual(
            mc_out["e_controlled"], batch
        )

        total_loss = action_loss.mean() + config.kl_alpha * mc_out["kl_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check at least some params changed
        changed = False
        for name, p in mc.named_parameters():
            if not torch.allclose(p, initial_params[name]):
                changed = True
                break
        assert changed, "No parameters were updated"


class TestVLATrainingLoss:
    """Test loss components."""

    def test_action_loss_is_mse(self):
        """Action loss should be MSE (not cross-entropy)."""
        wrapper = _DummyWrapper(embed_dim=64, action_dim=7, action_horizon=4)
        batch = {
            "residual": torch.randn(2, 10, 64),
            "noise": torch.randn(2, 4, 7),
        }

        controlled = torch.randn(2, 10, 64)
        loss, v_t = wrapper.predict_with_controlled_residual(controlled, batch)

        # MSE loss should be non-negative
        assert (loss >= 0).all()
        # Shape should match action dimensions
        assert v_t.shape == (2, 4, 7)

    def test_kl_loss_nonnegative(self):
        """KL loss should be non-negative."""
        config = VLAMetaControllerConfig(embed_dim=64, latent_dim=8)
        mc = VLAMetaController(config)
        e_seq = torch.randn(2, 10, 64)
        out = mc(e_seq)
        assert out["kl_loss"] >= 0
