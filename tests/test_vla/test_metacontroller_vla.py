"""Tests for VLA metacontroller with scaled dimensions."""

import pytest
import torch

from internalrl.vla.models.metacontroller_vla import (
    VLAMetaController,
    VLAMetaControllerConfig,
)


@pytest.fixture
def small_config():
    """Small config for fast testing (64-dim instead of 2048)."""
    return VLAMetaControllerConfig(
        embed_dim=64,
        latent_dim=8,
        gru_dim=16,
        seq_embed_dim=16,
        encoder_hidden=32,
        decoder_hidden=16,
        controller_rank=8,
        controlled_layer=2,
    )


@pytest.fixture
def small_mc(small_config):
    return VLAMetaController(small_config)


class TestVLAMetaControllerShapes:
    """Test output shapes for various input dimensions."""

    def test_forward_shapes(self, small_mc, small_config):
        """MC output should have correct shapes."""
        B, T, D = 2, 20, small_config.embed_dim
        e_seq = torch.randn(B, T, D)
        out = small_mc(e_seq)

        assert out["e_controlled"].shape == (B, T, D)
        assert out["z_seq"].shape == (B, T, small_config.latent_dim)
        assert out["beta_seq"].shape == (B, T, 1)
        assert out["mu"].shape == (B, T, small_config.latent_dim)
        assert out["logvar"].shape == (B, T, small_config.latent_dim)
        assert out["z_tilde"].shape == (B, T, small_config.latent_dim)
        assert out["kl_loss"].dim() == 0  # scalar

    def test_batch_size_1(self, small_mc, small_config):
        """Should work with batch size 1."""
        e_seq = torch.randn(1, 10, small_config.embed_dim)
        out = small_mc(e_seq)
        assert out["e_controlled"].shape == (1, 10, small_config.embed_dim)

    def test_long_sequence(self, small_mc, small_config):
        """Should handle long sequences (200+ tokens)."""
        e_seq = torch.randn(1, 200, small_config.embed_dim)
        out = small_mc(e_seq)
        assert out["e_controlled"].shape == (1, 200, small_config.embed_dim)

    def test_full_2048_dim(self):
        """Test with full 2048-dim (as used in VLA)."""
        config = VLAMetaControllerConfig(
            embed_dim=2048,
            latent_dim=16,
            gru_dim=64,
            seq_embed_dim=64,
            encoder_hidden=128,
            decoder_hidden=64,
            controller_rank=32,
        )
        mc = VLAMetaController(config)
        e_seq = torch.randn(1, 10, 2048)
        out = mc(e_seq)
        assert out["e_controlled"].shape == (1, 10, 2048)
        assert out["z_seq"].shape == (1, 10, 16)


class TestVLAMetaControllerGradients:
    """Test gradient flow through metacontroller."""

    def test_gradients_flow(self, small_mc, small_config):
        """Gradients should flow through all MC parameters."""
        e_seq = torch.randn(2, 10, small_config.embed_dim)
        out = small_mc(e_seq)

        loss = out["e_controlled"].mean() + out["kl_loss"]
        loss.backward()

        for name, param in small_mc.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_kl_loss_has_grad(self, small_mc, small_config):
        """KL loss should be differentiable."""
        e_seq = torch.randn(2, 10, small_config.embed_dim)
        out = small_mc(e_seq)
        out["kl_loss"].backward()

        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in small_mc.parameters()
        )
        assert has_grad


class TestVLAMetaControllerBeta:
    """Test switching gate behavior."""

    def test_beta_range(self, small_mc, small_config):
        """Beta should be in [0, 1]."""
        e_seq = torch.randn(2, 20, small_config.embed_dim)
        out = small_mc(e_seq)
        assert (out["beta_seq"] >= 0).all()
        assert (out["beta_seq"] <= 1).all()

    def test_temporal_integration(self, small_mc, small_config):
        """z_seq should be a temporally-integrated version of z_tilde."""
        e_seq = torch.randn(1, 5, small_config.embed_dim)
        out = small_mc(e_seq)

        # z_seq should not be identical to z_tilde (integration smooths)
        assert not torch.allclose(out["z_seq"], out["z_tilde"], atol=1e-5)


class TestVLAMetaControllerAccessors:
    """Test decoder/switching unit accessors."""

    def test_get_decoder(self, small_mc):
        decoder = small_mc.get_decoder()
        assert decoder is not None
        assert hasattr(decoder, "apply_controller")

    def test_get_switching_unit(self, small_mc):
        switch = small_mc.get_switching_unit()
        assert switch is not None
        assert hasattr(switch, "step")
