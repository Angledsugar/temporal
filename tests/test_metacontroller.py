"""Tests for Metacontroller and SSM modules."""

import torch
import pytest

from internalrl.models.ssm import SSMBlock, SSMStack, LinearRecurrentUnit
from internalrl.models.metacontroller import (
    MetaController, InternalSequenceEmbedder,
    ControllerEncoder, SwitchingUnit, ControllerDecoder,
)
from internalrl.models.rl_policy import CausalSSMPolicy


class TestSSM:
    def test_lru_forward(self):
        lru = LinearRecurrentUnit(64, 128)
        x = torch.randn(2, 10, 64)
        y = lru(x)
        assert y.shape == (2, 10, 64)

    def test_lru_step(self):
        lru = LinearRecurrentUnit(64, 128)
        x = torch.randn(2, 64)
        y, h = lru.step(x)
        assert y.shape == (2, 64)
        assert h.shape == (2, 128)

    def test_ssm_block(self):
        block = SSMBlock(64, hidden_dim=64, mlp_dim=128)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_ssm_stack(self):
        stack = SSMStack(num_layers=2, embed_dim=64, hidden_dim=64)
        x = torch.randn(2, 10, 64)
        y = stack(x)
        assert y.shape == (2, 10, 64)


class TestMetacontrollerComponents:
    def test_sequence_embedder(self):
        emb = InternalSequenceEmbedder(embed_dim=64, output_dim=16)
        e = torch.randn(2, 10, 64)
        s = emb(e)
        assert s.shape == (2, 10, 16)

    def test_controller_encoder(self):
        enc = ControllerEncoder(embed_dim=64, seq_embed_dim=16, gru_dim=16, hidden_dim=32, latent_dim=4)
        e = torch.randn(2, 10, 64)
        s = torch.randn(2, 10, 16)
        mu, logvar, h = enc(e, s)
        assert mu.shape == (2, 10, 4)
        assert logvar.shape == (2, 10, 4)
        assert h.shape == (2, 10, 16)

    def test_switching_unit(self):
        sw = SwitchingUnit(embed_dim=64, gru_dim=16, latent_dim=4)
        e = torch.randn(2, 10, 64)
        h = torch.randn(2, 10, 16)
        z = torch.randn(2, 10, 4)
        beta = sw(e, h, z)
        assert beta.shape == (2, 10, 1)
        assert (beta >= 0).all() and (beta <= 1).all()

    def test_controller_decoder(self):
        dec = ControllerDecoder(latent_dim=4, embed_dim=64, rank=8, hidden_dim=16)
        z = torch.randn(2, 10, 4)
        B, A = dec(z)
        assert B.shape == (2, 10, 64, 8)
        assert A.shape == (2, 10, 8, 64)

    def test_controller_apply(self):
        dec = ControllerDecoder(latent_dim=4, embed_dim=64, rank=8, hidden_dim=16)
        e = torch.randn(2, 10, 64)
        z = torch.randn(2, 10, 4)
        e_prime = dec.apply_controller(e, z)
        assert e_prime.shape == (2, 10, 64)


class TestMetacontrollerFull:
    def test_forward(self):
        mc = MetaController(
            embed_dim=64, latent_dim=4, gru_dim=16,
            seq_embed_dim=16, encoder_hidden=32,
            decoder_hidden=16, controller_rank=8,
        )
        e = torch.randn(2, 10, 64)
        out = mc(e)
        assert out["e_controlled"].shape == (2, 10, 64)
        assert out["z_seq"].shape == (2, 10, 4)
        assert out["beta_seq"].shape == (2, 10, 1)
        assert out["mu"].shape == (2, 10, 4)
        assert out["logvar"].shape == (2, 10, 4)
        assert out["kl_loss"].ndim == 0  # scalar

    def test_gradients_flow(self):
        mc = MetaController(
            embed_dim=64, latent_dim=4, gru_dim=16,
            seq_embed_dim=16, encoder_hidden=32,
            decoder_hidden=16, controller_rank=8,
        )
        e = torch.randn(2, 10, 64, requires_grad=False)
        out = mc(e)
        loss = out["e_controlled"].sum() + out["kl_loss"]
        loss.backward()
        # Check that metacontroller params have gradients
        for name, p in mc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestRLPolicy:
    def test_forward(self):
        policy = CausalSSMPolicy(embed_dim=64, latent_dim=4, hidden_dim=64)
        e = torch.randn(2, 10, 64)
        mu, log_std = policy(e)
        assert mu.shape == (2, 10, 4)
        assert log_std.shape == (2, 10, 4)

    def test_step(self):
        policy = CausalSSMPolicy(embed_dim=64, latent_dim=4, hidden_dim=64)
        e = torch.randn(2, 64)
        mu, log_std, z, states = policy.step(e)
        assert mu.shape == (2, 4)
        assert z.shape == (2, 4)
        assert len(states) == 1  # 1-layer SSM

    def test_log_prob(self):
        policy = CausalSSMPolicy(embed_dim=64, latent_dim=4, hidden_dim=64)
        mu = torch.zeros(2, 4)
        log_std = torch.zeros(2, 4)
        z = torch.randn(2, 4)
        lp = policy.log_prob(mu, log_std, z)
        assert lp.shape == (2,)
        assert (lp <= 0).all()  # Log-probs should be non-positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
