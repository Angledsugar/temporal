"""Causal SSM policy for Internal RL (Stage 3).

Replaces the non-causal encoder from Stage 2 with a causal policy that
operates on the residual stream in real-time.

Architecture (Table A13):
- 1-layer SSM, embed_dim=256
- Outputs μ, σ for z ~ N(μ, σ²)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SSMBlock


class CausalSSMPolicy(nn.Module):
    """Causal policy for Internal RL operating in latent code space.

    Takes residual stream activations e_{t,l} as input and outputs
    latent codes z_t for controlling the base model.

    Args:
        embed_dim: Input dimension from residual stream (256).
        latent_dim: Output latent code dimension (8).
        hidden_dim: SSM hidden dimension (256).
        num_layers: Number of SSM layers (1).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # SSM backbone
        self.ssm_layers = nn.ModuleList([
            SSMBlock(embed_dim, hidden_dim=hidden_dim, mlp_dim=hidden_dim * 2)
            for _ in range(num_layers)
        ])

        # Output heads
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.log_std_head = nn.Linear(embed_dim, latent_dim)

        # Initialize log_std to give reasonable initial exploration
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(
        self, e_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process full sequence (for training batches).

        Args:
            e_seq: (B, T, embed_dim) residual stream sequence.

        Returns:
            mu: (B, T, latent_dim)
            log_std: (B, T, latent_dim)
        """
        x = e_seq
        for layer in self.ssm_layers:
            x = layer(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mu, log_std

    def step(
        self,
        e_t: torch.Tensor,
        states: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Single-step inference.

        Args:
            e_t: (B, embed_dim) current residual activation.
            states: SSM hidden states from previous step.

        Returns:
            mu: (B, latent_dim)
            log_std: (B, latent_dim)
            z: (B, latent_dim) sampled action
            new_states: Updated SSM states.
        """
        if states is None:
            states = [None] * len(self.ssm_layers)

        x = e_t
        new_states = []
        for layer, state in zip(self.ssm_layers, states):
            x, new_state = layer.step(x, state)
            new_states.append(new_state)

        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), -5.0, 2.0)

        # Sample z
        std = torch.exp(log_std)
        z = mu + std * torch.randn_like(std)

        return mu, log_std, z, new_states

    def log_prob(
        self, mu: torch.Tensor, log_std: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of z under the Gaussian policy.

        Args:
            mu: (B, [T,] latent_dim)
            log_std: (B, [T,] latent_dim)
            z: (B, [T,] latent_dim)

        Returns:
            log_prob: (B, [T,]) sum over latent dimensions.
        """
        std = torch.exp(log_std)
        var = std.pow(2)
        log_p = -0.5 * (
            math.log(2 * math.pi)
            + 2 * log_std
            + (z - mu).pow(2) / var
        )
        return log_p.sum(dim=-1)  # Sum over latent dims
