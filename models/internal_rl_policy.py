"""Causal RL Policy for Phase 3 (Internal RL).

Replaces the non-causal BiGRU encoder from Phase 2.
Observes only past+present residual stream activations (causal).
Outputs controller codes z_t at switching points.

Phase 2 (training):  z_t depends on s(e_{1:T})  <-- future info included
Phase 3 (RL):        z_t = pi_psi(z_t | e_{1:t}) <-- causal, past only
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalRLPolicy(nn.Module):
    """Causal GRU policy for Internal RL.

    The policy is trained via policy gradient in the abstract
    controller-code space. Because the switching gate beta_t binarises
    decisions, the policy only acts at M << T switch points per episode,
    dramatically reducing gradient variance.

    Architecture:
        GRU(e_{1:t}) -> h_t -> mu_t, std_t, value_t
        z_t ~ N(mu_t, std_t^2)
    """

    def __init__(self, n_e: int = 1024, n_z: int = 32, hidden: int = 256):
        super().__init__()
        self.n_z = n_z
        self.hidden_size = hidden
        self.gru = nn.GRU(n_e, hidden, batch_first=True)
        self.mu_head = nn.Linear(hidden, n_z)
        self.logstd_head = nn.Linear(hidden, n_z)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, e_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch forward for training (processes full causal sequence).

        Args:
            e_seq: (B, T, n_e) -- causal sequence of residual stream.

        Returns:
            mu:    (B, T, n_z) -- mean of z_t distribution.
            std:   (B, T, n_z) -- std of z_t distribution.
            value: (B, T, 1)   -- state value estimate.
        """
        h_seq, _ = self.gru(e_seq)
        mu = self.mu_head(h_seq)
        logstd = self.logstd_head(h_seq).clamp(-5.0, 2.0)
        std = logstd.exp()
        value = self.value_head(h_seq)
        return mu, std, value

    def sample(
        self,
        e_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step sampling for online execution / rollout.

        Args:
            e_t:    (B, n_e) -- residual stream at current timestep.
            h_prev: (1, B, hidden) or None -- previous GRU hidden state.

        Returns:
            z_t:   (B, n_z)       -- sampled controller code.
            mu:    (B, n_z)       -- mean (for log_prob computation).
            std:   (B, n_z)       -- std (for log_prob computation).
            value: (B, 1)         -- state value estimate.
            h_new: (1, B, hidden) -- updated GRU hidden state.
        """
        if h_prev is None:
            h_prev = torch.zeros(
                1, e_t.shape[0], self.hidden_size, device=e_t.device
            )

        out, h_new = self.gru(e_t.unsqueeze(1), h_prev)
        out = out.squeeze(1)                       # (B, hidden)

        mu = self.mu_head(out)                     # (B, n_z)
        logstd = self.logstd_head(out).clamp(-5.0, 2.0)
        std = logstd.exp()                         # (B, n_z)
        z_t = mu + std * torch.randn_like(std)     # reparameterisation
        value = self.value_head(out)               # (B, 1)

        return z_t, mu, std, value, h_new

    def log_prob(
        self, z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of z under N(mu, std^2).

        Args:
            z:   (B, n_z)
            mu:  (B, n_z)
            std: (B, n_z)

        Returns:
            log_p: (B,) -- sum over n_z dimensions.
        """
        var = std.pow(2)
        log_p = -0.5 * (
            ((z - mu).pow(2) / var) + var.log() + torch.log(torch.tensor(2 * torch.pi))
        )
        return log_p.sum(dim=-1)
