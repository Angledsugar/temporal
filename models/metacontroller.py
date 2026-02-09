"""MetaController: the core innovation of TempoRAL.

Discovers subtask boundaries from the frozen action expert's residual
stream via a self-supervised variational objective (Phase 2), then
provides the low-rank controller decoder for Internal RL (Phase 3).

Architecture:
    Encoder (BiGRU)  -->  mu_t, sigma_t  -->  z_tilde_t
    Switching Unit   -->  beta_t ∈ [0,1]
    Temporal Integration: z_t = beta_t * z_tilde_t + (1 - beta_t) * z_{t-1}
    Decoder (Hypernetwork): z_t --> U_t = B_t @ A_t  (low-rank)
    Residual Control: e'_{t,l} = e_{t,l} + U_t @ e_{t,l}

Reference: Kobayashi et al., "Emergent temporal abstractions in
autoregressive models enable hierarchical reinforcement learning", 2025.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MetaControllerEncoder(nn.Module):
    """Bidirectional GRU encoder (non-causal, Phase 2 only).

    Processes the FULL sequence of residual-stream activations
    (including future information) to produce per-timestep latent
    statistics. Non-causal access is justified by the variational
    information-theoretic argument: conditioning on the future allows
    the encoder to discover boundaries that anticipate transitions.
    """

    def __init__(self, n_e: int = 1024, n_z: int = 32, hidden: int = 128):
        super().__init__()
        self.bigru = nn.GRU(n_e, hidden, bidirectional=True, batch_first=True)
        self.mu_head = nn.Linear(hidden * 2, n_z)
        self.logvar_head = nn.Linear(hidden * 2, n_z)

    def forward(
        self, e_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode full residual-stream sequence.

        Args:
            e_seq: (B, T, n_e) -- full sequence, non-causal.

        Returns:
            mu:     (B, T, n_z) -- mean of z_t distribution.
            logvar: (B, T, n_z) -- log-variance of z_t distribution.
            h_seq:  (B, T, hidden*2) -- encoder hidden states.
        """
        h_seq, _ = self.bigru(e_seq)
        mu = self.mu_head(h_seq)
        logvar = self.logvar_head(h_seq)
        return mu, logvar, h_seq


class SwitchingUnit(nn.Module):
    """Predicts beta_t: should we switch to a new controller code?

    beta_t ≈ 0: maintain previous controller (same subtask continues)
    beta_t ≈ 1: adopt new controller code (subtask boundary)

    Key finding from Kobayashi et al.: beta_t learns quasi-binary,
    sparse switching patterns WITHOUT explicit regularisation,
    aligned with ground-truth sub-goal changes.
    """

    def __init__(self, n_e: int = 1024, h_dim: int = 256, n_z: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_e + h_dim + n_z, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, e_t: torch.Tensor, h_t: torch.Tensor, z_prev: torch.Tensor
    ) -> torch.Tensor:
        """Compute switching probability.

        Args:
            e_t:    (B, n_e)  -- residual stream at time t.
            h_t:    (B, h_dim) -- encoder hidden state at time t.
            z_prev: (B, n_z)  -- previous controller code.

        Returns:
            beta_t: (B,) -- switching probability in [0, 1].
        """
        inp = torch.cat([e_t, h_t, z_prev], dim=-1)
        return self.net(inp).squeeze(-1)


class ControllerDecoder(nn.Module):
    """Decodes z_t -> low-rank controller U_t = B_t @ A_t.

    The controller modifies the residual stream additively:
        e'_{t,l} = e_{t,l} + U_t @ e_{t,l}
                 = e_{t,l} + B_t @ (A_t @ e_{t,l})

    Low-rank factorisation keeps parameters small (~2M)
    relative to the frozen 300M action expert.
    """

    def __init__(self, n_z: int = 32, n_e: int = 1024, rank: int = 32):
        super().__init__()
        self.n_e = n_e
        self.rank = rank
        self.proj_B = nn.Linear(n_z, n_e * rank)
        self.proj_A = nn.Linear(n_z, rank * n_e)

    def forward(
        self, z_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode controller code to low-rank matrices.

        Args:
            z_t: (B, n_z) -- controller code.

        Returns:
            B_mat: (B, n_e, rank)
            A_mat: (B, rank, n_e)
        """
        B = z_t.shape[0]
        B_mat = self.proj_B(z_t).view(B, self.n_e, self.rank)
        A_mat = self.proj_A(z_t).view(B, self.rank, self.n_e)
        return B_mat, A_mat

    def apply_control(
        self, e_t: torch.Tensor, z_t: torch.Tensor
    ) -> torch.Tensor:
        """Apply additive control to residual stream.

        e'_t = e_t + U_t @ e_t = e_t + B_t @ (A_t @ e_t)

        Args:
            e_t: (B, n_e) -- residual stream activation.
            z_t: (B, n_z) -- controller code.

        Returns:
            e_controlled: (B, n_e) -- controlled residual stream.
        """
        B_mat, A_mat = self.forward(z_t)
        Ae = torch.einsum("bri,bi->br", A_mat, e_t)    # (B, rank)
        BAe = torch.einsum("bor,br->bo", B_mat, Ae)     # (B, n_e)
        return e_t + BAe


class MetaController(nn.Module):
    """Full MetaController: encoder + switching unit + decoder.

    Phase 2: All components trained together (base model frozen).
    Phase 3: Only decoder is used (frozen); encoder replaced by causal RL policy.

    Loss (Phase 2):
        L(phi) = sum_t [ -ln p(a_t | o_{1:t}, z_{1:t})
                         + alpha * KL(N(mu_t, sigma_t^2) || N(0, I)) ]
    """

    def __init__(
        self,
        n_e: int = 1024,
        n_z: int = 32,
        rank: int = 32,
        encoder_hidden: int = 128,
    ):
        super().__init__()
        self.n_z = n_z
        self.encoder = MetaControllerEncoder(n_e, n_z, encoder_hidden)
        self.switch = SwitchingUnit(n_e, encoder_hidden * 2, n_z)
        self.decoder = ControllerDecoder(n_z, n_e, rank)

    def forward(
        self, e_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for Phase 2 training.

        Args:
            e_seq: (B, T, n_e) -- residual stream from frozen action expert.

        Returns:
            z_seq:    (B, T, n_z) -- controller codes per timestep.
            kl_loss:  scalar      -- mean KL divergence.
            beta_seq: (B, T)      -- switching probabilities.
        """
        B, T, _ = e_seq.shape
        mu, logvar, h_seq = self.encoder(e_seq)

        z_list: list[torch.Tensor] = []
        beta_list: list[torch.Tensor] = []
        kl_list: list[torch.Tensor] = []
        z_prev = torch.zeros(B, self.n_z, device=e_seq.device)

        for t in range(T):
            # Sample z_tilde from q(z_t | e_{1:T})
            std = torch.exp(0.5 * logvar[:, t])
            z_tilde = mu[:, t] + std * torch.randn_like(std)

            # KL divergence: KL(N(mu, sigma^2) || N(0, I))
            kl = -0.5 * (1 + logvar[:, t] - mu[:, t].pow(2) - logvar[:, t].exp())
            kl = kl.sum(dim=-1)  # (B,)

            # Switching gate
            beta = self.switch(e_seq[:, t], h_seq[:, t], z_prev)

            # Temporal integration
            z_t = beta.unsqueeze(-1) * z_tilde + (1 - beta.unsqueeze(-1)) * z_prev

            z_list.append(z_t)
            beta_list.append(beta)
            kl_list.append(kl)
            z_prev = z_t

        z_seq = torch.stack(z_list, dim=1)
        beta_seq = torch.stack(beta_list, dim=1)
        kl_loss = torch.stack(kl_list, dim=1).mean()

        return z_seq, kl_loss, beta_seq

    def compute_beta(
        self, e_t: torch.Tensor, z_prev: torch.Tensor
    ) -> torch.Tensor:
        """Compute switching probability for a single timestep.

        Used during Phase 3 (Internal RL) and deployment for
        runtime boundary detection.

        Args:
            e_t:    (B, n_e) -- current residual stream.
            z_prev: (B, n_z) -- current controller code.

        Returns:
            beta_t: (B,) -- switching probability.
        """
        # Use a zero hidden state as proxy (no encoder in Phase 3)
        h_dummy = torch.zeros(
            e_t.shape[0], self.switch.net[0].in_features - e_t.shape[-1] - z_prev.shape[-1],
            device=e_t.device,
        )
        return self.switch(e_t, h_dummy, z_prev)
