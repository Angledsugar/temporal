"""Metacontroller for unsupervised temporally-abstract action discovery.

Discovers temporal abstractions from the residual stream of a frozen
base autoregressive model. The metacontroller learns to:
1. Generate latent controller codes z_t
2. Determine when to switch (β_t switching gate)
3. Map codes to linear residual stream controllers U_t

Architecture (Appendix D.2, Figure 5, Tables A9/A10):
- Internal Sequence Embedder: non-causal SSM → s(e_{1:T,l})
- Controller Encoder: GRU + MLP → μ_t, Σ_t for z̃_t ~ N(μ,Σ)
- Switching Unit: MLP → β_t ∈ [0,1]
- Temporal Integration: z_t = β_t⊙z̃_t + (1-β_t)⊙z_{t-1}
- Controller Decoder: hypernetwork z_t → U_t (low-rank linear)
- Internal Controller: ê_{t,l} = e_{t,l} + U_t·e_{t,l}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SSMBlock


class InternalSequenceEmbedder(nn.Module):
    """Non-causal SSM that processes full residual stream sequence.

    Produces a per-timestep embedding s(e_{1:T,l}) that captures
    sequence-level (acausal) context.

    Args:
        embed_dim: Input embedding dimension (n_e = 256).
        output_dim: Sequence embedding dimension (n_s = 32).
    """

    def __init__(self, embed_dim: int = 256, output_dim: int = 32):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, output_dim)
        # Non-causal: we process forward then backward and combine
        self.forward_ssm = SSMBlock(output_dim, hidden_dim=output_dim, mlp_dim=output_dim * 2)
        self.backward_ssm = SSMBlock(output_dim, hidden_dim=output_dim, mlp_dim=output_dim * 2)
        self.proj_out = nn.Linear(output_dim * 2, output_dim)

    def forward(self, e_seq: torch.Tensor) -> torch.Tensor:
        """Process full residual stream sequence (non-causal).

        Args:
            e_seq: (B, T, embed_dim) residual stream at layer l.

        Returns:
            s: (B, T, output_dim) sequence embedding per timestep.
        """
        x = self.proj_in(e_seq)  # (B, T, output_dim)

        # Forward pass
        fwd = self.forward_ssm(x)
        # Backward pass
        bwd = self.backward_ssm(x.flip(dims=[1])).flip(dims=[1])

        # Combine
        combined = torch.cat([fwd, bwd], dim=-1)  # (B, T, 2*output_dim)
        return self.proj_out(combined)  # (B, T, output_dim)


class ControllerEncoder(nn.Module):
    """GRU-based encoder producing latent code proposals.

    At each timestep, combines:
    - Current residual activation e_{t,l}
    - GRU history state h_{t-1}
    - Sequence embedding s_t

    to produce μ_t, Σ_t for z̃_t ~ N(μ_t, Σ_t).

    Args:
        embed_dim: Residual stream dimension (n_e = 256).
        seq_embed_dim: Sequence embedding dimension (n_s = 32).
        gru_dim: GRU hidden dimension (n_h = 32).
        hidden_dim: MLP hidden dimension (64).
        latent_dim: Latent code dimension (n_z = 8).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        seq_embed_dim: int = 32,
        gru_dim: int = 32,
        hidden_dim: int = 64,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.gru_dim = gru_dim
        self.latent_dim = latent_dim

        # GRU for history summarization (Eq. 12)
        self.gru = nn.GRUCell(embed_dim, gru_dim)

        # MLP: concat(e_{t,l}, h_{t-1}, s_t) → μ, log_var
        input_dim = embed_dim + gru_dim + seq_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        e_seq: torch.Tensor,
        s_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process full sequence.

        Args:
            e_seq: (B, T, embed_dim) residual stream.
            s_seq: (B, T, seq_embed_dim) sequence embedding.

        Returns:
            mu: (B, T, latent_dim) means.
            logvar: (B, T, latent_dim) log-variances.
            h_seq: (B, T, gru_dim) GRU hidden states.
        """
        B, T, _ = e_seq.shape
        h = torch.zeros(B, self.gru_dim, device=e_seq.device, dtype=e_seq.dtype)

        mus, logvars, h_states = [], [], []
        for t in range(T):
            e_t = e_seq[:, t]   # (B, embed_dim)
            s_t = s_seq[:, t]   # (B, seq_embed_dim)

            # Store h before update (h_{t-1} for conditioning)
            h_prev = h

            # Update GRU
            h = self.gru(e_t, h)
            h_states.append(h)

            # Encode
            inp = torch.cat([e_t, h_prev, s_t], dim=-1)
            feat = self.mlp(inp)
            mus.append(self.mu_head(feat))
            logvars.append(self.logvar_head(feat))

        return (
            torch.stack(mus, dim=1),
            torch.stack(logvars, dim=1),
            torch.stack(h_states, dim=1),
        )

    def step(
        self,
        e_t: torch.Tensor,
        s_t: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step encoding.

        Returns:
            mu, logvar, h_new
        """
        B = e_t.shape[0]
        if h is None:
            h = torch.zeros(B, self.gru_dim, device=e_t.device, dtype=e_t.dtype)

        h_prev = h
        h_new = self.gru(e_t, h)
        inp = torch.cat([e_t, h_prev, s_t], dim=-1)
        feat = self.mlp(inp)
        mu = self.mu_head(feat)
        logvar = self.logvar_head(feat)
        return mu, logvar, h_new


class SwitchingUnit(nn.Module):
    """Produces temporal integration rate β_t ∈ [0, 1].

    Determines when to switch to a new abstract action.
    β_t ≈ 1 → switch to new controller; β_t ≈ 0 → keep current.

    Input: concat(e_{t,l}, h_{t-1}, z_{t-1})
    Output: β_t via sigmoid
    """

    def __init__(self, embed_dim: int = 256, gru_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        input_dim = embed_dim + gru_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        e_seq: torch.Tensor,
        h_seq: torch.Tensor,
        z_prev_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute β for full sequence.

        Args:
            e_seq: (B, T, embed_dim)
            h_seq: (B, T, gru_dim) - note: h_{t-1} should be used
            z_prev_seq: (B, T, latent_dim) - z_{t-1}

        Returns:
            beta: (B, T, 1) switching rates.
        """
        inp = torch.cat([e_seq, h_seq, z_prev_seq], dim=-1)
        return torch.sigmoid(self.net(inp))  # (B, T, 1)

    def step(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step switching decision.

        Returns:
            beta_t: (B, 1)
        """
        inp = torch.cat([e_t, h_t, z_prev], dim=-1)
        return torch.sigmoid(self.net(inp))


class ControllerDecoder(nn.Module):
    """Hypernetwork: maps latent code z_t to linear controller U_t.

    U_t = B_t @ A_t where B_t: (n_e, rank), A_t: (rank, n_e)
    This gives a low-rank (rank=16) linear controller.

    Args:
        latent_dim: Latent code dimension (8).
        embed_dim: Residual stream dimension (256).
        rank: Controller rank (16).
        hidden_dim: Decoder MLP hidden dimension (32).
    """

    def __init__(
        self,
        latent_dim: int = 8,
        embed_dim: int = 256,
        rank: int = 16,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        # z_t → B_t (n_e × rank) and A_t (rank × n_e)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.B_head = nn.Linear(hidden_dim, embed_dim * rank)
        self.A_head = nn.Linear(hidden_dim, rank * embed_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate low-rank controller matrices.

        Args:
            z: (B, [T,] latent_dim)

        Returns:
            B_mat: (B, [T,] embed_dim, rank)
            A_mat: (B, [T,] rank, embed_dim)
        """
        shape = z.shape[:-1]
        feat = self.net(z)
        B_flat = self.B_head(feat)
        A_flat = self.A_head(feat)
        B_mat = B_flat.view(*shape, self.embed_dim, self.rank)
        A_mat = A_flat.view(*shape, self.rank, self.embed_dim)
        return B_mat, A_mat

    def apply_controller(
        self, e: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Apply low-rank controller: e' = e + (B @ A) @ e.

        Args:
            e: (B, T, embed_dim) or (B, embed_dim) residual stream.
            z: (B, T, latent_dim) or (B, latent_dim) latent codes.

        Returns:
            e_prime: (B, T, embed_dim) or (B, embed_dim) controlled residual.
        """
        B_mat, A_mat = self.forward(z)

        if e.dim() == 3:
            # Batched: (B, T, embed_dim)
            # U @ e = B @ (A @ e)
            # A @ e: (B, T, rank, embed_dim) @ (B, T, embed_dim, 1) → (B, T, rank, 1)
            Ae = torch.einsum("...re,...e->...r", A_mat, e)  # (B, T, rank)
            Ue = torch.einsum("...er,...r->...e", B_mat, Ae)  # (B, T, embed_dim)
        else:
            Ae = torch.einsum("...re,...e->...r", A_mat, e)
            Ue = torch.einsum("...er,...r->...e", B_mat, Ae)

        return e + Ue


class MetaController(nn.Module):
    """Full metacontroller combining all components.

    Discovers temporally-abstract actions from frozen residual streams.
    Operates at a specific layer l of the base autoregressive model.

    Args:
        embed_dim: Base model embedding dimension (256).
        latent_dim: Latent code dimension (8).
        gru_dim: GRU hidden dimension (32).
        seq_embed_dim: Sequence embedding dimension (32).
        encoder_hidden: Encoder MLP hidden (64).
        decoder_hidden: Decoder MLP hidden (32).
        controller_rank: Low-rank controller rank (16).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 8,
        gru_dim: int = 32,
        seq_embed_dim: int = 32,
        encoder_hidden: int = 64,
        decoder_hidden: int = 32,
        controller_rank: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.sequence_embedder = InternalSequenceEmbedder(embed_dim, seq_embed_dim)
        self.encoder = ControllerEncoder(
            embed_dim, seq_embed_dim, gru_dim, encoder_hidden, latent_dim
        )
        self.switching_unit = SwitchingUnit(embed_dim, gru_dim, latent_dim)
        self.decoder = ControllerDecoder(latent_dim, embed_dim, controller_rank, decoder_hidden)

    def forward(
        self,
        e_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training (non-causal).

        Args:
            e_seq: (B, T, embed_dim) residual stream at controlled layer.

        Returns:
            Dictionary with:
                e_controlled: (B, T, embed_dim) controlled residual stream
                z_seq: (B, T, latent_dim) integrated latent codes
                beta_seq: (B, T, 1) switching rates
                mu: (B, T, latent_dim) encoder means
                logvar: (B, T, latent_dim) encoder log-variances
                kl_loss: scalar KL divergence loss
        """
        B, T, _ = e_seq.shape

        # 1. Sequence embedding (non-causal)
        s_seq = self.sequence_embedder(e_seq)  # (B, T, seq_embed_dim)

        # 2. Controller encoder: produce latent proposals
        mu, logvar, h_seq = self.encoder(e_seq, s_seq)  # each (B, T, ...)

        # 3. Sample z̃_t ~ N(μ_t, Σ_t)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_tilde = mu + std * eps  # (B, T, latent_dim)

        # 4. Switching unit: β_t
        # Use h_{t-1} for switching (shift h_seq by 1)
        h_prev = torch.cat([
            torch.zeros(B, 1, h_seq.shape[-1], device=e_seq.device, dtype=e_seq.dtype),
            h_seq[:, :-1],
        ], dim=1)

        # z_{t-1} for switching
        z_prev = torch.cat([
            torch.zeros(B, 1, self.latent_dim, device=e_seq.device, dtype=e_seq.dtype),
            z_tilde[:, :-1],  # Initially use z_tilde before integration
        ], dim=1)

        beta_seq = self.switching_unit(e_seq, h_prev, z_prev)  # (B, T, 1)

        # 5. Temporal integration: z_t = β_t⊙z̃_t + (1-β_t)⊙z_{t-1}
        z_list = []
        z_prev_integrated = torch.zeros(B, self.latent_dim, device=e_seq.device, dtype=e_seq.dtype)
        for t in range(T):
            beta_t = beta_seq[:, t]  # (B, 1)
            z_t = beta_t * z_tilde[:, t] + (1 - beta_t) * z_prev_integrated
            z_list.append(z_t)
            z_prev_integrated = z_t
        z_seq = torch.stack(z_list, dim=1)  # (B, T, latent_dim)

        # 6. Apply controller: e' = e + U @ e where U = decoder(z)
        e_controlled = self.decoder.apply_controller(e_seq, z_seq)

        # 7. KL divergence: D_KL(N(μ,Σ) || N(0,I))
        # Weighted by β_t (only active when switching)
        kl_per_t = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (B, T)
        # Weight KL by beta (continuous relaxation)
        kl_loss = (kl_per_t * beta_seq.squeeze(-1)).mean()

        return {
            "e_controlled": e_controlled,
            "z_seq": z_seq,
            "beta_seq": beta_seq,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl_loss,
            "z_tilde": z_tilde,
        }

    def get_decoder(self) -> ControllerDecoder:
        """Return the controller decoder (for Internal RL)."""
        return self.decoder

    def get_switching_unit(self) -> SwitchingUnit:
        """Return the switching unit (for Internal RL)."""
        return self.switching_unit
