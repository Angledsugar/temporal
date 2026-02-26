"""Hawk-style SSM (State Space Model) layer.

Used for:
1. Internal Sequence Embedder in the metacontroller (non-causal)
2. Internal RL policy (causal)

Architecture (Appendix D.1.1, Table A15):
- Linear Recurrent Unit (LRU) core with gating
- Pre-norm layer architecture
- MLP channel mixing with ReLU
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRecurrentUnit(nn.Module):
    """Linear Recurrent Unit (LRU) - core of Hawk SSM.

    Diagonal linear recurrence with input/output gating:
        h_t = a * h_{t-1} + b * x_t
        y_t = c * h_t

    where a is a learned complex diagonal (parameterized in log-space for stability).
    """

    def __init__(self, input_dim: int, hidden_dim: int, init_scale: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Gate
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Recurrence parameters (real-valued diagonal for simplicity)
        # a = sigmoid(log_a) ensures |a| < 1 for stability
        self.log_a = nn.Parameter(torch.randn(hidden_dim) * init_scale - 1.0)

        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        for proj in (self.input_proj, self.gate_proj, self.output_proj):
            nn.init.normal_(proj.weight, std=scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel scan over full sequence.

        Args:
            x: (B, T, input_dim)

        Returns:
            y: (B, T, input_dim)
        """
        B, T, _ = x.shape
        u = self.input_proj(x)          # (B, T, H)
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, H)

        a = torch.sigmoid(self.log_a)   # (H,) decay rates in (0, 1)

        # Sequential scan (simple implementation; can be parallelized with associative scan)
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = a * h + u[:, t]
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)  # (B, T, H)
        y = self.output_proj(h_seq * gate)   # (B, T, input_dim)
        return y

    def step(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step for causal inference.

        Args:
            x: (B, input_dim)
            h: (B, hidden_dim) previous hidden state

        Returns:
            y: (B, input_dim) output
            h_new: (B, hidden_dim) new hidden state
        """
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)

        u = self.input_proj(x)         # (B, H)
        gate = torch.sigmoid(self.gate_proj(x))
        a = torch.sigmoid(self.log_a)

        h_new = a * h + u
        y = self.output_proj(h_new * gate)
        return y, h_new


class SSMBlock(nn.Module):
    """Full SSM block: RMSNorm -> LRU -> residual -> RMSNorm -> MLP -> residual."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        mlp_dim: int = 512,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.lru = LinearRecurrentUnit(embed_dim, hidden_dim, init_scale)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence forward.

        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        x = x + self.lru(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(
        self, x: torch.Tensor, state: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Single step forward.

        Args:
            x: (B, embed_dim)
            state: Previous hidden state dict.

        Returns:
            output: (B, embed_dim)
            new_state: Updated state dict.
        """
        if state is None:
            state = {"lru_h": None}

        h = self.norm1(x)
        lru_out, new_h = self.lru.step(h, state.get("lru_h"))
        x = x + lru_out
        x = x + self.mlp(self.norm2(x))

        return x, {"lru_h": new_h}


class SSMStack(nn.Module):
    """Stack of SSM blocks.

    Args:
        num_layers: Number of SSM blocks.
        embed_dim: Embedding dimension.
        hidden_dim: LRU hidden dimension.
        mlp_dim: MLP hidden dimension.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        hidden_dim: int = 256,
        mlp_dim: int = 512,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SSMBlock(embed_dim, hidden_dim, mlp_dim, init_scale)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def step(
        self, x: torch.Tensor, states: list[dict] | None = None
    ) -> tuple[torch.Tensor, list[dict]]:
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer.step(x, state)
            new_states.append(new_state)
        return self.final_norm(x), new_states
