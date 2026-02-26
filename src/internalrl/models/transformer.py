"""Causal Transformer for next-action prediction (Base model f_θ).

6-layer pre-norm Transformer with relative positional encoding.
Produces residual stream activations at every layer for metacontroller use.

Architecture (Appendix D.1.2, Table A16):
- Embedding dim: 256, Heads: 4, Head dim: 64
- MLP hidden: 512, ReLU activation
- Relative positional encoding with 32 buckets
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Compute relative position buckets (T5-style)."""
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    num_buckets //= 2
    relative_buckets += (relative_position > 0).long() * num_buckets
    relative_position = relative_position.abs()

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    relative_position_if_large = torch.clamp(
        relative_position_if_large, max=num_buckets - 1
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional bias."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        num_rel_pos_buckets: int = 32,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        self.rel_pos_bias = nn.Embedding(num_rel_pos_buckets, num_heads)
        self.num_rel_pos_buckets = num_rel_pos_buckets

        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, std=scale)

    def _compute_bias(self, T: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(T, device=device)
        relative_position = positions[:, None] - positions[None, :]
        buckets = _relative_position_bucket(
            relative_position, num_buckets=self.num_rel_pos_buckets
        )
        bias = self.rel_pos_bias(buckets)  # (T, T, num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # KV cache for autoregressive inference
        new_cache = None
        if kv_cache is not None:
            if "k" in kv_cache and kv_cache["k"] is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            new_cache = {"k": k, "v": v}

        T_k = k.shape[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T_k)

        # Relative positional bias
        bias = self._compute_bias(T_k, x.device)
        if T < T_k:
            bias = bias[:, :, -T:, :]
        attn = attn + bias

        # Causal mask
        if mask is None:
            causal = torch.triu(
                torch.ones(T, T_k, device=x.device, dtype=torch.bool), diagonal=T_k - T + 1
            )
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out), new_cache


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> MHA -> residual -> RMSNorm -> MLP -> residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        num_rel_pos_buckets: int = 32,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, head_dim, num_rel_pos_buckets, init_scale
        )
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self._init_mlp(init_scale)

    def _init_mlp(self, scale: float) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, kv_cache: dict | None = None
    ) -> tuple[torch.Tensor, dict | None]:
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class CausalTransformer(nn.Module):
    """6-layer Causal Transformer for gridworld action prediction.

    Input: observation sequence o_{1:T} of shape (B, T, obs_dim)
    Output: action logits (B, T, num_actions), obs prediction (B, T, obs_dim)
    Also returns residual stream activations e_{t,l} at every layer.

    The model supports:
    1. Full forward: processes entire sequence (for training)
    2. Split forward: forward_up_to_layer(l) + forward_from_layer(l) (for metacontroller)
    3. Step-by-step with KV cache (for Internal RL inference)
    """

    def __init__(
        self,
        obs_dim: int = 637,
        num_actions: int = 4,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        head_dim: int = 64,
        mlp_dim: int = 512,
        num_rel_pos_buckets: int = 32,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Input projection
        self.obs_proj = nn.Linear(obs_dim, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, head_dim, mlp_dim, num_rel_pos_buckets, init_scale
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.final_norm = nn.RMSNorm(embed_dim)
        self.action_head = nn.Linear(embed_dim, num_actions)
        self.obs_head = nn.Linear(embed_dim, obs_dim)

    def forward(
        self,
        obs_seq: torch.Tensor,
        return_residuals: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            obs_seq: (B, T, obs_dim) observation sequence.
            return_residuals: If True, return residual stream at each layer.

        Returns:
            Dictionary with action_logits, obs_pred, and optionally residuals.
        """
        x = self.obs_proj(obs_seq)  # (B, T, embed_dim)

        residuals = {}
        if return_residuals:
            residuals[0] = x.clone()

        for i, block in enumerate(self.blocks):
            x, _ = block(x)
            if return_residuals:
                residuals[i + 1] = x.clone()

        h = self.final_norm(x)
        action_logits = self.action_head(h)  # (B, T, num_actions)
        obs_pred = self.obs_head(h)          # (B, T, obs_dim)

        result = {"action_logits": action_logits, "obs_pred": obs_pred}
        if return_residuals:
            result["residuals"] = residuals
        return result

    def forward_up_to_layer(
        self,
        obs_seq: torch.Tensor,
        layer: int,
        kv_caches: list[dict] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Forward through layers 0..layer-1, returning residual stream at layer l.

        Args:
            obs_seq: (B, T, obs_dim)
            layer: Layer index to stop at (0 = after embedding, l = after block l-1).
            kv_caches: Optional list of KV caches per layer.

        Returns:
            residual_stream: (B, T, embed_dim) at the specified layer.
            new_kv_caches: Updated KV caches.
        """
        x = self.obs_proj(obs_seq)
        new_caches = []

        for i in range(layer):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = self.blocks[i](x, kv_cache=cache)
            new_caches.append(new_cache if new_cache is not None else {})

        return x, new_caches

    def forward_from_layer(
        self,
        residual: torch.Tensor,
        layer: int,
        kv_caches: list[dict] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict]]:
        """Forward from layer l through remaining layers to output.

        Args:
            residual: (B, T, embed_dim) residual stream at layer l.
            layer: Layer index to start from.
            kv_caches: Optional KV caches for layers >= layer.

        Returns:
            output: Dictionary with action_logits and obs_pred.
            new_kv_caches: Updated KV caches.
        """
        x = residual
        new_caches = []

        for i in range(layer, self.num_layers):
            cache_idx = i - layer
            cache = kv_caches[cache_idx] if kv_caches is not None else None
            x, new_cache = self.blocks[i](x, kv_cache=cache)
            new_caches.append(new_cache if new_cache is not None else {})

        h = self.final_norm(x)
        action_logits = self.action_head(h)
        obs_pred = self.obs_head(h)

        return {"action_logits": action_logits, "obs_pred": obs_pred}, new_caches

    def forward_with_intervention(
        self,
        obs_seq: torch.Tensor,
        controller: torch.Tensor,
        layer: int,
    ) -> dict[str, torch.Tensor]:
        """Forward with metacontroller intervention at a specific layer.

        Args:
            obs_seq: (B, T, obs_dim)
            controller: (B, T, embed_dim, embed_dim) or applied externally.
            layer: Layer at which to apply intervention.

        Returns:
            Dictionary with action_logits and obs_pred.
        """
        e, _ = self.forward_up_to_layer(obs_seq, layer)

        # Apply intervention: e' = e + U @ e
        # controller is (B, T, embed_dim) additive term (pre-computed U @ e)
        e_prime = e + controller

        result, _ = self.forward_from_layer(e_prime, layer)
        return result
