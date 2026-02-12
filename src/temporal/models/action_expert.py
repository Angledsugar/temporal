"""Action Expert: Gemma-300M Transformer with Flow Matching.

Full PyTorch implementation of the pi0.5 action expert (Gemma-300M variant):
  - width=1024, depth=18, mlp_dim=4096, 8 heads (GQA, 1 KV head)
  - Conditional flow matching for action generation
  - Residual stream extraction at configurable intervention layer
  - JAX checkpoint loading for pretrained pi0.5-DROID weights

Architecture matches openpi/src/openpi/models/gemma.py configs[1].

After Phase 1 pretraining, theta is FROZEN permanently.
This is critical: co-training with MetaController causes
temporal abstractions to collapse (Kobayashi et al., 2025).
"""

from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ActionExpertConfig:
    """Configuration mirroring openpi gemma_300m."""

    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    controlled_layer: int = 9
    action_dim: int = 7  # canonical EE: xyz + quat(3) + gripper
    action_horizon: int = 50  # H = 50 steps per chunk


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------


class AdaRMSNorm(nn.Module):
    """Adaptive RMS normalization with optional conditioning."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self._norm(x.float()).type_as(x)
        if cond is None or self.dense is None:
            return normed * (1.0 + self.weight), None
        modulation = self.dense(cond)
        if len(x.shape) == 3 and len(modulation.shape) == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return normed * (1 + scale) + shift, gate


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        emb = torch.einsum("i,j->ij", t, freqs)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = x.shape[-2]
        cos = self.cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAttention(nn.Module):
    """Grouped-query attention (GQA) with RoPE."""

    def __init__(self, width: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(B, T, -1))


class GeGLU_FFN(nn.Module):
    """Gated GELU feed-forward network."""

    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    """Single transformer block with adaRMSNorm + GQA + GeGLU."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_dim: int,
        cond_dim: int,
    ):
        super().__init__()
        self.input_layernorm = AdaRMSNorm(width, cond_dim=cond_dim)
        self.self_attn = GQAttention(width, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = AdaRMSNorm(width, cond_dim=cond_dim)
        self.mlp = GeGLU_FFN(width, mlp_dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        normed, gate = self.input_layernorm(x, cond=cond)
        attn_out = self.self_attn(normed)
        x = x + attn_out * gate if gate is not None else x + attn_out
        normed, gate = self.post_attention_layernorm(x, cond=cond)
        ffn_out = self.mlp(normed)
        x = x + ffn_out * gate if gate is not None else x + ffn_out
        return x


# ---------------------------------------------------------------------------
# Action Expert (Gemma-300M)
# ---------------------------------------------------------------------------


class ActionExpert(nn.Module):
    """Gemma-300M action expert with flow matching and residual stream extraction.

    This is the core model trained in Phase 1 and frozen for all subsequent phases.

    Forward modes:
        1. Training (flow matching): predicts vector field v_theta(a^t, t | s, q)
        2. Inference (decode): iterative denoising to produce action chunks
        3. Residual extraction: forward through layer l, return hidden states

    Args:
        cfg: ActionExpertConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: ActionExpertConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ActionExpertConfig()
        self.cfg = cfg

        # Input/output projections
        self.action_in_proj = nn.Linear(cfg.action_dim, cfg.width)
        self.action_out_proj = nn.Linear(cfg.width, cfg.action_dim)
        self.proprio_proj = nn.Linear(14, cfg.width)  # proprio_dim=14

        # Time embedding (sinusoidal + MLP)
        self.time_mlp_in = nn.Linear(cfg.width, cfg.width)
        self.time_mlp_out = nn.Linear(cfg.width, cfg.width)

        # Text conditioning projection (from sentence embedding)
        self.text_proj = nn.Linear(384, cfg.width)  # sentence-transformers default dim

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.width,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.head_dim,
                    cfg.mlp_dim,
                    cond_dim=cfg.width,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.final_norm = AdaRMSNorm(cfg.width, cond_dim=cfg.width)

        # Internal state for residual stream extraction
        self._residual_stream: torch.Tensor | None = None
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to trade compute for memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _sinusoidal_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding matching pi0.5."""
        min_ts, max_ts = 4e-3, 4.0
        half = self.cfg.width // 2
        log_inc = math.log(max_ts / min_ts) / max(half - 1, 1)
        inv_ts = min_ts * torch.exp(
            torch.arange(half, device=timestep.device).float() * -log_inc
        )
        scaled = timestep.unsqueeze(-1) * inv_ts.unsqueeze(0)
        return torch.cat([scaled.sin(), scaled.cos()], dim=-1)

    def forward(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        text_embed: torch.Tensor | None = None,
        extract_residual: bool = False,
    ) -> torch.Tensor:
        """Forward pass: predict vector field for flow matching.

        Args:
            actions: (B, T, action_dim) noised actions a^(t).
            timestep: (B,) diffusion time in [0, 1].
            proprio: (B, T, 14) proprioceptive state (optional conditioning).
            text_embed: (B, embed_dim) text embedding (optional conditioning).
            extract_residual: If True, cache residual stream at intervention layer.

        Returns:
            predicted: (B, T, action_dim) predicted vector field.
        """
        x = self.action_in_proj(actions)

        # Add proprioception conditioning
        if proprio is not None:
            x = x + self.proprio_proj(proprio)

        # Build conditioning from time + text
        cond = None
        if timestep is not None:
            t_emb = self._sinusoidal_embedding(timestep)
            cond = F.silu(self.time_mlp_in(t_emb))
            cond = F.silu(self.time_mlp_out(cond))

        if text_embed is not None:
            text_cond = self.text_proj(text_embed)
            cond = cond + text_cond if cond is not None else text_cond

        # Transformer forward pass
        self._residual_stream = None
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cond, use_reentrant=False
                )
            else:
                x = layer(x, cond=cond)
            if extract_residual and i == self.cfg.controlled_layer:
                self._residual_stream = x.detach()

        normed, _ = self.final_norm(x, cond=cond)
        return self.action_out_proj(normed)

    def forward_up_to_layer(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        text_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through first `controlled_layer` blocks, return residual stream.

        Returns:
            e_l: (B, T, width) residual stream at intervention layer.
        """
        x = self.action_in_proj(actions)

        if proprio is not None:
            x = x + self.proprio_proj(proprio)

        cond = None
        if timestep is not None:
            t_emb = self._sinusoidal_embedding(timestep)
            cond = F.silu(self.time_mlp_in(t_emb))
            cond = F.silu(self.time_mlp_out(cond))

        if text_embed is not None:
            text_cond = self.text_proj(text_embed)
            cond = cond + text_cond if cond is not None else text_cond

        for i in range(self.cfg.controlled_layer + 1):
            x = self.layers[i](x, cond=cond)

        return x

    def forward_from_layer(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        text_embed: torch.Tensor | None = None,
        start_layer: int | None = None,
    ) -> torch.Tensor:
        """Continue forward pass from layer `start_layer` to final output.

        Used after MetaController applies additive control:
            x = x + U_t @ x

        Args:
            x: (B, T, width) hidden state to continue from.
            timestep: (B,) for conditioning.
            text_embed: (B, embed_dim) for conditioning.
            start_layer: Layer to resume from. Defaults to controlled_layer + 1.

        Returns:
            output: (B, T, action_dim) decoded actions.
        """
        if start_layer is None:
            start_layer = self.cfg.controlled_layer + 1

        cond = None
        if timestep is not None:
            t_emb = self._sinusoidal_embedding(timestep)
            cond = F.silu(self.time_mlp_in(t_emb))
            cond = F.silu(self.time_mlp_out(cond))

        if text_embed is not None:
            text_cond = self.text_proj(text_embed)
            cond = cond + text_cond if cond is not None else text_cond

        for i in range(start_layer, self.cfg.depth):
            x = self.layers[i](x, cond=cond)

        normed, _ = self.final_norm(x, cond=cond)
        return self.action_out_proj(normed)

    def get_residual_stream(self) -> torch.Tensor:
        """Return cached residual stream from last forward with extract_residual=True."""
        assert self._residual_stream is not None, (
            "No residual stream cached. Call forward() with extract_residual=True first."
        )
        return self._residual_stream

    @torch.no_grad()
    def decode_action(
        self,
        proprio: torch.Tensor,
        text_embed: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Flow matching inference: iterative denoising to produce action chunk.

        Args:
            proprio: (B, T, 14) proprioceptive state.
            text_embed: (B, embed_dim) text embedding.
            num_steps: Number of Euler steps for ODE integration.

        Returns:
            actions: (B, T, action_dim) decoded clean actions.
        """
        B, T, _ = proprio.shape
        device = proprio.device

        # Start from noise
        x = torch.randn(B, T, self.cfg.action_dim, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = torch.full((B,), step * dt, device=device)
            v = self.forward(x, timestep=t_val, proprio=proprio, text_embed=text_embed)
            x = x + v * dt

        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# JAX checkpoint loading (for pretrained pi0.5-DROID)
# ---------------------------------------------------------------------------


def load_from_jax_checkpoint(
    checkpoint_dir: str | Path,
    cfg: ActionExpertConfig | None = None,
    device: torch.device | str = "cpu",
) -> ActionExpert:
    """Load pi0.5-DROID JAX checkpoint and convert to PyTorch ActionExpert.

    Args:
        checkpoint_dir: Path to pi0.5-DROID checkpoint directory.
        cfg: Config override (default: ActionExpertConfig()).
        device: Target device.

    Returns:
        Frozen ActionExpert loaded with pretrained weights.
    """
    import orbax.checkpoint as ocp
    import jax
    from flax import traverse_util

    if cfg is None:
        cfg = ActionExpertConfig()

    checkpoint_dir = Path(checkpoint_dir)
    params_path = str(checkpoint_dir / "params")

    logger.info(f"Loading JAX checkpoint from {params_path}...")

    # Try multiple loading strategies
    params = None
    for strategy_name, strategy_fn in [
        ("PyTreeCheckpointer.restore", lambda: ocp.PyTreeCheckpointer().restore(params_path)),
        ("StandardCheckpointer", lambda: ocp.StandardCheckpointer().restore(params_path)),
    ]:
        try:
            raw = strategy_fn()
            params = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
            logger.info(f"Loaded with {strategy_name}")
            break
        except Exception as e:
            logger.warning(f"{strategy_name} failed: {e}")

    if params is None:
        raise RuntimeError(f"Could not load checkpoint from {params_path}")

    # Convert to float32 numpy
    def to_numpy_f32(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32) if x.dtype != np.float32 else x
        if hasattr(x, "__array__"):
            return np.asarray(x, dtype=np.float32)
        return x

    params = jax.tree.map(to_numpy_f32, params)

    # Strip 'value' suffix if present
    flat = traverse_util.flatten_dict(params)
    if flat and all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    params = traverse_util.unflatten_dict(flat)

    # Flatten PaliGemma params
    def flatten_dict_custom(d, parent_key="", sep="/"):
        items = {}
        for k, v in d.items():
            key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict_custom(v, key, sep))
            else:
                items[key] = v
        return items

    pali_flat = flatten_dict_custom(params["PaliGemma"])

    # Convert layer weights
    sd = {}
    p = "llm/layers"
    q_einsum = pali_flat[f"{p}/attn/q_einsum_1/w"]
    kv_einsum = pali_flat[f"{p}/attn/kv_einsum_1/w"]
    attn_vec = pali_flat[f"{p}/attn/attn_vec_einsum_1/w"]
    gating = pali_flat[f"{p}/mlp_1/gating_einsum"]
    linear = pali_flat[f"{p}/mlp_1/linear"]
    in_norm_k = pali_flat[f"{p}/pre_attention_norm_1/Dense_0/kernel"]
    in_norm_b = pali_flat[f"{p}/pre_attention_norm_1/Dense_0/bias"]
    ff_norm_k = pali_flat[f"{p}/pre_ffw_norm_1/Dense_0/kernel"]
    ff_norm_b = pali_flat[f"{p}/pre_ffw_norm_1/Dense_0/bias"]

    nh, hd, w = cfg.num_heads, cfg.head_dim, cfg.width

    for i in range(cfg.depth):
        lp = f"layers.{i}"
        sd[f"{lp}.self_attn.q_proj.weight"] = torch.from_numpy(
            q_einsum[i].transpose(0, 2, 1).reshape(nh * hd, w).copy()
        )
        sd[f"{lp}.self_attn.k_proj.weight"] = torch.from_numpy(
            kv_einsum[i, 0, 0].T.copy()
        )
        sd[f"{lp}.self_attn.v_proj.weight"] = torch.from_numpy(
            kv_einsum[i, 1, 0].T.copy()
        )
        sd[f"{lp}.self_attn.o_proj.weight"] = torch.from_numpy(
            attn_vec[i].reshape(nh * hd, w).T.copy()
        )
        sd[f"{lp}.mlp.gate_proj.weight"] = torch.from_numpy(gating[i, 0].T.copy())
        sd[f"{lp}.mlp.up_proj.weight"] = torch.from_numpy(gating[i, 1].T.copy())
        sd[f"{lp}.mlp.down_proj.weight"] = torch.from_numpy(linear[i].T.copy())
        sd[f"{lp}.input_layernorm.dense.weight"] = torch.from_numpy(in_norm_k[i].T.copy())
        sd[f"{lp}.input_layernorm.dense.bias"] = torch.from_numpy(in_norm_b[i].copy())
        sd[f"{lp}.post_attention_layernorm.dense.weight"] = torch.from_numpy(ff_norm_k[i].T.copy())
        sd[f"{lp}.post_attention_layernorm.dense.bias"] = torch.from_numpy(ff_norm_b[i].copy())

    # Final norm
    fn_k = pali_flat["llm/final_norm_1/Dense_0/kernel"]
    fn_b = pali_flat["llm/final_norm_1/Dense_0/bias"]
    sd["final_norm.dense.weight"] = torch.from_numpy(fn_k.T.copy())
    sd["final_norm.dense.bias"] = torch.from_numpy(fn_b.copy())

    # Input/output projections
    for name in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
        kernel = params[name]["kernel"]
        bias = params[name]["bias"]
        if isinstance(kernel, dict):
            kernel, bias = kernel["value"], bias["value"]
        sd[f"{name}.weight"] = torch.from_numpy(np.array(kernel).T.copy())
        sd[f"{name}.bias"] = torch.from_numpy(np.array(bias).copy())

    logger.info(f"Converted {len(sd)} weight tensors")

    # Build model and load
    expert = ActionExpert(cfg)
    missing, unexpected = expert.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        logger.info("All weights loaded successfully!")

    # Freeze
    expert = expert.to(device).eval()
    for param in expert.parameters():
        param.requires_grad = False

    del params, pali_flat, sd
    gc.collect()

    n_params = expert.param_count()
    logger.info(f"ActionExpert loaded: {n_params / 1e6:.1f}M parameters (frozen)")

    return expert
