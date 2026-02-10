#!/usr/bin/env python3
"""Expert Distill: Train MetaController on frozen pi0.5-DROID residual stream.

Standalone script version of expert_distill_colab.ipynb for Docker execution.
Uses pi0.5-DROID pretrained checkpoint (frozen) as action expert,
and DROID droid_100 dataset for action sequences.

Hardware target: RTX 4090 (24GB VRAM), 128GB RAM
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Gemma-300M action expert (exact match with pi0.5)
    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256

    # Action space (DROID: 7 joint + 1 gripper = 8D, padded to 32D)
    action_dim: int = 32
    raw_action_dim: int = 8
    action_horizon: int = 15

    # Residual stream extraction
    intervention_layer: int = 9

    # MetaController
    n_z: int = 32
    rank: int = 32
    encoder_hidden: int = 128
    alpha: float = 0.05  # KL weight

    # Training (adjusted for RTX 4090 24GB)
    distill_steps: int = 5000
    distill_lr: float = 1e-3
    distill_batch_size: int = 16  # reduced from 32 for 24GB VRAM
    residual_extract_batch_size: int = 4  # conservative for extraction phase

    # Data processing
    subsequence_len: int = 50

    # Paths
    checkpoint_dir: str = "/checkpoint/pi05_droid"
    dataset_dir: str = "/dataset/droid_100"
    output_dir: str = "/output"
    log_every: int = 100
    save_every: int = 1000


# ---------------------------------------------------------------------------
# Pi0.5 Action Expert Architecture (Gemma-300M + adaRMSNorm)
# ---------------------------------------------------------------------------
class AdaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, cond_dim=None):
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

    def _norm(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, cond=None):
        normed = self._norm(x.float()).type_as(x)
        if cond is None or self.dense is None:
            return normed * (1.0 + self.weight), None
        modulation = self.dense(cond)
        if len(x.shape) == 3 and len(modulation.shape) == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return normed * (1 + scale) + shift, gate


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        emb = torch.einsum("i,j->ij", t, freqs)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, offset=0):
        T = x.shape[-2]
        cos = self.cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, width, num_heads, num_kv_heads, head_dim):
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

    def forward(self, x):
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
    def __init__(self, width, mlp_dim):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, width, num_heads, num_kv_heads, head_dim, mlp_dim, cond_dim):
        super().__init__()
        self.input_layernorm = AdaRMSNorm(width, cond_dim=cond_dim)
        self.self_attn = GQAttention(width, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = AdaRMSNorm(width, cond_dim=cond_dim)
        self.mlp = GeGLU_FFN(width, mlp_dim)

    def forward(self, x, cond=None):
        normed, gate = self.input_layernorm(x, cond=cond)
        attn_out = self.self_attn(normed)
        x = x + attn_out * gate if gate is not None else x + attn_out
        normed, gate = self.post_attention_layernorm(x, cond=cond)
        ffn_out = self.mlp(normed)
        x = x + ffn_out * gate if gate is not None else x + ffn_out
        return x


class Pi05ActionExpert(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.action_in_proj = nn.Linear(cfg.action_dim, cfg.width)
        self.action_out_proj = nn.Linear(cfg.width, cfg.action_dim)
        self.time_mlp_in = nn.Linear(cfg.width, cfg.width)
        self.time_mlp_out = nn.Linear(cfg.width, cfg.width)
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
        self._residual_stream = None

    def _sinusoidal_embedding(self, timestep):
        min_ts, max_ts = 4e-3, 4.0
        half = self.cfg.width // 2
        log_inc = math.log(max_ts / min_ts) / max(half - 1, 1)
        inv_ts = min_ts * torch.exp(
            torch.arange(half, device=timestep.device).float() * -log_inc
        )
        scaled = timestep.unsqueeze(-1) * inv_ts.unsqueeze(0)
        return torch.cat([scaled.sin(), scaled.cos()], dim=-1)

    def forward(self, actions, timestep=None, extract_residual=False):
        x = self.action_in_proj(actions)
        cond = None
        if timestep is not None:
            t_emb = self._sinusoidal_embedding(timestep)
            cond = F.silu(self.time_mlp_in(t_emb))
            cond = F.silu(self.time_mlp_out(cond))
        self._residual_stream = None
        for i, layer in enumerate(self.layers):
            x = layer(x, cond=cond)
            if extract_residual and i == self.cfg.intervention_layer:
                self._residual_stream = x.detach()
        normed, _ = self.final_norm(x, cond=cond)
        return self.action_out_proj(normed)

    def get_residual_stream(self):
        assert self._residual_stream is not None
        return self._residual_stream


# ---------------------------------------------------------------------------
# MetaController Architecture
# ---------------------------------------------------------------------------
class MetaControllerEncoder(nn.Module):
    def __init__(self, n_e, n_z, hidden):
        super().__init__()
        self.bigru = nn.GRU(n_e, hidden, bidirectional=True, batch_first=True)
        self.mu_head = nn.Linear(hidden * 2, n_z)
        self.logvar_head = nn.Linear(hidden * 2, n_z)

    def forward(self, e_seq):
        h_seq, _ = self.bigru(e_seq)
        return self.mu_head(h_seq), self.logvar_head(h_seq), h_seq


class SwitchingUnit(nn.Module):
    def __init__(self, n_e, h_dim, n_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_e + h_dim + n_z, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_t, h_t, z_prev):
        return self.net(torch.cat([e_t, h_t, z_prev], dim=-1)).squeeze(-1)


class ControllerDecoder(nn.Module):
    def __init__(self, n_z, n_e, rank):
        super().__init__()
        self.n_e = n_e
        self.rank = rank
        self.proj_B = nn.Linear(n_z, n_e * rank)
        self.proj_A = nn.Linear(n_z, rank * n_e)

    def apply_control(self, e_t, z_t):
        B_mat = self.proj_B(z_t).view(-1, self.n_e, self.rank)
        A_mat = self.proj_A(z_t).view(-1, self.rank, self.n_e)
        Ae = torch.einsum("bri,bi->br", A_mat, e_t)
        BAe = torch.einsum("bor,br->bo", B_mat, Ae)
        return e_t + BAe


class MetaController(nn.Module):
    def __init__(self, n_e, n_z, rank, encoder_hidden):
        super().__init__()
        self.n_z = n_z
        self.encoder = MetaControllerEncoder(n_e, n_z, encoder_hidden)
        self.switch = SwitchingUnit(n_e, encoder_hidden * 2, n_z)
        self.decoder = ControllerDecoder(n_z, n_e, rank)

    def forward(self, e_seq):
        B, T, _ = e_seq.shape
        mu, logvar, h_seq = self.encoder(e_seq)
        z_list, beta_list, kl_list = [], [], []
        z_prev = torch.zeros(B, self.n_z, device=e_seq.device)

        for t in range(T):
            std = torch.exp(0.5 * logvar[:, t])
            z_tilde = mu[:, t] + std * torch.randn_like(std)
            kl = -0.5 * (1 + logvar[:, t] - mu[:, t].pow(2) - logvar[:, t].exp())
            beta = self.switch(e_seq[:, t], h_seq[:, t], z_prev)
            z_t = beta.unsqueeze(-1) * z_tilde + (1 - beta.unsqueeze(-1)) * z_prev
            z_list.append(z_t)
            beta_list.append(beta)
            kl_list.append(kl.sum(dim=-1))
            z_prev = z_t

        z_seq = torch.stack(z_list, dim=1)
        beta_seq = torch.stack(beta_list, dim=1)
        kl_loss = torch.stack(kl_list, dim=1).mean()
        return z_seq, kl_loss, beta_seq


# ---------------------------------------------------------------------------
# Checkpoint Loading: JAX -> PyTorch
# ---------------------------------------------------------------------------
def load_expert_from_jax(checkpoint_dir: str, cfg: Config, device: torch.device):
    """Load pi0.5-DROID JAX checkpoint and convert to PyTorch."""
    import orbax.checkpoint as ocp
    import jax
    from flax import traverse_util

    params_path = f"{checkpoint_dir}/params/"
    logger.info(f"Loading JAX checkpoint from {params_path}...")

    if os.path.exists(params_path):
        contents = os.listdir(params_path)
        logger.info(f"  Directory contents ({len(contents)} items): {contents[:10]}...")

    params = None

    # Strategy 1: Direct restore
    try:
        ckptr = ocp.PyTreeCheckpointer()
        raw = ckptr.restore(params_path)
        params = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
        logger.info("Loaded with PyTreeCheckpointer.restore() (direct)")
    except Exception as e1:
        logger.warning(f"Strategy 1 failed: {e1}")
        try:
            ckptr = ocp.PyTreeCheckpointer()
            metadata = ckptr.metadata(params_path)
            tree_meta = (
                metadata.item_metadata
                if hasattr(metadata, "item_metadata")
                else metadata
            )
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), tree_meta
            )
            raw = ckptr.restore(params_path, restore_args=restore_args)
            params = (
                raw["params"] if isinstance(raw, dict) and "params" in raw else raw
            )
            logger.info("Loaded with metadata restore args")
        except Exception as e2:
            logger.warning(f"Strategy 2 failed: {e2}")
            try:
                ckptr = ocp.StandardCheckpointer()
                raw = ckptr.restore(params_path)
                params = (
                    raw["params"]
                    if isinstance(raw, dict) and "params" in raw
                    else raw
                )
                logger.info("Loaded with StandardCheckpointer")
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load checkpoint. Try: pip install orbax-checkpoint==0.6.4\n"
                    f"Errors: {e1} / {e2} / {e3}"
                )

    def to_numpy_f32(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32) if x.dtype != np.float32 else x
        if hasattr(x, "__array__"):
            return np.asarray(x, dtype=np.float32)
        return x

    params = jax.tree.map(to_numpy_f32, params)

    flat = traverse_util.flatten_dict(params)
    if flat and all(kp[-1] == "value" for kp in flat):
        logger.info("Stripping 'value' suffix from keys...")
        flat = {kp[:-1]: v for kp, v in flat.items()}
    params = traverse_util.unflatten_dict(flat)

    logger.info(f"Top-level keys: {list(params.keys())}")

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

    # Convert weights
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

    logger.info(f"Weight shapes - q: {q_einsum.shape}, kv: {kv_einsum.shape}")

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
        sd[f"{lp}.input_layernorm.dense.weight"] = torch.from_numpy(
            in_norm_k[i].T.copy()
        )
        sd[f"{lp}.input_layernorm.dense.bias"] = torch.from_numpy(
            in_norm_b[i].copy()
        )
        sd[f"{lp}.post_attention_layernorm.dense.weight"] = torch.from_numpy(
            ff_norm_k[i].T.copy()
        )
        sd[f"{lp}.post_attention_layernorm.dense.bias"] = torch.from_numpy(
            ff_norm_b[i].copy()
        )

    fn_k = pali_flat["llm/final_norm_1/Dense_0/kernel"]
    fn_b = pali_flat["llm/final_norm_1/Dense_0/bias"]
    sd["final_norm.dense.weight"] = torch.from_numpy(fn_k.T.copy())
    sd["final_norm.dense.bias"] = torch.from_numpy(fn_b.copy())

    for name in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
        kernel = params[name]["kernel"]
        bias = params[name]["bias"]
        if isinstance(kernel, dict):
            kernel, bias = kernel["value"], bias["value"]
        sd[f"{name}.weight"] = torch.from_numpy(np.array(kernel).T.copy())
        sd[f"{name}.bias"] = torch.from_numpy(np.array(bias).copy())

    logger.info(f"Converted {len(sd)} weight tensors")

    # Build model and load
    expert = Pi05ActionExpert(cfg)
    missing, unexpected = expert.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        logger.info("All weights loaded successfully!")

    expert = expert.to(device).eval()
    for p in expert.parameters():
        p.requires_grad = False

    del params, pali_flat, sd
    gc.collect()

    if torch.cuda.is_available():
        logger.info(f"Expert GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return expert


# ---------------------------------------------------------------------------
# DROID Data Loading
# ---------------------------------------------------------------------------
def load_droid_episodes(dataset_dir: str, cfg: Config):
    """Load DROID droid_100 episodes via tensorflow-datasets."""
    import tensorflow_datasets as tfds
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    logger.info(f"Loading DROID dataset from {dataset_dir}...")

    try:
        builder = tfds.builder_from_directory(dataset_dir)
        ds = builder.as_dataset(split="train")
        logger.info("Loaded with builder_from_directory")
    except Exception as e:
        logger.warning(f"builder_from_directory failed: {e}, trying tfds.load...")
        ds = tfds.load(
            "droid_100", data_dir=os.path.dirname(dataset_dir), split="train"
        )

    episodes_raw = []
    for episode in ds:
        actions = []
        for step in episode["steps"]:
            act = step["action"]
            if isinstance(act, dict):
                joint = act.get(
                    "joint_position", act.get("world_vector", tf.zeros(7))
                )
                gripper = act.get(
                    "gripper_position",
                    act.get("gripper_closedness_action", tf.zeros(1)),
                )
                a = tf.concat(
                    [tf.reshape(joint, [-1]), tf.reshape(gripper, [-1])], axis=0
                )
            else:
                a = act
            actions.append(a.numpy())
        if len(actions) > 10:
            episodes_raw.append(np.array(actions, dtype=np.float32))

    logger.info(
        f"Loaded {len(episodes_raw)} episodes, "
        f"lengths: {min(len(e) for e in episodes_raw)}-{max(len(e) for e in episodes_raw)}"
    )

    # Quantile normalization + padding to 32D
    all_actions_raw = np.concatenate(episodes_raw, axis=0)
    q01 = np.percentile(all_actions_raw, 1, axis=0)
    q99 = np.percentile(all_actions_raw, 99, axis=0)

    episodes = []
    for ep in episodes_raw:
        range_ = np.maximum(q99 - q01, 1e-6)
        normed = np.clip((ep - q01) / range_ * 2 - 1, -1, 1)
        raw_dim = normed.shape[-1]
        if raw_dim < cfg.action_dim:
            pad_width = [(0, 0)] * (normed.ndim - 1) + [
                (0, cfg.action_dim - raw_dim)
            ]
            normed = np.pad(normed, pad_width, mode="constant", constant_values=0)
        episodes.append(normed.astype(np.float32))

    logger.info(f"Processed {len(episodes)} episodes, action_dim={episodes[0].shape[1]}")
    return episodes


# ---------------------------------------------------------------------------
# Residual Stream Extraction
# ---------------------------------------------------------------------------
def extract_residual_streams(
    expert: Pi05ActionExpert,
    episodes: list[np.ndarray],
    cfg: Config,
    device: torch.device,
):
    """Extract residual streams from frozen expert for all episodes."""
    all_streams = []
    horizon = cfg.action_horizon
    batch_size = cfg.residual_extract_batch_size

    logger.info(
        f"Extracting residual streams (horizon={horizon}, batch={batch_size})..."
    )
    start = time.time()

    with torch.no_grad():
        for ep_idx, ep in enumerate(episodes):
            T = len(ep)
            if T < horizon:
                continue

            windows = np.array(
                [ep[t : t + horizon] for t in range(T - horizon + 1)],
                dtype=np.float32,
            )

            ep_streams = []
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
                t_val = torch.zeros(batch.shape[0], device=device)
                expert(batch, timestep=t_val, extract_residual=True)
                residual = expert.get_residual_stream()
                pooled = residual.mean(dim=1).cpu().numpy()
                ep_streams.append(pooled)

            ep_streams = np.concatenate(ep_streams, axis=0)
            all_streams.append(ep_streams)

            if (ep_idx + 1) % 20 == 0:
                logger.info(f"  {ep_idx+1}/{len(episodes)} episodes processed")

    elapsed = time.time() - start
    total_frames = sum(len(s) for s in all_streams)
    logger.info(
        f"Extracted {len(all_streams)} episode streams ({total_frames} frames) "
        f"in {elapsed:.1f}s"
    )
    return all_streams


# ---------------------------------------------------------------------------
# Approximate Boundaries (evaluation reference only)
# ---------------------------------------------------------------------------
def derive_approximate_boundaries(
    episodes: list[np.ndarray], cfg: Config, vel_threshold=0.3
):
    horizon = cfg.action_horizon
    all_boundaries = []

    for ep in episodes:
        T = len(ep)
        if T < horizon:
            continue
        effective_T = T - horizon + 1
        joints = ep[:effective_T, :7]
        gripper = ep[:effective_T, 7]

        joint_vel = np.zeros(effective_T)
        joint_vel[1:] = np.linalg.norm(np.diff(joints, axis=0), axis=1)
        gripper_change = np.zeros(effective_T)
        gripper_change[1:] = np.abs(np.diff(gripper))

        vel_norm = joint_vel / (np.max(joint_vel) + 1e-8)
        grip_norm = gripper_change / (np.max(gripper_change) + 1e-8)
        boundary_score = 0.5 * vel_norm + 0.5 * grip_norm
        boundaries = (boundary_score > vel_threshold).astype(np.float32)

        min_gap = 5
        cleaned = np.zeros_like(boundaries)
        last_b = -min_gap
        for t in range(len(boundaries)):
            if boundaries[t] > 0 and (t - last_b) >= min_gap:
                cleaned[t] = 1.0
                last_b = t
        all_boundaries.append(cleaned)

    return all_boundaries


# ---------------------------------------------------------------------------
# Training Data Preparation
# ---------------------------------------------------------------------------
def create_training_data(residual_streams, episodes, approx_boundaries, cfg):
    seq_len = cfg.subsequence_len
    horizon = cfg.action_horizon

    all_residuals, all_actions, all_bounds = [], [], []

    for ep_idx in range(len(residual_streams)):
        rs = residual_streams[ep_idx]
        ep = episodes[ep_idx]
        bd = approx_boundaries[ep_idx]
        T = len(rs)

        stride = seq_len // 2
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            all_residuals.append(rs[start:end])
            all_actions.append(ep[start:end, : cfg.action_dim])
            all_bounds.append(bd[start:end])

    residuals_tensor = torch.from_numpy(np.array(all_residuals, dtype=np.float32))
    actions_tensor = torch.from_numpy(np.array(all_actions, dtype=np.float32))
    bounds_tensor = torch.from_numpy(np.array(all_bounds, dtype=np.float32))

    logger.info(
        f"Training subsequences: {len(residuals_tensor)}, "
        f"shape: {residuals_tensor.shape}"
    )
    return residuals_tensor, actions_tensor, bounds_tensor


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_expert_distill(
    meta: MetaController,
    action_decoder: nn.Module,
    res_data: torch.Tensor,
    act_data: torch.Tensor,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
):
    optimizer = torch.optim.AdamW(
        list(meta.parameters()) + list(action_decoder.parameters()),
        lr=cfg.distill_lr,
        weight_decay=0.01,
    )

    meta.train()
    action_decoder.train()

    history = {
        "total": [],
        "recon": [],
        "kl": [],
        "beta_mean": [],
        "beta_sparsity": [],
    }
    start = time.time()
    N = len(res_data)

    logger.info(
        f"Starting Expert Distill training: {cfg.distill_steps} steps, "
        f"batch={cfg.distill_batch_size}, lr={cfg.distill_lr}, alpha={cfg.alpha}"
    )

    for step in range(cfg.distill_steps):
        idx = torch.randint(0, N, (cfg.distill_batch_size,))
        e_seq = res_data[idx].to(device)
        target = act_data[idx].to(device)

        z_seq, kl_loss, beta_seq = meta(e_seq)

        B, T, n_e = e_seq.shape
        e_flat = e_seq.reshape(B * T, n_e)
        z_flat = z_seq.reshape(B * T, cfg.n_z)
        e_controlled = meta.decoder.apply_control(e_flat, z_flat)
        pred = action_decoder(e_controlled).reshape(B, T, cfg.action_dim)

        recon_loss = F.mse_loss(pred, target)
        total_loss = recon_loss + cfg.alpha * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta.parameters(), 1.0)
        optimizer.step()

        history["total"].append(total_loss.item())
        history["recon"].append(recon_loss.item())
        history["kl"].append(kl_loss.item())
        history["beta_mean"].append(beta_seq.mean().item())
        history["beta_sparsity"].append((beta_seq > 0.5).float().mean().item())

        if (step + 1) % cfg.log_every == 0:
            logger.info(
                f"Step {step+1}/{cfg.distill_steps} | "
                f"Total: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                f"KL: {kl_loss.item():.4f} | beta_mean: {beta_seq.mean().item():.3f} | "
                f"beta>0.5: {(beta_seq > 0.5).float().mean().item():.3f} | "
                f"{time.time()-start:.0f}s"
            )

        if (step + 1) % cfg.save_every == 0:
            ckpt_path = output_dir / f"metacontroller_step{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "metacontroller_state_dict": meta.state_dict(),
                    "action_decoder_state_dict": action_decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    elapsed = time.time() - start
    logger.info(f"Expert Distill complete in {elapsed:.0f}s")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_and_save_plots(
    meta: MetaController,
    res_data: torch.Tensor,
    act_data: torch.Tensor,
    bnd_data: torch.Tensor,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
):
    from sklearn.metrics import normalized_mutual_info_score

    meta.eval()

    # Beta pattern visualization
    with torch.no_grad():
        e_seq = res_data[:5].to(device)
        _, _, beta_seq = meta(e_seq)
        beta_np = beta_seq.cpu().numpy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for i, ax in enumerate(axes):
        t = np.arange(cfg.subsequence_len)
        ax.plot(t, beta_np[i], "b-", linewidth=1.5, alpha=0.8, label="Learned beta_t")
        ax.fill_between(t, 0, beta_np[i], alpha=0.2, color="blue")
        gt = bnd_data[i].numpy()
        for idx in np.where(gt > 0.5)[0]:
            ax.axvline(x=idx, color="red", linestyle="--", alpha=0.7, linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"Seq {i}")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Learned beta_t (blue) vs Approximate Boundaries (red)")
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "beta_patterns.png", dpi=150)
    plt.close()
    logger.info(f"Saved beta_patterns.png")

    # Metrics
    all_nmi, all_f1, all_tcr = [], [], []
    tolerance = 3

    with torch.no_grad():
        for i in range(0, len(res_data), 64):
            e_seq = res_data[i : i + 64].to(device)
            gt_batch = bnd_data[i : i + 64]
            _, _, beta_seq = meta(e_seq)
            beta_np = beta_seq.cpu().numpy()
            gt_np = gt_batch.numpy()

            for j in range(len(beta_np)):
                pred_b = (beta_np[j] > 0.5).astype(float)
                gt_b = (gt_np[j] > 0.5).astype(float)
                nmi = normalized_mutual_info_score(gt_b, pred_b)
                all_nmi.append(nmi)

                pred_idx = set(np.where(pred_b > 0.5)[0])
                gt_idx = set(np.where(gt_b > 0.5)[0])
                tp = sum(
                    1
                    for p_val in pred_idx
                    if any(abs(p_val - g) <= tolerance for g in gt_idx)
                )
                prec = tp / max(len(pred_idx), 1)
                rec = tp / max(len(gt_idx), 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                all_f1.append(f1)
                n_sw = max(int(pred_b.sum()), 1)
                all_tcr.append(len(pred_b) / n_sw)

    metrics_str = (
        f"{'='*50}\n"
        f"Expert Distill Evaluation (Real pi0.5-DROID)\n"
        f"{'='*50}\n"
        f"Switching NMI:         {np.mean(all_nmi):.4f} +/- {np.std(all_nmi):.4f}\n"
        f"Boundary F1 (tol={tolerance}):  {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}\n"
        f"Temporal Contraction:   {np.mean(all_tcr):.1f}x +/- {np.std(all_tcr):.1f}x\n"
        f"Avg switches/seq:       {cfg.subsequence_len / np.mean(all_tcr):.1f}\n"
        f"Avg GT boundaries/seq:  {bnd_data.sum(dim=1).mean():.1f}"
    )
    logger.info(f"\n{metrics_str}")

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(metrics_str)

    # Beta distribution (quasi-binary check)
    with torch.no_grad():
        e_seq = res_data[: min(200, len(res_data))].to(device)
        _, _, beta_seq = meta(e_seq)
        all_betas = beta_seq.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_betas, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[0].set_xlabel("beta_t")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of beta_t")
    axes[0].legend()

    near_0 = (all_betas < 0.1).mean() * 100
    near_1 = (all_betas > 0.9).mean() * 100
    middle = ((all_betas >= 0.1) & (all_betas <= 0.9)).mean() * 100

    axes[1].bar(
        ["near 0\n(< 0.1)", "middle\n(0.1-0.9)", "near 1\n(> 0.9)"],
        [near_0, middle, near_1],
        color=["#3498db", "#e74c3c", "#2ecc71"],
        edgecolor="black",
    )
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Quasi-binary check")
    plt.suptitle("Beta_t Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "beta_distribution.png", dpi=150)
    plt.close()

    quasi_binary = "YES" if middle < 30 else "NOT YET"
    logger.info(
        f"Beta distribution: near_0={near_0:.1f}%, middle={middle:.1f}%, "
        f"near_1={near_1:.1f}% -> Quasi-binary: {quasi_binary}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Expert Distill (Docker)")
    parser.add_argument("--checkpoint-dir", default="/checkpoint/pi05_droid")
    parser.add_argument("--dataset-dir", default="/dataset/droid_100")
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--distill-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--subsequence-len", type=int, default=50)
    args = parser.parse_args()

    cfg = Config(
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        distill_steps=args.distill_steps,
        distill_batch_size=args.batch_size,
        distill_lr=args.lr,
        alpha=args.alpha,
        subsequence_len=args.subsequence_len,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )

    # 1. Load frozen expert
    logger.info("=" * 60)
    logger.info("Phase 1: Loading pi0.5-DROID expert (frozen)")
    logger.info("=" * 60)
    expert = load_expert_from_jax(cfg.checkpoint_dir, cfg, device)

    # 2. Load DROID data
    logger.info("=" * 60)
    logger.info("Phase 2: Loading DROID droid_100 dataset")
    logger.info("=" * 60)
    episodes = load_droid_episodes(cfg.dataset_dir, cfg)

    # 3. Extract residual streams
    logger.info("=" * 60)
    logger.info("Phase 3: Extracting residual streams")
    logger.info("=" * 60)
    residual_streams = extract_residual_streams(expert, episodes, cfg, device)

    # Free expert from GPU after extraction
    del expert
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Freed expert from GPU memory")

    # 4. Derive approximate boundaries (for evaluation)
    approx_boundaries = derive_approximate_boundaries(episodes, cfg)

    # 5. Create training data
    logger.info("=" * 60)
    logger.info("Phase 4: Preparing training data")
    logger.info("=" * 60)
    res_data, act_data, bnd_data = create_training_data(
        residual_streams, episodes, approx_boundaries, cfg
    )

    # 6. Build MetaController
    logger.info("=" * 60)
    logger.info("Phase 5: Training MetaController")
    logger.info("=" * 60)
    meta = MetaController(
        n_e=cfg.width, n_z=cfg.n_z, rank=cfg.rank, encoder_hidden=cfg.encoder_hidden
    ).to(device)
    action_decoder = nn.Linear(cfg.width, cfg.action_dim).to(device)

    n_meta = sum(p.numel() for p in meta.parameters())
    logger.info(f"MetaController: {n_meta / 1e6:.2f}M parameters")

    # 7. Train
    history = train_expert_distill(
        meta, action_decoder, res_data, act_data, cfg, device, output_dir
    )

    # Save final checkpoint
    torch.save(
        {
            "metacontroller_state_dict": meta.state_dict(),
            "action_decoder_state_dict": action_decoder.state_dict(),
            "history": history,
            "config": vars(cfg),
        },
        output_dir / "metacontroller_final.pt",
    )
    logger.info(f"Saved final checkpoint: {output_dir / 'metacontroller_final.pt'}")

    # Save training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    smooth = lambda x, w=50: np.convolve(x, np.ones(w) / w, mode="valid")
    for ax, key, title in [
        (axes[0, 0], "total", "Total Loss"),
        (axes[0, 1], "recon", "Reconstruction Loss"),
        (axes[1, 0], "kl", "KL Divergence"),
        (axes[1, 1], "beta_sparsity", "Beta Sparsity (frac > 0.5)"),
    ]:
        ax.plot(history[key], alpha=0.3, color="blue")
        if len(history[key]) > 50:
            ax.plot(smooth(history[key]), "r-", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    plt.suptitle("Expert Distill Training (pi0.5-DROID)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    logger.info("Saved training_curves.png")

    # 8. Evaluate
    logger.info("=" * 60)
    logger.info("Phase 6: Evaluation")
    logger.info("=" * 60)
    evaluate_and_save_plots(
        meta, res_data, act_data, bnd_data, cfg, device, output_dir
    )

    logger.info("=" * 60)
    logger.info("ALL DONE. Check /output/ for results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
