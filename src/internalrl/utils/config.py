"""Configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BaseModelConfig:
    """Configuration for Stage 1: Base autoregressive model."""
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    head_dim: int = 64
    mlp_dim: int = 512
    num_rel_pos_buckets: int = 32
    obs_dim: int = 637
    num_actions: int = 4
    obs_coeff: float = 0.01  # lambda for observation prediction loss
    dropout: float = 0.0
    init_scale: float = 0.1

    # Training
    train_steps: int = 256000
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.03
    betas: tuple[float, float] = (0.9, 0.999)
    max_seq_len: int = 100


@dataclass
class MetacontrollerConfig:
    """Configuration for Stage 2: Metacontroller."""
    latent_dim: int = 8           # n_z
    gru_dim: int = 32             # n_h
    seq_embed_dim: int = 32       # n_s
    encoder_hidden: int = 64
    decoder_hidden: int = 32
    controller_rank: int = 16     # low-rank for controller
    controlled_layer: int = 3     # L/2

    # KL regularization
    kl_alpha: float = 0.17        # searched in [0, 0.05, 0.1, 0.17, 0.3, 0.5, 1]
    obs_coeff: float = 0.0

    # Training
    train_steps: int = 64000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.03
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class InternalRLConfig:
    """Configuration for Stage 3: Internal RL."""
    policy_type: str = "ssm"
    policy_depth: int = 1
    policy_embed_dim: int = 256
    beta_threshold: float = 0.5

    # Training (GRPO)
    train_steps: int = 100000
    batch_size: int = 1024
    lr: float = 3e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    entropy_coeff: float = 0.0
    clip_epsilon: float = 0.2
    advantage_delta: float = 1e-3


@dataclass
class Config:
    """Full pipeline configuration."""
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    metacontroller: MetacontrollerConfig = field(default_factory=MetacontrollerConfig)
    internal_rl: InternalRLConfig = field(default_factory=InternalRLConfig)

    # Data
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "base_model" in raw:
            for k, v in raw["base_model"].items():
                setattr(cfg.base_model, k, v)
        if "metacontroller" in raw:
            for k, v in raw["metacontroller"].items():
                setattr(cfg.metacontroller, k, v)
        if "internal_rl" in raw:
            for k, v in raw["internal_rl"].items():
                setattr(cfg.internal_rl, k, v)
        for k in ("data_dir", "checkpoint_dir", "log_dir", "seed"):
            if k in raw:
                setattr(cfg, k, raw[k])
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# --- VLA Configurations ---

@dataclass
class VLAModelConfig:
    """Configuration for VLA model wrapper."""
    type: str = "pi05"                  # "pi05" or "groot"
    checkpoint_path: str = ""
    openpi_path: str = "~/project/openpi"
    groot_path: str = "~/project/Isaac-GR00T"
    # π0.5 specific
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    max_token_len: int = 200
    # Groot specific
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    select_layer: int = 16
    # Common
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


@dataclass
class VLAMetacontrollerConfig:
    """Configuration for VLA metacontroller (Stage 2)."""
    embed_dim: int = 2048
    latent_dim: int = 16
    gru_dim: int = 64
    seq_embed_dim: int = 64
    encoder_hidden: int = 128
    decoder_hidden: int = 64
    controller_rank: int = 32
    controlled_layer: int = 9       # π0.5=9, groot=12
    kl_alpha: float = 0.17

    # Training
    train_steps: int = 64000
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.03
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class VLAInternalRLConfig:
    """Configuration for VLA internal RL (Stage 3)."""
    policy_type: str = "ssm"
    policy_depth: int = 1
    policy_embed_dim: int = 2048
    beta_threshold: float = 0.5

    train_steps: int = 100000
    batch_size: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    clip_epsilon: float = 0.2


@dataclass
class VLADataConfig:
    """Configuration for VLA data loading."""
    type: str = "dummy"             # "dummy" or "real"
    num_samples: int = 100
    data_dir: str = "data/vla"


@dataclass
class VLAConfig:
    """Full VLA pipeline configuration."""
    model: VLAModelConfig = field(default_factory=VLAModelConfig)
    metacontroller: VLAMetacontrollerConfig = field(default_factory=VLAMetacontrollerConfig)
    internal_rl: VLAInternalRLConfig = field(default_factory=VLAInternalRLConfig)
    data: VLADataConfig = field(default_factory=VLADataConfig)

    checkpoint_dir: str = "checkpoints/vla"
    log_dir: str = "logs/vla"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> VLAConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if "metacontroller" in raw:
            for k, v in raw["metacontroller"].items():
                setattr(cfg.metacontroller, k, v)
        if "internal_rl" in raw:
            for k, v in raw["internal_rl"].items():
                setattr(cfg.internal_rl, k, v)
        if "data" in raw:
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)
        for k in ("checkpoint_dir", "log_dir", "seed"):
            if k in raw:
                setattr(cfg, k, raw[k])
        return cfg
