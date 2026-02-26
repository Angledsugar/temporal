"""Standalone Gemma config — JAX-free port of openpi.models.gemma.get_config().

This avoids importing the full OpenPI gemma module which requires JAX/Flax.
Values are copied exactly from openpi/src/openpi/models/gemma.py:58-109.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GemmaConfig:
    """Gemma model configuration (mirrors openpi.models.gemma.Config)."""
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict = field(default_factory=dict)


_CONFIGS = {
    "dummy": GemmaConfig(
        width=64, depth=4, mlp_dim=128,
        num_heads=8, num_kv_heads=1, head_dim=16,
    ),
    "gemma_300m": GemmaConfig(
        width=1024, depth=18, mlp_dim=4096,
        num_heads=8, num_kv_heads=1, head_dim=256,
    ),
    "gemma_2b": GemmaConfig(
        width=2048, depth=18, mlp_dim=16384,
        num_heads=8, num_kv_heads=1, head_dim=256,
    ),
    "gemma_2b_lora": GemmaConfig(
        width=2048, depth=18, mlp_dim=16384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={"attn": {"rank": 16, "alpha": 16.0},
                      "ffn": {"rank": 16, "alpha": 16.0}},
    ),
    "gemma_300m_lora": GemmaConfig(
        width=1024, depth=18, mlp_dim=4096,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={"attn": {"rank": 32, "alpha": 32.0},
                      "ffn": {"rank": 32, "alpha": 32.0}},
    ),
}


def get_config(variant: str) -> GemmaConfig:
    """Return config for the specified Gemma variant."""
    if variant not in _CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(_CONFIGS)}")
    return _CONFIGS[variant]
