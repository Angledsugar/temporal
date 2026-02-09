"""Action Expert Wrapper.

Wraps the OpenPi Gemma-300M transformer to expose intermediate
residual-stream activations for MetaController intervention.

Key integration with openpi/src/openpi/models/gemma.py:
  - configs[0] = PaliGemma VLM backbone
  - configs[1] = Action expert (gemma_300m: width=1024, depth=18)
  - Block.__call__ processes xs = [vlm_hidden, action_hidden]
  - We intercept xs[1] at a chosen layer l to get e_{t,l}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class ActionExpertConfig:
    """Configuration mirroring openpi gemma_300m."""
    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    controlled_layer: int = 9  # mid-depth for optimal controllability


class ActionExpertWrapper:
    """Wraps the frozen OpenPi Gemma-300M to expose residual stream.

    The action expert is ALWAYS frozen after Phase 1.
    This is critical: co-training with MetaController causes
    temporal abstractions to collapse (Kobayashi et al., 2025).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        controlled_layer: int = 9,
    ):
        self.config = ActionExpertConfig(controlled_layer=controlled_layer)
        self.width = self.config.width        # 1024 = n_e
        self.depth = self.config.depth        # 18
        self.controlled_layer = controlled_layer
        self.params: dict[str, Any] | None = None
        self._model = None

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> None:
        """Load pretrained action expert weights."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # TODO: Load actual OpenPi checkpoint
        # self.params = load_checkpoint(checkpoint_path)
        # self._model = build_model(self.config)

    def extract_residual_stream(
        self,
        params: dict[str, Any],
        tokens: jnp.ndarray,
        positions: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray | None]]:
        """Forward pass through first `controlled_layer` blocks.

        Returns intermediate hidden states e_{t,l} for the action expert.

        Key insight from gemma.py:
          - Block.__call__ takes xs = [vlm_hidden, action_hidden]
          - Each block returns updated xs via gated residual connections
          - We intercept xs[1] (action expert) at layer l

        Args:
            params: Model parameters (frozen).
            tokens: Input token ids, shape (B, T).
            positions: Position ids, shape (B, T).
            mask: Attention mask, shape (B, T, S).

        Returns:
            residual_stream: Hidden states at layer l, shape (B, T, 1024).
            xs: Full list of hidden states [vlm_hidden, action_hidden]
                for continuing forward pass from layer l+1.
        """
        # TODO: Implement with actual OpenPi model
        # embedded = [None, action_embedded]  # None for VLM (not used standalone)
        #
        # xs = embedded
        # kv_caches = []
        # for layer_idx in range(self.controlled_layer):
        #     xs, kv = model.layers[layer_idx](xs, kv_cache, positions, mask, ...)
        #     kv_caches.append(kv)
        #
        # residual_stream = xs[1]  # (B, T, 1024)
        # return residual_stream, xs
        raise NotImplementedError("Requires OpenPi model integration")

    def forward_from_layer(
        self,
        params: dict[str, Any],
        xs: list[jnp.ndarray | None],
        positions: jnp.ndarray,
        mask: jnp.ndarray,
        start_layer: int | None = None,
    ) -> jnp.ndarray:
        """Continue forward pass from layer `start_layer` to final layer.

        Used after MetaController applies additive control to e_{t,l}:
            xs[1] = xs[1] + U_t @ xs[1]
        Then this method completes the remaining layers.

        Args:
            params: Model parameters (frozen).
            xs: Hidden states from extract_residual_stream.
            positions: Position ids.
            mask: Attention mask.
            start_layer: Layer to resume from. Defaults to controlled_layer.

        Returns:
            final_hidden: Final layer output for action decoding, shape (B, T, 1024).
        """
        if start_layer is None:
            start_layer = self.controlled_layer

        # TODO: Implement with actual OpenPi model
        # for layer_idx in range(start_layer, self.depth):
        #     xs, kv = model.layers[layer_idx](xs, ...)
        # return xs[1]
        raise NotImplementedError("Requires OpenPi model integration")

    def decode_action(
        self,
        final_hidden: jnp.ndarray,
        num_steps: int = 10,
    ) -> jnp.ndarray:
        """Flow matching decode from final hidden states to motor commands.

        Runs K denoising steps to produce a clean action chunk.

        Args:
            final_hidden: Output of forward_from_layer, shape (B, T, 1024).
            num_steps: Number of flow matching denoising steps (default 10).

        Returns:
            actions: Motor commands, shape (B, H, action_dim).
                H = action chunk size (50 steps = 1 second at 50Hz).
        """
        # TODO: Implement flow matching sampling
        # noise = jax.random.normal(key, shape=(B, H, action_dim))
        # dt = 1.0 / num_steps
        # for step in range(num_steps):
        #     t = step * dt
        #     v = model.vector_field(noise, t, final_hidden)
        #     noise = noise + v * dt
        # return noise
        raise NotImplementedError("Requires flow matching implementation")
