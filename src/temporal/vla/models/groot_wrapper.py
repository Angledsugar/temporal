"""Groot N1.6 VLA wrapper for metacontroller integration.

Uses output_hidden_states=True to extract intermediate VLM representations
at the controlled layer (layer 12), then feeds MC-modified features to
the DiT action head.

Unlike π0.5 (which processes VLM+Expert in lockstep), Groot has a
clean separation: Eagle backbone → features → DiT action head.
This makes intervention straightforward — intercept backbone_features
and replace with MC-controlled version.

Usage:
    wrapper = GrootWrapper(config)
    wrapper.freeze_vlm()

    residual = wrapper.extract_residual(batch)          # (B, T, 2048)
    mc_out = metacontroller(residual)                    # apply MC
    loss, v_t = wrapper.predict_with_controlled_residual(
        mc_out["e_controlled"], batch
    )
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_wrapper import BaseVLAWrapper

logger = logging.getLogger(__name__)


@dataclass
class GrootWrapperConfig:
    """Configuration for Groot N1.6 wrapper."""
    checkpoint_path: str = ""
    groot_path: str = "./Isaac-GR00T"
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    select_layer: int = 16
    controlled_layer: int = 12
    action_dim: int = 29
    action_horizon: int = 16
    state_dim: int = 29
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


def _ensure_groot_importable(groot_path: str) -> None:
    """Add Isaac-GR00T source to sys.path if not already importable."""
    expanded = os.path.expanduser(groot_path)
    if expanded not in sys.path:
        sys.path.insert(0, expanded)


class GrootWrapper(BaseVLAWrapper, nn.Module):
    """Wrapper around Gr00tN1d6 providing hidden state extraction.

    Uses Eagle backbone's output_hidden_states=True to access
    intermediate layer representations. The controlled residual
    is then passed to the DiT action head as backbone_features.

    Architecture:
        Eagle backbone (Qwen3 1.7B, 28 layers)
          → output_hidden_states[controlled_layer]  (B, T, 2048)
          → metacontroller intervention
          → remaining layers (controlled_layer..select_layer)
          → DiT action head (32 layers, flow matching)

    Args:
        config: GrootWrapperConfig with model settings.
    """

    def __init__(self, config: GrootWrapperConfig):
        nn.Module.__init__(self)
        self.config = config
        self._controlled_layer = config.controlled_layer
        self._select_layer = config.select_layer

        # Import Groot modules
        _ensure_groot_importable(config.groot_path)

        self._groot_model = None
        self._backbone = None
        self._action_head = None

        if config.checkpoint_path:
            self._load_model(config.checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load Gr00tN1d6 from checkpoint."""
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

        self._groot_model = Gr00tN1d6.from_pretrained(checkpoint_path)
        self._backbone = self._groot_model.backbone
        self._action_head = self._groot_model.action_head

        if self.config.dtype == "bfloat16":
            self._groot_model.to(dtype=torch.bfloat16)

        logger.info(f"Loaded Groot N1.6 from {checkpoint_path}")

    def freeze_vlm(self) -> None:
        """Freeze Eagle backbone + action head parameters."""
        if self._groot_model is not None:
            for p in self._groot_model.parameters():
                p.requires_grad_(False)

    @property
    def vlm_embed_dim(self) -> int:
        return 2048  # Qwen3 1.7B hidden size

    @property
    def controlled_layer(self) -> int:
        return self._controlled_layer

    @property
    def device(self) -> torch.device:
        if self._groot_model is not None:
            return next(self._groot_model.parameters()).device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        if self._groot_model is not None:
            return next(self._groot_model.parameters()).dtype
        return torch.bfloat16

    def _get_vlm_layers(self):
        """Access the Qwen3 decoder layers from Eagle backbone."""
        return self._backbone.model.language_model.model.layers

    def extract_residual(self, batch: dict) -> Tensor:
        """Extract VLM hidden states at controlled_layer.

        Uses Eagle backbone's output_hidden_states=True to get
        all intermediate layer representations, then returns
        the one at controlled_layer.

        Args:
            batch: Dict with 'images', 'state', 'actions', etc.

        Returns:
            residual: (B, seq_len, 2048) hidden states at layer 12.
        """
        if self._backbone is None:
            raise RuntimeError("Model not loaded. Set checkpoint_path in config.")

        # Prepare backbone input (vision + language tokens)
        vl_input = self._prepare_backbone_input(batch)

        # Forward with all hidden states
        outputs = self._backbone.model(
            **vl_input,
            output_hidden_states=True,
        )

        # hidden_states[0] = embedding, [1..28] = after each layer
        # controlled_layer=12 → index 12 (after layer 11, 0-indexed)
        all_hidden_states = outputs["hidden_states"]
        residual = all_hidden_states[self._controlled_layer]

        # Cache batch state for predict_with_controlled_residual
        self._cached_batch_state = {
            "batch": batch,
            "all_hidden_states": all_hidden_states,
            "vl_input": vl_input,
        }

        return residual

    def predict_with_controlled_residual(
        self,
        controlled_residual: Tensor,
        batch: dict,
    ) -> tuple[Tensor, Tensor]:
        """Pass controlled residual through remaining VLM layers + action head.

        For Groot, we:
        1. Run VLM layers [controlled_layer..select_layer) manually
        2. Feed the result as backbone_features to the DiT action head
        3. Compute flow-matching MSE loss

        Args:
            controlled_residual: (B, seq_len, 2048) MC-modified hidden states.
            batch: Original batch dict.

        Returns:
            action_loss: MSE flow-matching loss.
            v_t: Predicted velocity.
        """
        state = self._cached_batch_state
        vlm_layers = self._get_vlm_layers()

        # Run remaining VLM layers on the controlled residual
        hidden = controlled_residual
        for layer_idx in range(self._controlled_layer, self._select_layer):
            if layer_idx < len(vlm_layers):
                layer_output = vlm_layers[layer_idx](
                    hidden,
                    attention_mask=state["vl_input"].get("attention_mask"),
                )
                hidden = layer_output[0]

        # Apply final norm if available
        if hasattr(self._backbone.model.language_model.model, "norm"):
            hidden = self._backbone.model.language_model.model.norm(hidden)

        # Use as backbone_features for action head
        backbone_features = hidden

        # Get action targets from batch
        actions = batch["actions"]
        if actions.device != backbone_features.device:
            actions = actions.to(backbone_features.device)

        # Action head forward (DiT flow matching)
        # The action head takes backbone_features as cross-attention context
        # and predicts velocity for denoising
        if self._action_head is not None:
            action_loss, v_t = self._action_head_forward(
                backbone_features, batch, actions
            )
        else:
            # Fallback: simple projection for testing without full model
            v_t = backbone_features[:, :self.config.action_horizon, :self.config.action_dim]
            noise = torch.randn_like(actions)
            u_t = noise - actions
            action_loss = F.mse_loss(u_t, v_t, reduction="none")

        self._cached_batch_state = None
        return action_loss, v_t

    def _action_head_forward(
        self,
        backbone_features: Tensor,
        batch: dict,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward through DiT action head with flow matching.

        Args:
            backbone_features: (B, seq_len, 2048) VLM features.
            batch: Input batch with state, actions.
            actions: (B, action_horizon, action_dim) target actions.

        Returns:
            loss: MSE loss.
            v_t: Predicted velocity.
        """
        # Sample noise and time for flow matching
        noise = torch.randn_like(actions)
        time = torch.rand(actions.shape[0], device=actions.device)

        # Noisy actions: x_t = t*noise + (1-t)*actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity

        # Forward through action head
        # Note: Exact interface depends on Groot's ActionHead implementation
        # This is a simplified version — actual implementation needs to match
        # Gr00tN1d6ActionHead.forward() signature
        v_t = self._action_head(
            vl_embeddings=backbone_features,
            noisy_actions=x_t,
            timestep=time,
            state=batch.get("state"),
        )

        loss = F.mse_loss(u_t, v_t, reduction="none")
        return loss, v_t

    def _prepare_backbone_input(self, batch: dict) -> dict:
        """Convert batch dict to Eagle backbone input format.

        This handles the image processing and tokenization that
        Eagle expects as input.
        """
        # For dummy data: create minimal valid input
        device = self.device

        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        else:
            # Generate dummy tokens
            batch_size = next(iter(batch["images"].values())).shape[0]
            input_ids = torch.ones(batch_size, 64, dtype=torch.long, device=device)
            attention_mask = torch.ones(batch_size, 64, dtype=torch.bool, device=device)

        # Process images
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"].to(device)
        else:
            # Stack camera images
            img_list = list(batch["images"].values())
            pixel_values = torch.stack(img_list, dim=1).to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
