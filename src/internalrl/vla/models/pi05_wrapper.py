"""π0.5 VLA wrapper with split forward for metacontroller integration.

Wraps PI0Pytorch from OpenPI to expose a split-forward interface,
allowing metacontroller intervention at a specific VLM layer.

The key challenge is that π0.5 processes VLM (PaliGemma) and Action Expert
(Gemma 300M) in lockstep through `compute_layer_complete()`. We replicate
this per-layer computation to enable splitting at any layer.

Usage:
    wrapper = Pi05Wrapper(config)
    wrapper.freeze_vlm()

    residual = wrapper.extract_residual(batch)          # (B, T, 2048)
    mc_out = metacontroller(residual)                    # apply MC
    loss, v_t = wrapper.predict_with_controlled_residual(
        mc_out["e_controlled"], batch
    )
"""

from __future__ import annotations

import logging
import math
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
class Pi05WrapperConfig:
    """Configuration for π0.5 wrapper."""
    checkpoint_path: str = ""
    openpi_path: str = "./openpi"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    dtype: str = "bfloat16"
    controlled_layer: int = 9
    gradient_checkpointing: bool = True


def _ensure_openpi_importable(openpi_path: str) -> None:
    """Add OpenPI source to sys.path if not already importable."""
    expanded = os.path.expanduser(openpi_path)
    src_path = os.path.join(expanded, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


class Pi05Wrapper(BaseVLAWrapper, nn.Module):
    """Wrapper around PI0Pytorch providing split-forward for metacontroller.

    Replicates the per-layer computation from
    PaliGemmaWithExpertModel.forward() (gemma_pytorch.py:158-238)
    as a standalone _compute_layer() method, enabling intervention
    at any VLM layer.

    Args:
        config: Pi05WrapperConfig with model settings.
    """

    def __init__(self, config: Pi05WrapperConfig):
        nn.Module.__init__(self)
        self.config = config
        self._controlled_layer = config.controlled_layer

        # Install JAX-free shims, then add OpenPI to sys.path
        from internalrl.vla.models._openpi_shims import install_shims
        install_shims()
        _ensure_openpi_importable(config.openpi_path)

        # Build a lightweight Pi0Config-like object for PI0Pytorch
        pi0_config = _Pi05ConfigShim(
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            dtype=config.dtype,
            pi05=True,
        )

        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        self.pi0_model = PI0Pytorch(pi0_config)

        # Load checkpoint if specified
        if config.checkpoint_path:
            import safetensors.torch
            safetensors.torch.load_model(
                self.pi0_model, config.checkpoint_path, strict=False
            )
            logger.info(f"Loaded π0.5 weights from {config.checkpoint_path}")

        if config.gradient_checkpointing:
            self.pi0_model.gradient_checkpointing_enable()

        # Cache model references for layer access
        self._vlm_model = (
            self.pi0_model.paligemma_with_expert.paligemma.language_model
        )
        self._expert_model = (
            self.pi0_model.paligemma_with_expert.gemma_expert.model
        )
        self._num_layers = self._vlm_model.config.num_hidden_layers

    def freeze_vlm(self) -> None:
        """Freeze all π0.5 parameters (VLM + Action Expert + projections)."""
        for p in self.pi0_model.parameters():
            p.requires_grad_(False)

    @property
    def vlm_embed_dim(self) -> int:
        return self._vlm_model.config.hidden_size  # 2048

    @property
    def controlled_layer(self) -> int:
        return self._controlled_layer

    @property
    def device(self) -> torch.device:
        return next(self.pi0_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.pi0_model.parameters()).dtype

    # --- Preprocessing helpers ---

    def _prepare_batch(self, batch: dict) -> tuple:
        """Prepare batch into embeddings, masks, and targets.

        Returns:
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids,
            adarms_cond, u_t (target velocity), prefix_len
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        images = batch["images"]
        img_masks = batch["image_masks"]
        lang_tokens = batch["tokenized_prompt"]
        lang_masks = batch["tokenized_prompt_mask"]
        state = batch["state"]
        actions = batch["actions"]

        # Sample noise and time
        noise = batch.get("noise")
        time = batch.get("time")
        if noise is None:
            noise = self.pi0_model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.pi0_model.sample_time(actions.shape[0], actions.device)

        # Flow matching: x_t = t*noise + (1-t)*actions, u_t = noise - actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (images + language → VLM)
        img_list = list(images.values())
        img_mask_list = list(img_masks.values())
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.pi0_model.embed_prefix(img_list, img_mask_list, lang_tokens, lang_masks)
        )

        # Embed suffix (state + noisy actions + time → Action Expert)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.pi0_model.embed_suffix(state, x_t, time)
        )

        # Cast to model dtype
        model_dtype = self._vlm_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Build attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.pi0_model._prepare_attention_masks_4d(att_2d_masks)

        prefix_len = prefix_embs.shape[1]

        return (
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids,
            adarms_cond, u_t, prefix_len, time
        )

    # --- Per-layer computation ---

    def _compute_layer(
        self,
        layer_idx: int,
        inputs_embeds: list[Tensor],
        attention_mask: Tensor,
        position_ids: Tensor,
        adarms_cond: list[Tensor | None],
    ) -> list[Tensor]:
        """Compute one layer of the dual-stream model.

        Replicates compute_layer_complete() from gemma_pytorch.py:158-238.

        Args:
            layer_idx: Layer index (0 to num_layers-1).
            inputs_embeds: [prefix_hidden (B, P, 2048), suffix_hidden (B, S, 1024)]
            attention_mask: (B, 1, P+S, P+S) 4D attention mask.
            position_ids: (B, P+S) position IDs.
            adarms_cond: [None, adarms_cond_for_expert]

        Returns:
            Updated [prefix_hidden, suffix_hidden].
        """
        from transformers.models.gemma import modeling_gemma

        models = [self._vlm_model, self._expert_model]

        # Pre-attention: layernorm + Q/K/V projection
        query_states = []
        key_states = []
        value_states = []
        gates = []

        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            normed, gate = layer.input_layernorm(
                hidden_states, cond=adarms_cond[i]
            )
            gates.append(gate)

            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        # Concatenate streams for joint attention
        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)

        # RoPE positional encoding
        dummy_tensor = torch.zeros(
            query_states.shape[0], query_states.shape[2], query_states.shape[-1],
            device=query_states.device, dtype=query_states.dtype,
        )
        rotary_emb = (
            self.pi0_model.paligemma_with_expert
            .paligemma.model.language_model.rotary_emb
        )
        cos, sin = rotary_emb(dummy_tensor, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        # Joint attention
        scaling = self._vlm_model.layers[layer_idx].self_attn.scaling
        att_output, _ = modeling_gemma.eager_attention_forward(
            self._vlm_model.layers[layer_idx].self_attn,
            query_states, key_states, value_states,
            attention_mask, scaling,
        )
        head_dim = self._vlm_model.layers[layer_idx].self_attn.head_dim
        batch_size = att_output.shape[0]
        att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

        # Split attention output back and apply O-projection + residual
        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]

            att_slice = att_output[:, start_pos:end_pos]
            if att_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                att_slice = att_slice.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_slice)

            # First residual
            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
            after_first_residual = out_emb.clone()

            # Post-attention norm + MLP
            out_emb, gate = layer.post_attention_layernorm(
                out_emb, cond=adarms_cond[i]
            )
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            # Second residual
            out_emb = modeling_gemma._gated_residual(
                after_first_residual, out_emb, gate
            )
            outputs_embeds.append(out_emb)
            start_pos = end_pos

        return outputs_embeds

    # --- Split forward interface ---

    def _forward_layers(
        self,
        inputs_embeds: list[Tensor],
        attention_mask: Tensor,
        position_ids: Tensor,
        adarms_cond: list[Tensor | None],
        start_layer: int,
        end_layer: int,
    ) -> list[Tensor]:
        """Process layers [start_layer, end_layer) of dual-stream model."""
        for layer_idx in range(start_layer, end_layer):
            inputs_embeds = self._compute_layer(
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
            )
        return inputs_embeds

    def _apply_final_norm(
        self,
        inputs_embeds: list[Tensor],
        adarms_cond: list[Tensor | None],
    ) -> list[Tensor]:
        """Apply final RMSNorm to both streams."""
        models = [self._vlm_model, self._expert_model]
        outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            out, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            outputs.append(out)
        return outputs

    # --- BaseVLAWrapper interface ---

    def extract_residual(self, batch: dict) -> Tensor:
        """Extract VLM prefix residual at controlled_layer.

        Returns:
            prefix_hidden: (B, prefix_len, 2048) at controlled_layer.
        """
        (
            prefix_embs, suffix_embs, att_mask, pos_ids,
            adarms_cond, u_t, prefix_len, time
        ) = self._prepare_batch(batch)

        # Forward layers 0..controlled_layer-1
        hidden = self._forward_layers(
            [prefix_embs, suffix_embs],
            att_mask, pos_ids, [None, adarms_cond],
            start_layer=0, end_layer=self._controlled_layer,
        )

        # Store batch state for predict_with_controlled_residual
        self._cached_batch_state = {
            "suffix_hidden": hidden[1],
            "att_mask": att_mask,
            "pos_ids": pos_ids,
            "adarms_cond": adarms_cond,
            "u_t": u_t,
            "prefix_len": prefix_len,
        }

        return hidden[0]  # prefix_hidden (B, prefix_len, 2048)

    def predict_with_controlled_residual(
        self,
        controlled_residual: Tensor,
        batch: dict,
    ) -> tuple[Tensor, Tensor]:
        """Continue from controlled residual to action prediction + loss.

        Args:
            controlled_residual: (B, prefix_len, 2048) modified by MC.
            batch: Original batch (used for cached state lookup).

        Returns:
            action_loss: (B, action_horizon, action_dim) MSE loss per element.
            v_t: (B, action_horizon, action_dim) predicted velocity.
        """
        state = self._cached_batch_state
        suffix_hidden = state["suffix_hidden"]
        att_mask = state["att_mask"]
        pos_ids = state["pos_ids"]
        adarms_cond = state["adarms_cond"]
        u_t = state["u_t"]

        # Forward layers controlled_layer..num_layers-1
        hidden = self._forward_layers(
            [controlled_residual, suffix_hidden],
            att_mask, pos_ids, [None, adarms_cond],
            start_layer=self._controlled_layer,
            end_layer=self._num_layers,
        )

        # Final norm
        normed = self._apply_final_norm(hidden, [None, adarms_cond])

        # Extract action predictions from suffix output
        suffix_out = normed[1]
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.pi0_model.action_out_proj(suffix_out)

        # MSE loss
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        # Clean up cached state
        self._cached_batch_state = None

        return action_loss, v_t


class _Pi05ConfigShim:
    """Minimal config shim matching PI0Pytorch.__init__ expectations."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
