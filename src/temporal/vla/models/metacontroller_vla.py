"""VLA-adapted metacontroller wrapping the existing MetaController.

Scales the metacontroller architecture from gridworld (256-dim) to
VLA models (2048-dim) while keeping the core logic identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ...models.metacontroller import MetaController


@dataclass
class VLAMetaControllerConfig:
    """Configuration for VLA metacontroller, scaled for 2048-dim VLM."""

    embed_dim: int = 2048       # VLM hidden dim (both π0.5 and Groot)
    latent_dim: int = 16        # scaled from 8 (gridworld)
    gru_dim: int = 64           # scaled from 32
    seq_embed_dim: int = 64     # scaled from 32
    encoder_hidden: int = 128   # scaled from 64
    decoder_hidden: int = 64    # scaled from 32
    controller_rank: int = 32   # scaled from 16
    controlled_layer: int = 9   # π0.5=9, groot=12 (override in yaml)
    kl_alpha: float = 0.17

    # Training
    train_steps: int = 64000
    batch_size: int = 4         # small due to VLM memory
    lr: float = 1e-4
    weight_decay: float = 0.03
    betas: tuple[float, float] = (0.9, 0.999)


class VLAMetaController(nn.Module):
    """VLA-adapted metacontroller.

    Thin wrapper around the existing MetaController with 2048-dim scaling.
    The MetaController is fully model-agnostic — it only requires
    (B, T, embed_dim) residual stream tensors as input.

    Args:
        config: VLA metacontroller configuration.
    """

    def __init__(self, config: VLAMetaControllerConfig):
        super().__init__()
        self.config = config

        self.mc = MetaController(
            embed_dim=config.embed_dim,
            latent_dim=config.latent_dim,
            gru_dim=config.gru_dim,
            seq_embed_dim=config.seq_embed_dim,
            encoder_hidden=config.encoder_hidden,
            decoder_hidden=config.decoder_hidden,
            controller_rank=config.controller_rank,
        )

    def forward(self, residual: Tensor) -> dict[str, Tensor]:
        """Apply metacontroller to VLM residual stream.

        Args:
            residual: (B, T, embed_dim) VLM hidden states at controlled layer.

        Returns:
            Dict with keys:
                e_controlled: (B, T, embed_dim) controlled residual
                z_seq: (B, T, latent_dim) integrated latent codes
                beta_seq: (B, T, 1) switching rates
                mu, logvar: encoder parameters
                kl_loss: scalar KL divergence loss
                z_tilde: (B, T, latent_dim) pre-integration proposals
        """
        return self.mc(residual)

    def get_decoder(self):
        """Return the controller decoder (for Internal RL stage)."""
        return self.mc.get_decoder()

    def get_switching_unit(self):
        """Return the switching unit (for Internal RL stage)."""
        return self.mc.get_switching_unit()
