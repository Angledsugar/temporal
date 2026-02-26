"""Stage 3: VLA Internal RL training (stub).

Learns a causal policy in the abstract action (latent code z_t) space
discovered by the metacontroller. Requires a robot simulator for
reward signals.

This is a stub — full implementation requires:
1. Robot simulator integration (Isaac Sim for Groot, or LIBERO for π0.5)
2. Internal RL environment wrapper (analogous to gridworld internal_env.py)
3. Reward shaping for robotic manipulation tasks

The RL policy architecture (CausalSSMPolicy) and GRPO algorithm
are reused directly from the gridworld PoC with embed_dim=2048.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ...models.rl_policy import CausalSSMPolicy
from ...training.grpo import GRPO
from ..models.base_wrapper import BaseVLAWrapper
from ..models.metacontroller_vla import VLAMetaController

logger = logging.getLogger(__name__)


class VLAInternalRLTrainer:
    """Trainer for Stage 3: VLA Internal RL.

    Uses GRPO in the abstract action space z_t discovered by the
    metacontroller. The policy π(z_t | e_{1:t}) replaces the
    non-causal encoder from Stage 2 with a causal SSM.

    Args:
        wrapper: Frozen VLA model wrapper.
        metacontroller: Trained (frozen) metacontroller from Stage 2.
        config: Internal RL configuration.
        device: Torch device.
    """

    def __init__(
        self,
        wrapper: BaseVLAWrapper,
        metacontroller: VLAMetaController,
        config,
        device: str = "cuda",
    ):
        self.wrapper = wrapper
        self.wrapper.freeze_vlm()

        self.metacontroller = metacontroller.to(device)
        for p in self.metacontroller.parameters():
            p.requires_grad_(False)

        # Causal RL policy: maps residual → latent codes z_t
        self.policy = CausalSSMPolicy(
            embed_dim=config.policy_embed_dim,     # 2048
            latent_dim=metacontroller.config.latent_dim,  # 16
            depth=config.policy_depth,              # 1
        ).to(device)

        self.config = config
        self.device = device

        logger.warning(
            "VLA Internal RL is a stub. Full implementation requires "
            "robot simulator integration for reward signals."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "VLA Internal RL training requires a robot simulator. "
            "See docs for Isaac Sim (Groot) or LIBERO (π0.5) integration."
        )
