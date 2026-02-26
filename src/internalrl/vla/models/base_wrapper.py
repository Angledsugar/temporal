"""Abstract base class for VLA model wrappers.

Defines the common interface that both π0.5 and Groot wrappers implement,
allowing the training pipeline to be model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseVLAWrapper(ABC):
    """Common interface for VLA model wrappers.

    Both Pi05Wrapper and GrootWrapper implement this interface so that
    the metacontroller training pipeline (VLAMetacontrollerTrainer) can
    operate without knowing which VLA model is being used.
    """

    @abstractmethod
    def freeze_vlm(self) -> None:
        """Freeze all VLM backbone parameters (requires_grad=False)."""

    @abstractmethod
    def extract_residual(self, batch: dict) -> Tensor:
        """Extract VLM residual stream at controlled_layer.

        Args:
            batch: Model-specific input dict (images, tokens, state, actions).

        Returns:
            Residual stream tensor of shape (B, T, embed_dim) at the
            controlled layer. T is the number of prefix tokens (images
            + language for π0.5) or full sequence tokens (for Groot).
        """

    @abstractmethod
    def predict_with_controlled_residual(
        self,
        controlled_residual: Tensor,
        batch: dict,
    ) -> tuple[Tensor, Tensor]:
        """Continue forward pass from controlled residual to action prediction.

        Takes the metacontroller-modified residual stream and produces
        action predictions + loss.

        Args:
            controlled_residual: (B, T, embed_dim) modified by metacontroller.
            batch: Model-specific input dict (for targets, noise, time, etc).

        Returns:
            action_loss: (B, ...) MSE flow-matching loss.
            v_t: (B, action_horizon, action_dim) predicted velocity.
        """

    @property
    @abstractmethod
    def vlm_embed_dim(self) -> int:
        """VLM hidden dimension (2048 for both π0.5 and Groot)."""

    @property
    @abstractmethod
    def controlled_layer(self) -> int:
        """Layer index where metacontroller intervenes."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device of the model parameters."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters (typically bfloat16)."""
