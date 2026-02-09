"""Phase 1: Action Expert Pretraining with Flow Matching.

Trains Gemma-300M on human manipulation data using conditional
flow-matching objective. After training, theta is FROZEN permanently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Phase1Config:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 500_000
    grad_clip: float = 1.0
    flow_matching_steps: int = 10
    sigma_min: float = 0.001
    log_every: int = 100
    save_every: int = 10_000
    output_dir: str = "checkpoints/phase1/"


class Phase1Trainer:
    """Flow-matching pretraining loop for the action expert.

    Loss:
        L_FM(theta) = E_{t, tau, eps}
            || v_theta(a^(t), t | s, q) - u_t(a^(t) | a) ||^2

    where a^(t) is the noised action at diffusion time t,
    u_t is the conditional vector field.
    """

    def __init__(self, model: nn.Module, dataloader, config: Phase1Config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        subtask_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional flow-matching loss.

        Args:
            actions: (B, T, action_dim) -- ground truth actions.
            proprio: (B, T, proprio_dim) -- proprioceptive state.
            subtask_embed: (B, embed_dim) -- subtask text embedding.

        Returns:
            loss: scalar -- MSE between predicted and target vector field.
        """
        B, T, D = actions.shape

        # Sample diffusion time uniformly
        t = torch.rand(B, 1, 1, device=actions.device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Interpolate: a^(t) = (1-t)*noise + t*actions
        a_t = (1 - t) * noise + t * actions

        # Target vector field: u_t = actions - noise
        target = actions - noise

        # Predicted vector field
        predicted = self.model(a_t, t.squeeze(), proprio, subtask_embed)

        return nn.functional.mse_loss(predicted, target)

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        step = 0

        while step < self.config.max_steps:
            for batch in self.dataloader:
                if step >= self.config.max_steps:
                    break

                actions = batch["actions"].cuda()
                proprio = batch["proprioception"].cuda()
                # TODO: encode subtask text to embedding
                subtask_embed = torch.zeros(
                    actions.shape[0], 256, device=actions.device
                )

                loss = self.flow_matching_loss(actions, proprio, subtask_embed)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()

                if step % self.config.log_every == 0:
                    logger.info(f"Step {step}: loss={loss.item():.4f}")

                if step % self.config.save_every == 0 and step > 0:
                    self._save_checkpoint(step)

                step += 1

        self._save_checkpoint(step)
        logger.info("Phase 1 training complete. Theta is now FROZEN.")

    def _save_checkpoint(self, step: int) -> None:
        path = f"{self.config.output_dir}/action_expert_step{step}.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")
