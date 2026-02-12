"""Expert Distill: MetaController Self-Supervised Training.

Distills temporal abstraction structure from the frozen action expert's
residual stream. Discovers subtask boundaries (beta_t) without any
boundary supervision.

Loss:
    L(phi) = sum_t [ -ln p(a_t | o_{1:t}, z_{1:t})
                     + alpha * KL(N(mu_t, sigma_t^2) || N(0, I)) ]

CRITICAL: The action expert (theta) MUST be frozen.
Co-training causes temporal abstractions to collapse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from temporal.models.metacontroller import MetaController

logger = logging.getLogger(__name__)


@dataclass
class ExpertDistillConfig:
    alpha: float = 0.05          # KL weight (rate-distortion tradeoff)
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 200_000
    grad_clip: float = 1.0
    controlled_layer: int = 9    # mid-depth of 18 layers
    log_every: int = 100
    save_every: int = 5_000
    visualize_beta_every: int = 1_000
    output_dir: str = "checkpoints/expert_distill/"


class ExpertDistillTrainer:
    """Self-supervised MetaController training loop."""

    def __init__(
        self,
        action_expert,
        metacontroller: MetaController,
        dataloader,
        config: ExpertDistillConfig,
    ):
        self.expert = action_expert
        self.meta = metacontroller
        self.dataloader = dataloader
        self.config = config
        self.device = next(metacontroller.parameters()).device

        # Freeze action expert -- CRITICAL
        # ActionExpertWrapper is JAX-based; only freeze if it's an nn.Module
        if isinstance(self.expert, nn.Module):
            for param in self.expert.parameters():
                param.requires_grad = False
            self.expert.eval()

        self.optimizer = torch.optim.AdamW(
            self.meta.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_step(
        self, e_seq: torch.Tensor, target_actions: torch.Tensor
    ) -> dict[str, float]:
        """Single training step.

        Args:
            e_seq: (B, T, n_e) -- residual stream from frozen expert.
            target_actions: (B, T, action_dim) -- ground truth actions.

        Returns:
            Dictionary of loss components.
        """
        # MetaController forward
        z_seq, kl_loss, beta_seq = self.meta(e_seq)

        # Apply control to residual stream and decode actions
        B, T, n_e = e_seq.shape
        recon_losses = []

        for t in range(T - 1):
            e_controlled = self.meta.decoder.apply_control(
                e_seq[:, t], z_seq[:, t]
            )
            # TODO: decode controlled residual stream to action prediction
            # pred_action = self.expert.decode_from_hidden(e_controlled)
            # recon_loss = F.mse_loss(pred_action, target_actions[:, t])
            # recon_losses.append(recon_loss)

        # Placeholder until expert decode is implemented
        recon_loss = torch.tensor(0.0, device=e_seq.device)
        if recon_losses:
            recon_loss = torch.stack(recon_losses).mean()

        # Total loss (variational lower bound)
        total_loss = recon_loss + self.config.alpha * kl_loss

        return {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "beta_mean": beta_seq.mean().item(),
            "beta_sparsity": (beta_seq > 0.5).float().mean().item(),
        }

    def train(self) -> None:
        """Main training loop with tqdm progress bar."""
        self.meta.train()
        step = 0
        start_time = time.time()

        pbar = tqdm(total=self.config.max_steps, desc="Expert Distill", unit="step")

        while step < self.config.max_steps:
            for batch in self.dataloader:
                if step >= self.config.max_steps:
                    break

                actions = batch["actions"].to(self.device)

                # Extract residual stream from frozen expert
                with torch.no_grad():
                    # TODO: actual residual stream extraction
                    # e_seq = self.expert.extract_residual_stream(batch)
                    e_seq = torch.randn(
                        actions.shape[0], actions.shape[1], 1024,
                        device=actions.device,
                    )

                metrics = self.train_step(e_seq, actions)

                # Backward pass
                z_seq, kl_loss, beta_seq = self.meta(e_seq)
                loss = self.config.alpha * kl_loss  # simplified until recon works

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.meta.parameters(), self.config.grad_clip
                )
                self.optimizer.step()

                # Update progress bar
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    loss=f"{metrics['total']:.4f}",
                    kl=f"{metrics['kl']:.4f}",
                    beta=f"{metrics['beta_sparsity']:.3f}",
                    sps=f"{steps_per_sec:.1f}",
                )
                pbar.update(1)

                if step % self.config.log_every == 0:
                    logger.info(
                        f"Step {step}/{self.config.max_steps} | "
                        f"loss={metrics['total']:.4f} recon={metrics['recon']:.4f} "
                        f"kl={metrics['kl']:.4f} beta_sparsity={metrics['beta_sparsity']:.3f} | "
                        f"{steps_per_sec:.1f} steps/s"
                    )

                if step % self.config.save_every == 0 and step > 0:
                    self._save_checkpoint(step)

                step += 1

        pbar.close()
        self._save_checkpoint(step)
        total_time = time.time() - start_time
        logger.info(
            f"Expert Distill complete: {step} steps in {total_time:.1f}s "
            f"({step / total_time:.1f} steps/s)"
        )

    def _save_checkpoint(self, step: int) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        path = f"{self.config.output_dir}/metacontroller_step{step}.pt"
        torch.save(self.meta.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")
