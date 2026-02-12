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
import torch.nn.functional as F
from tqdm import tqdm

from temporal.models.action_expert import ActionExpert
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
    batch_size: int = 32
    subsequence_len: int = 50    # subsequence length for training
    log_every: int = 100
    save_every: int = 5_000
    output_dir: str = "checkpoints/expert_distill/"


class ExpertDistillTrainer:
    """Self-supervised MetaController training loop.

    Pipeline:
        1. Extract residual streams from frozen action expert
        2. Train MetaController (encoder + switching + decoder)
           to reconstruct actions from controlled residual streams
        3. Switching signal beta_t emerges as subtask boundaries
    """

    def __init__(
        self,
        action_expert: ActionExpert,
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
        self.expert.eval()
        for param in self.expert.parameters():
            param.requires_grad = False

        # Action decoder: residual stream -> action prediction
        self.action_decoder = nn.Linear(
            self.expert.cfg.width, self.expert.cfg.action_dim
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.meta.parameters()) + list(self.action_decoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def extract_residual_streams(
        self, actions: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract residual streams from frozen expert.

        Args:
            actions: (B, T, action_dim) ground truth actions.
            proprio: (B, T, 14) proprioceptive state (optional).

        Returns:
            e_seq: (B, T, n_e) residual stream at intervention layer.
        """
        with torch.no_grad():
            t_zero = torch.zeros(actions.shape[0], device=actions.device)
            self.expert(
                actions,
                timestep=t_zero,
                proprio=proprio,
                extract_residual=True,
            )
            return self.expert.get_residual_stream()

    def train_step(
        self, e_seq: torch.Tensor, target_actions: torch.Tensor
    ) -> dict[str, float]:
        """Single training step.

        Args:
            e_seq: (B, T, n_e) -- residual stream from frozen expert.
            target_actions: (B, T, action_dim) -- ground truth actions.

        Returns:
            Dictionary of loss components and metrics.
        """
        # MetaController forward
        z_seq, kl_loss, beta_seq = self.meta(e_seq)

        # Apply control to residual stream and decode actions
        B, T, n_e = e_seq.shape
        e_flat = e_seq.reshape(B * T, n_e)
        z_flat = z_seq.reshape(B * T, self.meta.n_z)

        # Apply low-rank control: e' = e + B @ (A @ e)
        e_controlled = self.meta.decoder.apply_control(e_flat, z_flat)

        # Decode to action predictions
        pred_actions = self.action_decoder(e_controlled).reshape(B, T, -1)
        recon_loss = F.mse_loss(pred_actions, target_actions)

        # Total loss (variational lower bound)
        total_loss = recon_loss + self.config.alpha * kl_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.meta.parameters()) + list(self.action_decoder.parameters()),
            self.config.grad_clip,
        )
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "beta_mean": beta_seq.mean().item(),
            "beta_sparsity": (beta_seq > 0.5).float().mean().item(),
        }

    def train(self) -> dict[str, list[float]]:
        """Main training loop.

        Returns:
            history: Dict of loss/metric lists.
        """
        self.meta.train()
        self.action_decoder.train()
        step = 0
        start_time = time.time()

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        history: dict[str, list[float]] = {
            "total": [], "recon": [], "kl": [],
            "beta_mean": [], "beta_sparsity": [],
        }

        pbar = tqdm(total=self.config.max_steps, desc="Expert Distill", unit="step")

        while step < self.config.max_steps:
            for batch in self.dataloader:
                if step >= self.config.max_steps:
                    break

                actions = batch["actions"].to(self.device)
                proprio = batch["proprioception"].to(self.device)

                # Extract residual stream from frozen expert
                e_seq = self.extract_residual_streams(actions, proprio)

                # Train step
                metrics = self.train_step(e_seq, actions)

                # Record history
                for k, v in metrics.items():
                    history[k].append(v)

                # Update progress bar
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    loss=f"{metrics['total']:.4f}",
                    recon=f"{metrics['recon']:.4f}",
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
                    self._save_checkpoint(step, history)

                step += 1

        pbar.close()
        self._save_checkpoint(step, history)
        total_time = time.time() - start_time
        logger.info(
            f"Expert Distill complete: {step} steps in {total_time:.1f}s "
            f"({step / total_time:.1f} steps/s)"
        )
        return history

    def _save_checkpoint(self, step: int, history: dict | None = None) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(self.config.output_dir) / f"metacontroller_step{step}.pt"
        torch.save({
            "step": step,
            "metacontroller_state_dict": self.meta.state_dict(),
            "action_decoder_state_dict": self.action_decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def evaluate_boundaries(
        self, num_samples: int = 100
    ) -> dict[str, float]:
        """Evaluate boundary detection quality.

        Returns:
            Dict with beta distribution statistics.
        """
        self.meta.eval()
        all_betas = []

        with torch.no_grad():
            count = 0
            for batch in self.dataloader:
                if count >= num_samples:
                    break

                actions = batch["actions"].to(self.device)
                proprio = batch["proprioception"].to(self.device)
                e_seq = self.extract_residual_streams(actions, proprio)
                _, _, beta_seq = self.meta(e_seq)
                all_betas.append(beta_seq.cpu())
                count += actions.shape[0]

        if not all_betas:
            return {}

        betas = torch.cat(all_betas, dim=0).numpy().flatten()
        near_0 = float((betas < 0.1).mean())
        near_1 = float((betas > 0.9).mean())
        middle = float(((betas >= 0.1) & (betas <= 0.9)).mean())
        avg_switches = float((betas > 0.5).mean())

        quasi_binary = near_0 + near_1 > 0.7

        logger.info(
            f"Beta distribution: near_0={near_0:.1%}, middle={middle:.1%}, "
            f"near_1={near_1:.1%} -> quasi-binary={quasi_binary}"
        )
        logger.info(f"Avg switching rate: {avg_switches:.3f}")

        self.meta.train()

        return {
            "near_0": near_0,
            "near_1": near_1,
            "middle": middle,
            "quasi_binary": quasi_binary,
            "avg_switching_rate": avg_switches,
        }
