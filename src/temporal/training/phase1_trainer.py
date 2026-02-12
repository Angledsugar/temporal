"""Phase 1: Action Expert Pretraining with Flow Matching.

Trains Gemma-300M on human manipulation data using conditional
flow-matching objective. After training, theta is FROZEN permanently.

Supports:
  - Multi-dataset weighted sampling (Inter-X + Ego4D + UniHand)
  - Text-conditioned flow matching (sentence-transformers encoding)
  - Cosine LR scheduling with warmup
  - Mixed precision training (bfloat16)
  - WandB logging
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    batch_size: int = 256
    grad_accum_steps: int = 1
    log_every: int = 100
    save_every: int = 10_000
    output_dir: str = "checkpoints/phase1/"
    use_wandb: bool = False
    wandb_project: str = "temporal-phase1"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


class TextEncoder:
    """Lazy-loaded sentence-transformers text encoder for subtask embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self._model = None
        self._model_name = model_name
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info(f"Loaded text encoder: {self._model_name}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode list of texts to embeddings.

        Args:
            texts: List of B text strings.

        Returns:
            embeddings: (B, 384) float32 tensor.
        """
        self._load()
        embeddings = self._model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self._device)


class Phase1Trainer:
    """Flow-matching pretraining loop for the action expert.

    Loss:
        L_FM(theta) = E_{t, tau, eps}
            || v_theta(a^(t), t | s, q) - u_t(a^(t) | a) ||^2

    where a^(t) is the noised action at diffusion time t,
    u_t is the conditional vector field.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        config: Phase1Config,
        text_encoder: TextEncoder | None = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = next(model.parameters()).device

        self.text_encoder = text_encoder or TextEncoder(device=str(self.device))

        # Enable gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Cosine LR schedule with warmup
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(config.warmup_steps, 1)
            progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.use_amp = config.dtype == "bfloat16" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"))

        # WandB
        self._wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    config=vars(config),
                )
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional flow-matching loss.

        Args:
            actions: (B, T, action_dim) -- ground truth actions.
            proprio: (B, T, proprio_dim) -- proprioceptive state.
            text_embed: (B, embed_dim) -- subtask text embedding.

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
        predicted = self.model(
            a_t,
            timestep=t.squeeze(-1).squeeze(-1),  # (B,)
            proprio=proprio,
            text_embed=text_embed,
        )

        return nn.functional.mse_loss(predicted, target)

    def train(self) -> None:
        """Main training loop with gradient accumulation."""
        self.model.train()
        step = 0
        accum_steps = self.config.grad_accum_steps
        start_time = time.time()

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        effective_batch = self.config.batch_size * accum_steps
        logger.info(
            f"Starting Phase 1 training: {self.config.max_steps} steps, "
            f"batch_size={self.config.batch_size}x{accum_steps}={effective_batch}, "
            f"lr={self.config.learning_rate}"
        )

        self.optimizer.zero_grad()
        accum_loss = 0.0
        micro_step = 0

        while step < self.config.max_steps:
            for batch in self.dataloader:
                if step >= self.config.max_steps:
                    break

                actions = batch["actions"].to(self.device)
                proprio = batch["proprioception"].to(self.device)
                texts = batch["text"]  # list of strings

                # Encode text
                text_embed = self.text_encoder.encode(texts).to(self.device)

                # Forward + loss with optional AMP
                amp_dtype = torch.bfloat16 if self.use_amp else None
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=amp_dtype):
                    loss = self.flow_matching_loss(actions, proprio, text_embed)
                    loss = loss / accum_steps  # scale for accumulation

                if self.config.dtype == "float16":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()
                micro_step += 1

                if micro_step % accum_steps != 0:
                    continue

                # Optimizer step after accumulation
                if self.config.dtype == "float16":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                if step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {step}/{self.config.max_steps}: "
                        f"loss={accum_loss:.4f} lr={lr:.2e} "
                        f"({steps_per_sec:.1f} steps/s)"
                    )
                    if self._wandb_run is not None:
                        import wandb
                        wandb.log({
                            "loss": accum_loss,
                            "lr": lr,
                            "steps_per_sec": steps_per_sec,
                        }, step=step)

                accum_loss = 0.0

                if step % self.config.save_every == 0 and step > 0:
                    self._save_checkpoint(step)

                step += 1

        self._save_checkpoint(step)
        total_time = time.time() - start_time
        logger.info(
            f"Phase 1 training complete: {step} steps in {total_time:.0f}s. "
            f"Theta is now FROZEN."
        )

    def _save_checkpoint(self, step: int) -> None:
        path = Path(self.config.output_dir) / f"action_expert_step{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint: {path}")
