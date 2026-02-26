"""Stage 1: Base autoregressive model pretraining.

Trains a causal Transformer on expert trajectory data using:
- Action prediction loss: -log p(a_t | o_{1:t})
- Observation prediction loss: -λ log p(o_{t+1} | o_{1:t})
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.transformer import CausalTransformer
from ..data.dataset import TrajectoryDataset
from ..utils.config import BaseModelConfig


class BaseModelTrainer:
    """Trainer for Stage 1: base autoregressive model.

    Args:
        config: Base model configuration.
        device: Torch device.
    """

    def __init__(self, config: BaseModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        self.model = CausalTransformer(
            obs_dim=config.obs_dim,
            num_actions=config.num_actions,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            mlp_dim=config.mlp_dim,
            num_rel_pos_buckets=config.num_rel_pos_buckets,
            init_scale=config.init_scale,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss.

        Args:
            batch: Dictionary from TrajectoryDataset.

        Returns:
            loss: Scalar loss tensor.
            metrics: Dictionary with loss components.
        """
        obs = batch["observations"].to(self.device)    # (B, T+1, obs_dim)
        actions = batch["actions"].to(self.device)      # (B, T)
        mask = batch["mask"].to(self.device)             # (B, T)

        # Input: observations o_{1:T}, target actions a_{1:T}, target obs o_{2:T+1}
        obs_input = obs[:, :-1]   # (B, T, obs_dim) — o_{1:T}
        obs_target = obs[:, 1:]   # (B, T, obs_dim) — o_{2:T+1}

        result = self.model(obs_input)
        action_logits = result["action_logits"]  # (B, T, num_actions)
        obs_pred = result["obs_pred"]             # (B, T, obs_dim)

        # Action prediction loss: cross-entropy
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, self.config.num_actions),
            actions.reshape(-1),
            reduction="none",
        ).reshape_as(actions)
        action_loss = (action_loss * mask).sum() / mask.sum().clamp(min=1)

        # Observation prediction loss: MSE (treating one-hot as continuous target)
        obs_loss = F.mse_loss(obs_pred, obs_target, reduction="none").sum(-1)
        obs_loss = (obs_loss * mask).sum() / mask.sum().clamp(min=1)

        total_loss = action_loss + self.config.obs_coeff * obs_loss

        metrics = {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "obs_loss": obs_loss.item(),
        }
        return total_loss, metrics

    def train(
        self,
        dataset: TrajectoryDataset,
        checkpoint_dir: str | Path = "checkpoints",
        log_interval: int = 1000,
        save_interval: int = 10000,
    ) -> CausalTransformer:
        """Run training loop.

        Args:
            dataset: Training dataset.
            checkpoint_dir: Where to save checkpoints.
            log_interval: Print metrics every N steps.
            save_interval: Save checkpoint every N steps.

        Returns:
            Trained model.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.model.train()
        step = 0
        pbar = tqdm(total=self.config.train_steps, desc="Stage 1: Pretraining")

        while step < self.config.train_steps:
            for batch in dataloader:
                if step >= self.config.train_steps:
                    break

                loss, metrics = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                step += 1
                pbar.update(1)

                if step % log_interval == 0:
                    pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

                if step % save_interval == 0:
                    path = checkpoint_dir / f"base_model_step{step}.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }, path)

        pbar.close()

        # Save final model
        final_path = checkpoint_dir / "base_model_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
        }, final_path)
        print(f"Saved final model to {final_path}")

        return self.model
