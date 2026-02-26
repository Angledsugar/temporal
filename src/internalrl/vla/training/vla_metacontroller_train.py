"""Stage 2: VLA Metacontroller training (model-agnostic).

Trains the metacontroller on frozen VLA model residual streams.
Works with both π0.5 and Groot through the BaseVLAWrapper interface.

Loss: L(φ) = MSE(u_t, v_t) + α · D_KL(N(μ,Σ) || N(0,I))
  - MSE: flow-matching velocity prediction loss (replaces gridworld cross-entropy)
  - KL: weighted by switching gate β_t (only active at transitions)

The VLA model is completely frozen; only metacontroller φ is trained.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.base_wrapper import BaseVLAWrapper
from ..models.metacontroller_vla import VLAMetaController, VLAMetaControllerConfig

logger = logging.getLogger(__name__)


class VLAMetacontrollerTrainer:
    """Trainer for Stage 2: VLA metacontroller.

    Freezes the VLA model and trains the metacontroller to discover
    temporally-abstract actions from the VLM residual stream.

    Model-agnostic: works with both Pi05Wrapper and GrootWrapper
    through the BaseVLAWrapper interface.

    Args:
        wrapper: VLA model wrapper (π0.5 or Groot).
        metacontroller: VLA metacontroller.
        config: Metacontroller training config.
        device: Torch device.
    """

    def __init__(
        self,
        wrapper: BaseVLAWrapper,
        metacontroller: VLAMetaController,
        config: VLAMetaControllerConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # Freeze VLA model
        self.wrapper = wrapper
        self.wrapper.freeze_vlm()

        self.metacontroller = metacontroller.to(device)

        self.optimizer = torch.optim.AdamW(
            self.metacontroller.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

    def compute_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute metacontroller training loss.

        Pipeline:
        1. Extract residual stream at controlled layer (no grad, frozen VLM)
        2. Run metacontroller to get controlled residual (with grad)
        3. Continue VLA forward to get action prediction (frozen params)
        4. Compute MSE flow-matching loss + KL divergence

        Args:
            batch: Model-specific input dict.

        Returns:
            total_loss: Scalar loss for backpropagation.
            metrics: Dict of metric values for logging.
        """
        # Move batch to device
        batch = self._to_device(batch)

        # 1. Extract residual (no grad for frozen VLM)
        with torch.no_grad():
            residual = self.wrapper.extract_residual(batch)

        # 2. Run metacontroller (with grad)
        mc_output = self.metacontroller(residual)
        e_controlled = mc_output["e_controlled"]

        # 3. Continue forward and compute action loss
        # Note: wrapper's predict_with_controlled_residual uses cached
        # state from extract_residual, so no need to pass batch again
        # for the forward computation. Batch is passed for interface
        # compatibility.
        action_loss, v_t = self.wrapper.predict_with_controlled_residual(
            e_controlled, batch
        )

        # 4. Combine losses
        action_loss_mean = action_loss.mean()
        kl_loss = mc_output["kl_loss"]

        total_loss = action_loss_mean + self.config.kl_alpha * kl_loss

        metrics = {
            "loss": total_loss.item(),
            "action_loss": action_loss_mean.item(),
            "kl_loss": kl_loss.item(),
            "beta_mean": mc_output["beta_seq"].mean().item(),
            "beta_binary": (
                ((mc_output["beta_seq"] > 0.9) | (mc_output["beta_seq"] < 0.1))
                .float().mean().item()
            ),
        }
        return total_loss, metrics

    def train(
        self,
        dataset,
        checkpoint_dir: str | Path = "checkpoints/vla",
        log_interval: int = 100,
        save_interval: int = 5000,
    ) -> VLAMetaController:
        """Run metacontroller training loop.

        Args:
            dataset: PyTorch Dataset providing VLA training data.
            checkpoint_dir: Directory for saving checkpoints.
            log_interval: Steps between logging metrics.
            save_interval: Steps between saving checkpoints.

        Returns:
            Trained metacontroller.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        self.metacontroller.train()
        step = 0
        pbar = tqdm(total=self.config.train_steps, desc="Stage 2: VLA Metacontroller")

        while step < self.config.train_steps:
            for batch in dataloader:
                if step >= self.config.train_steps:
                    break

                loss, metrics = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.metacontroller.parameters(), 1.0
                )
                self.optimizer.step()

                step += 1
                pbar.update(1)

                if step % log_interval == 0:
                    pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

                if step % save_interval == 0:
                    self._save_checkpoint(checkpoint_dir, step)

        pbar.close()

        # Save final checkpoint
        final_path = checkpoint_dir / "metacontroller_vla_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.metacontroller.state_dict(),
        }, final_path)
        logger.info(f"Saved VLA metacontroller to {final_path}")

        return self.metacontroller

    def _save_checkpoint(self, checkpoint_dir: Path, step: int) -> None:
        path = checkpoint_dir / f"metacontroller_vla_step{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.metacontroller.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "kl_alpha": self.config.kl_alpha,
                "controlled_layer": self.config.controlled_layer,
                "embed_dim": self.config.embed_dim,
            },
        }, path)

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            elif isinstance(v, dict):
                moved[k] = {
                    kk: vv.to(self.device) if isinstance(vv, torch.Tensor) else vv
                    for kk, vv in v.items()
                }
            else:
                moved[k] = v
        return moved
