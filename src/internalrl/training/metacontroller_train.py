"""Stage 2: Metacontroller training (unsupervised abstract action discovery).

Trains the metacontroller on frozen base model residual streams.
Loss: L(φ) = -log p_{θ,φ}(a_t | o_{1:t}, z_{1:t}) + α · D_KL(N(μ,Σ) || N(0,I))

The base model θ is completely frozen; only metacontroller φ is trained.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.transformer import CausalTransformer
from ..models.metacontroller import MetaController
from ..data.dataset import TrajectoryDataset
from ..utils.config import MetacontrollerConfig, BaseModelConfig


class MetacontrollerTrainer:
    """Trainer for Stage 2: metacontroller.

    Freezes the base model and trains the metacontroller to discover
    temporally-abstract actions from the residual stream.

    Args:
        base_model: Pretrained (frozen) base autoregressive model.
        config: Metacontroller configuration.
        base_config: Base model configuration (for dimensions).
        device: Torch device.
    """

    def __init__(
        self,
        base_model: CausalTransformer,
        config: MetacontrollerConfig,
        base_config: BaseModelConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.base_config = base_config
        self.device = device

        # Freeze base model
        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Create metacontroller
        self.metacontroller = MetaController(
            embed_dim=base_config.embed_dim,
            latent_dim=config.latent_dim,
            gru_dim=config.gru_dim,
            seq_embed_dim=config.seq_embed_dim,
            encoder_hidden=config.encoder_hidden,
            decoder_hidden=config.decoder_hidden,
            controller_rank=config.controller_rank,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.metacontroller.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute metacontroller training loss.

        Pipeline:
        1. Extract residual stream from frozen base model at controlled layer
        2. Run metacontroller to get controlled residual stream
        3. Continue through remaining base model layers
        4. Compute action prediction loss + KL divergence
        """
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        mask = batch["mask"].to(self.device)

        obs_input = obs[:, :-1]  # (B, T, obs_dim)
        layer = self.config.controlled_layer

        # 1. Extract residual stream at controlled layer (no grad)
        with torch.no_grad():
            e_at_layer, _ = self.base_model.forward_up_to_layer(obs_input, layer)

        # 2. Run metacontroller (with grad)
        mc_output = self.metacontroller(e_at_layer)
        e_controlled = mc_output["e_controlled"]  # (B, T, embed_dim)

        # 3. Continue through remaining layers (no grad for base model params)
        with torch.no_grad():
            result, _ = self.base_model.forward_from_layer(e_controlled, layer)

        action_logits = result["action_logits"]  # (B, T, num_actions)

        # 4. Action prediction loss
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, self.base_config.num_actions),
            actions.reshape(-1),
            reduction="none",
        ).reshape_as(actions)
        action_loss = (action_loss * mask).sum() / mask.sum().clamp(min=1)

        # 5. KL loss
        kl_loss = mc_output["kl_loss"]

        # Total loss
        total_loss = action_loss + self.config.kl_alpha * kl_loss

        metrics = {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
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
        dataset: TrajectoryDataset,
        checkpoint_dir: str | Path = "checkpoints",
        log_interval: int = 500,
        save_interval: int = 5000,
    ) -> MetaController:
        """Run metacontroller training loop."""
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

        self.metacontroller.train()
        step = 0
        pbar = tqdm(total=self.config.train_steps, desc="Stage 2: Metacontroller")

        while step < self.config.train_steps:
            for batch in dataloader:
                if step >= self.config.train_steps:
                    break

                loss, metrics = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.metacontroller.parameters(), 1.0)
                self.optimizer.step()

                step += 1
                pbar.update(1)

                if step % log_interval == 0:
                    pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

                if step % save_interval == 0:
                    path = checkpoint_dir / f"metacontroller_step{step}.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": self.metacontroller.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": {
                            "kl_alpha": self.config.kl_alpha,
                            "controlled_layer": self.config.controlled_layer,
                        },
                    }, path)

        pbar.close()

        final_path = checkpoint_dir / "metacontroller_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.metacontroller.state_dict(),
        }, final_path)
        print(f"Saved metacontroller to {final_path}")

        return self.metacontroller
