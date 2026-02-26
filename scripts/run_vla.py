"""Run VLA metacontroller pipeline.

Supports both π0.5 (OpenPI) and Groot N1.6 models.

Usage:
    # π0.5 with dummy data (structure verification)
    uv run python scripts/run_vla.py --model pi05 --dummy-data --quick

    # Groot with dummy data
    uv run python scripts/run_vla.py --model groot --dummy-data --quick

    # π0.5 with real data and checkpoint
    uv run python scripts/run_vla.py --model pi05 \\
        --config configs/pi05_metacontroller.yaml

    # Groot with real data and checkpoint
    uv run python scripts/run_vla.py --model groot \\
        --config configs/groot_metacontroller.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal.utils.config import VLAConfig
from temporal.vla.data.dummy_dataset import DummyResidualDataset
from temporal.vla.models.metacontroller_vla import (
    VLAMetaController,
    VLAMetaControllerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_mc_config(model_type: str, quick: bool = False) -> VLAMetaControllerConfig:
    """Create MC config for dummy data testing."""
    if model_type == "pi05":
        controlled_layer = 9
    elif model_type == "groot":
        controlled_layer = 12
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    config = VLAMetaControllerConfig(
        embed_dim=64 if quick else 2048,  # Small for quick mode
        latent_dim=8 if quick else 16,
        gru_dim=16 if quick else 64,
        seq_embed_dim=16 if quick else 64,
        encoder_hidden=32 if quick else 128,
        decoder_hidden=16 if quick else 64,
        controller_rank=8 if quick else 32,
        controlled_layer=controlled_layer,
        train_steps=10 if quick else 64000,
        batch_size=2,
        lr=1e-3 if quick else 1e-4,
    )
    return config


def run_dummy_training(model_type: str, quick: bool = True) -> None:
    """Run metacontroller training on dummy residual data.

    This bypasses the VLA model entirely — it tests the MC training
    pipeline with random residual streams.
    """
    logger.info(f"Running dummy MC training for {model_type} (quick={quick})")

    config = create_dummy_mc_config(model_type, quick=quick)
    embed_dim = config.embed_dim

    # Create dummy dataset
    dataset = DummyResidualDataset(
        num_samples=20 if quick else 100,
        seq_len=10 if quick else 50,
        embed_dim=embed_dim,
        action_dim=32 if model_type == "pi05" else 29,
        action_horizon=50 if model_type == "pi05" else 16,
    )

    # Create metacontroller
    mc = VLAMetaController(config)
    logger.info(
        f"MC params: {sum(p.numel() for p in mc.parameters()):,} "
        f"(embed_dim={embed_dim})"
    )

    # Simple training loop (no wrapper needed for dummy data)
    optimizer = torch.optim.AdamW(mc.parameters(), lr=config.lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    mc.train()
    for step, batch in enumerate(dataloader):
        if step >= config.train_steps:
            break

        residual = batch["residual"]
        mc_out = mc(residual)

        # Dummy action loss (just use e_controlled norm)
        action_loss = mc_out["e_controlled"].pow(2).mean()
        kl_loss = mc_out["kl_loss"]
        total_loss = action_loss + config.kl_alpha * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
        optimizer.step()

        if step % max(1, config.train_steps // 5) == 0:
            logger.info(
                f"Step {step}/{config.train_steps}: "
                f"loss={total_loss.item():.4f} "
                f"action={action_loss.item():.4f} "
                f"kl={kl_loss.item():.4f} "
                f"beta={mc_out['beta_seq'].mean().item():.3f}"
            )

    logger.info("Dummy MC training complete!")

    # Save checkpoint
    ckpt_dir = Path(f"checkpoints/vla/{model_type}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "metacontroller_vla_dummy.pt"
    torch.save({"model_state_dict": mc.state_dict()}, ckpt_path)
    logger.info(f"Saved dummy checkpoint to {ckpt_path}")


def run_with_config(config_path: str) -> None:
    """Run training with a YAML config file (requires VLA model)."""
    config = VLAConfig.from_yaml(config_path)
    model_type = config.model.type

    logger.info(f"Loading {model_type} model from config: {config_path}")

    if model_type == "pi05":
        from temporal.vla.models.pi05_wrapper import Pi05Wrapper, Pi05WrapperConfig

        wrapper_config = Pi05WrapperConfig(
            checkpoint_path=config.model.checkpoint_path,
            openpi_path=config.model.openpi_path,
            paligemma_variant=config.model.paligemma_variant,
            action_expert_variant=config.model.action_expert_variant,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            dtype=config.model.dtype,
            controlled_layer=config.metacontroller.controlled_layer,
            gradient_checkpointing=config.model.gradient_checkpointing,
        )
        wrapper = Pi05Wrapper(wrapper_config)

    elif model_type == "groot":
        from temporal.vla.models.groot_wrapper import GrootWrapper, GrootWrapperConfig

        wrapper_config = GrootWrapperConfig(
            checkpoint_path=config.model.checkpoint_path,
            groot_path=config.model.groot_path,
            model_name=config.model.model_name,
            select_layer=config.model.select_layer,
            controlled_layer=config.metacontroller.controlled_layer,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            state_dim=config.model.state_dim,
            dtype=config.model.dtype,
            gradient_checkpointing=config.model.gradient_checkpointing,
        )
        wrapper = GrootWrapper(wrapper_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create metacontroller
    mc_config = VLAMetaControllerConfig(
        embed_dim=config.metacontroller.embed_dim,
        latent_dim=config.metacontroller.latent_dim,
        gru_dim=config.metacontroller.gru_dim,
        seq_embed_dim=config.metacontroller.seq_embed_dim,
        encoder_hidden=config.metacontroller.encoder_hidden,
        decoder_hidden=config.metacontroller.decoder_hidden,
        controller_rank=config.metacontroller.controller_rank,
        controlled_layer=config.metacontroller.controlled_layer,
        kl_alpha=config.metacontroller.kl_alpha,
        train_steps=config.metacontroller.train_steps,
        batch_size=config.metacontroller.batch_size,
        lr=config.metacontroller.lr,
        weight_decay=config.metacontroller.weight_decay,
    )
    mc = VLAMetaController(mc_config)

    # Create trainer
    from temporal.vla.training.vla_metacontroller_train import VLAMetacontrollerTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = VLAMetacontrollerTrainer(
        wrapper=wrapper,
        metacontroller=mc,
        config=mc_config,
        device=device,
    )

    # Create dataset
    if config.data.type == "dummy":
        if model_type == "pi05":
            from temporal.vla.data.dummy_dataset import DummyPi05Dataset
            dataset = DummyPi05Dataset(num_samples=config.data.num_samples)
        else:
            from temporal.vla.data.dummy_dataset import DummyGrootDataset
            dataset = DummyGrootDataset(num_samples=config.data.num_samples)
    elif config.data.type == "lerobot":
        local_path = config.data.local_path or None
        repo_id = config.data.repo_id
        if model_type == "groot":
            from temporal.vla.data.groot_dataset import GrootLeRobotDataset
            dataset = GrootLeRobotDataset(
                repo_id=repo_id,
                local_path=local_path,
                action_horizon=config.model.action_horizon,
                image_size=config.data.image_size,
            )
        else:
            from temporal.vla.data.pi05_dataset import Pi05LeRobotDataset
            dataset = Pi05LeRobotDataset(
                repo_id=repo_id,
                local_path=local_path,
                action_horizon=config.model.action_horizon,
                image_size=config.data.image_size,
                max_token_len=config.model.max_token_len,
            )
        logger.info(f"Loaded LeRobot dataset: {len(dataset)} samples")
    else:
        raise ValueError(f"Unknown data type: {config.data.type}")

    # Train
    trainer.train(
        dataset=dataset,
        checkpoint_dir=config.checkpoint_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="VLA Metacontroller Pipeline")
    parser.add_argument(
        "--model", choices=["pi05", "groot"], default="pi05",
        help="VLA model type (default: pi05)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file path (loads VLA model)"
    )
    parser.add_argument(
        "--dummy-data", action="store_true",
        help="Use dummy data (no VLA model needed)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (small dimensions, few steps)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)"
    )
    args = parser.parse_args()

    if args.dummy_data:
        run_dummy_training(args.model, quick=args.quick)
    elif args.config:
        run_with_config(args.config)
    else:
        parser.error("Specify either --dummy-data or --config")


if __name__ == "__main__":
    main()
