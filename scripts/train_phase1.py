#!/usr/bin/env python3
"""Phase 1: Pretrain action expert on human motion data with flow matching.

Trains Gemma-300M on Inter-X + Ego4D + UniHand data using conditional
flow-matching. After this phase, theta is FROZEN permanently.

Usage:
    # Train with Inter-X only (for testing)
    python scripts/train_phase1.py \
        --config configs/phase1_interx.yaml

    # Train with full multi-dataset config
    python scripts/train_phase1.py \
        --config configs/phase1_pretrain.yaml

    # Train with single dataset (quick test)
    python scripts/train_phase1.py \
        --data dataset/interx_npz \
        --max-steps 1000 \
        --batch-size 16

    # Resume from checkpoint
    python scripts/train_phase1.py \
        --config configs/phase1_interx.yaml \
        --resume checkpoints/phase1/action_expert_step10000.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Flow-matching pretraining")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data", type=str, default=None, help="Single dataset path (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build training config
    from temporal.training.phase1_trainer import Phase1Config, Phase1Trainer, TextEncoder

    training_cfg = config.get("training", {})
    phase1_config = Phase1Config(
        learning_rate=args.lr or training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        max_steps=args.max_steps or training_cfg.get("max_steps", 500_000),
        grad_clip=training_cfg.get("grad_clip", 1.0),
        flow_matching_steps=config.get("flow_matching", {}).get("num_steps", 10),
        sigma_min=config.get("flow_matching", {}).get("sigma_min", 0.001),
        batch_size=args.batch_size or training_cfg.get("batch_size", 256),
        log_every=config.get("logging", {}).get("log_every", 100),
        save_every=training_cfg.get("checkpoint", {}).get("save_every", training_cfg.get("save_every", 10_000)),
        output_dir=args.output or training_cfg.get("checkpoint", {}).get("output_dir", "checkpoints/phase1/"),
        use_wandb=args.wandb,
        wandb_project=config.get("logging", {}).get("wandb_project", "temporal-phase1"),
        dtype=training_cfg.get("dtype", "bfloat16"),
        grad_accum_steps=training_cfg.get("grad_accum_steps", 1),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
    )

    Path(phase1_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    data_cfg = config.get("data", {})
    datasets_list = data_cfg.get("datasets", [])

    if args.data:
        # Single dataset mode (CLI override)
        from temporal.data.human_motion_dataset import create_dataloader
        dataloader = create_dataloader(
            data_root=args.data,
            split="train",
            batch_size=phase1_config.batch_size,
            num_workers=data_cfg.get("num_workers", 4),
            max_length=data_cfg.get("max_length", 256),
        )
    elif datasets_list:
        # Multi-dataset mode (from config)
        from temporal.data.human_motion_dataset import create_weighted_dataloader
        dataset_configs = [
            {"path": ds["path"], "weight": ds.get("weight", 1.0), "name": ds.get("name", "")}
            for ds in datasets_list
        ]
        logger.info(f"Loading {len(dataset_configs)} datasets with weighted sampling:")
        dataloader = create_weighted_dataloader(
            dataset_configs=dataset_configs,
            split="train",
            batch_size=phase1_config.batch_size,
            num_workers=data_cfg.get("num_workers", 4),
            max_length=data_cfg.get("max_length", 256),
            action_dim=data_cfg.get("action_dim", 7),
            proprio_dim=data_cfg.get("proprio_dim", 14),
        )
    else:
        parser.error("Must provide --data or config with data.datasets")

    # Model
    from temporal.models.action_expert import ActionExpert, ActionExpertConfig

    model_cfg = config.get("model", {})
    expert_config = ActionExpertConfig(
        width=model_cfg.get("width", 1024),
        depth=model_cfg.get("depth", 18),
        mlp_dim=model_cfg.get("mlp_dim", 4096),
        num_heads=model_cfg.get("num_heads", 8),
        num_kv_heads=model_cfg.get("num_kv_heads", 1),
        head_dim=model_cfg.get("head_dim", 256),
        controlled_layer=model_cfg.get("controlled_layer", 9),
        action_dim=data_cfg.get("action_dim", 7),
    )

    model = ActionExpert(expert_config).to(device)
    n_params = model.param_count()
    logger.info(f"ActionExpert: {n_params / 1e6:.1f}M parameters")

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from {args.resume}, step {start_step}")

    # Text encoder
    text_encoder = TextEncoder(device=str(device))

    # Train
    trainer = Phase1Trainer(
        model=model,
        dataloader=dataloader,
        config=phase1_config,
        text_encoder=text_encoder,
    )

    # If resuming, restore optimizer/scheduler state
    if args.resume and "optimizer_state_dict" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Restored optimizer and scheduler state")

    trainer.train()

    # Save final frozen model
    final_path = Path(phase1_config.output_dir) / "action_expert_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(expert_config),
    }, final_path)
    logger.info(f"Phase 1 complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
