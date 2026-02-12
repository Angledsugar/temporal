#!/usr/bin/env python3
"""Expert Distill: Train MetaController on frozen action expert's residual stream.

Usage:
    uv run python scripts/train_expert_distill.py
    uv run python scripts/train_expert_distill.py --config configs/expert_distill.yaml
    uv run python scripts/train_expert_distill.py --expert checkpoint/pi05_droid --data dataset/droid_100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from temporal.data.human_motion_dataset import create_dataloader
from temporal.models.action_expert import ActionExpertWrapper
from temporal.models.metacontroller import MetaController
from temporal.training.expert_distill_trainer import ExpertDistillConfig, ExpertDistillTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Distill: MetaController training")
    parser.add_argument("--config", type=str, default="configs/expert_distill.yaml")
    parser.add_argument("--expert", type=str, default=None, help="Override expert checkpoint path")
    parser.add_argument("--data", type=str, default=None, help="Override data root path")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from metacontroller checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve paths (CLI args override config values)
    expert_path = args.expert or config.get("expert_checkpoint", "checkpoint/pi05_droid")
    data_root = args.data or config.get("data_root", "dataset/droid_100")
    output_dir = args.output or config.get("output_dir", "checkpoint/expert_distill")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model config
    model_cfg = config["model"]
    n_e = model_cfg["n_e"]
    n_z = model_cfg["n_z"]
    rank = model_cfg["rank"]
    encoder_hidden = model_cfg["encoder_hidden"]
    controlled_layer = model_cfg.get("controlled_layer", 9)

    # Load frozen action expert
    logger.info(f"Loading action expert from {expert_path}")
    expert = ActionExpertWrapper(
        checkpoint_path=expert_path,
        controlled_layer=controlled_layer,
    )
    logger.info(f"Action expert loaded (width={expert.width}, depth={expert.depth}, layer={controlled_layer})")

    # Data
    data_cfg = config["data"]
    dataloader = create_dataloader(
        data_root=data_root,
        split=data_cfg.get("split", "train"),
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        max_length=data_cfg.get("max_length", 256),
    )
    logger.info(f"DataLoader ready: batch_size={data_cfg['batch_size']}, data_root={data_root}")

    # MetaController
    metacontroller = MetaController(
        n_e=n_e,
        n_z=n_z,
        rank=rank,
        encoder_hidden=encoder_hidden,
    ).to(device)
    param_count = sum(p.numel() for p in metacontroller.parameters())
    logger.info(f"MetaController: {param_count:,} parameters")

    # Resume if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        metacontroller.load_state_dict(ckpt)
        logger.info(f"Resumed from {args.resume}")

    # Training config
    train_cfg = config["training"]
    distill_config = ExpertDistillConfig(
        alpha=train_cfg["alpha"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_steps=train_cfg["max_steps"],
        grad_clip=train_cfg.get("grad_clip", 1.0),
        controlled_layer=controlled_layer,
        log_every=train_cfg.get("log_every", 100),
        save_every=train_cfg.get("save_every", 5000),
        output_dir=output_dir,
    )

    # Trainer
    trainer = ExpertDistillTrainer(
        action_expert=expert,
        metacontroller=metacontroller,
        dataloader=dataloader,
        config=distill_config,
    )

    logger.info("Starting Expert Distill training...")
    logger.info(f"  alpha={distill_config.alpha}, lr={distill_config.learning_rate}")
    logger.info(f"  max_steps={distill_config.max_steps}, save_every={distill_config.save_every}")
    logger.info(f"  output_dir={output_dir}")

    trainer.train()


if __name__ == "__main__":
    main()
