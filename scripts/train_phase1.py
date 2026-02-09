#!/usr/bin/env python3
"""Phase 1: Pretrain action expert on human motion data with flow matching.

Usage:
    python scripts/train_phase1.py \
        --config temporal/configs/phase1_pretrain.yaml \
        --data /path/to/human_motion_data \
        --output checkpoints/phase1/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from temporal.data.human_motion_dataset import HumanMotionDataset, create_dataloader
from temporal.models.action_expert import ActionExpertWrapper
from temporal.training.phase1_trainer import Phase1Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Flow-matching pretraining")
    parser.add_argument("--config", type=str, default="temporal/configs/phase1_pretrain.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to human motion data")
    parser.add_argument("--output", type=str, default="checkpoints/phase1/")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    dataset = HumanMotionDataset(
        data_dir=args.data,
        seq_len=config["data"]["seq_len"],
        overlap=config["data"]["overlap"],
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    logger.info(f"Dataset: {len(dataset)} sequences, batch_size={config['data']['batch_size']}")

    # Model
    model = ActionExpertWrapper(
        width=config["model"]["width"],
        depth=config["model"]["depth"],
        intervention_layer=config["model"]["intervention_layer"],
    ).to(device)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        dataloader=dataloader,
        lr=config["training"]["lr"],
        device=device,
    )

    # Train
    num_epochs = config["training"]["num_epochs"]
    save_every = config["training"].get("save_every", 10)

    for epoch in range(start_epoch, num_epochs):
        loss = trainer.train_epoch()
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"action_expert_epoch{epoch+1}.pt"
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "loss": loss},
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = output_dir / "action_expert.pt"
    torch.save({"epoch": num_epochs, "model_state_dict": model.state_dict()}, final_path)
    logger.info(f"Phase 1 complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
