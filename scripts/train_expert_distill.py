#!/usr/bin/env python3
"""Expert Distill: Train MetaController on frozen action expert's residual stream.

Usage:
    python scripts/train_expert_distill.py \
        --config temporal/configs/expert_distill.yaml \
        --expert checkpoints/phase1/action_expert.pt \
        --data /path/to/demonstration_data \
        --output checkpoints/expert_distill/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from temporal.data.human_motion_dataset import HumanMotionDataset, create_dataloader
from temporal.models.action_expert import ActionExpertWrapper
from temporal.models.metacontroller import MetaController
from temporal.training.expert_distill_trainer import ExpertDistillTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Distill: MetaController training")
    parser.add_argument("--config", type=str, default="temporal/configs/expert_distill.yaml")
    parser.add_argument("--expert", type=str, required=True, help="Phase 1 action expert checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to demonstration data")
    parser.add_argument("--output", type=str, default="checkpoints/expert_distill/")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load frozen action expert
    expert = ActionExpertWrapper(
        width=config["model"]["n_e"],
        depth=18,
        intervention_layer=config["model"].get("intervention_layer", 9),
    ).to(device)

    expert_ckpt = torch.load(args.expert, map_location=device)
    expert.load_state_dict(expert_ckpt["model_state_dict"])
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False
    logger.info(f"Loaded frozen action expert from {args.expert}")

    # Data
    dataset = HumanMotionDataset(
        data_dir=args.data,
        seq_len=config["data"]["seq_len"],
        overlap=config["data"].get("overlap", 0.5),
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
    )
    logger.info(f"Dataset: {len(dataset)} sequences")

    # MetaController
    metacontroller = MetaController(
        n_e=config["model"]["n_e"],
        n_z=config["model"]["n_z"],
        rank=config["model"]["rank"],
        encoder_hidden=config["model"]["encoder_hidden"],
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        metacontroller.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # Trainer
    trainer = ExpertDistillTrainer(
        expert=expert,
        metacontroller=metacontroller,
        dataloader=dataloader,
        lr=config["training"]["lr"],
        alpha=config["training"]["alpha"],
        device=device,
    )

    num_epochs = config["training"]["num_epochs"]
    save_every = config["training"].get("save_every", 10)

    for epoch in range(num_epochs):
        recon_loss, kl_loss, total_loss = trainer.train_epoch()
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Recon: {recon_loss:.6f} | KL: {kl_loss:.6f} | Total: {total_loss:.6f}"
        )

        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"metacontroller_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": metacontroller.state_dict(),
                    "loss": total_loss,
                },
                ckpt_path,
            )
            logger.info(f"Saved: {ckpt_path}")

    final_path = output_dir / "metacontroller.pt"
    torch.save(
        {"epoch": num_epochs, "model_state_dict": metacontroller.state_dict()},
        final_path,
    )
    logger.info(f"Expert Distill complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
