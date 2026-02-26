#!/usr/bin/env python3
"""Stage 1: Train base autoregressive model."""

import argparse
import torch

from internalrl.training.pretrain import BaseModelTrainer
from internalrl.data.dataset import TrajectoryDataset
from internalrl.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train base model")
    parser.add_argument("--config", type=str, default="configs/base_model.yaml")
    parser.add_argument("--data", type=str, default="data/pretraining_data.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    dataset = TrajectoryDataset(args.data, max_length=config.base_model.max_seq_len)
    print(f"Dataset: {len(dataset)} trajectories")

    trainer = BaseModelTrainer(config.base_model, device=args.device)
    model = trainer.train(
        dataset,
        checkpoint_dir=config.checkpoint_dir,
    )
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
