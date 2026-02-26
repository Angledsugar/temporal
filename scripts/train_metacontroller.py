#!/usr/bin/env python3
"""Stage 2: Train metacontroller on frozen base model."""

import argparse
import torch

from temporal.training.metacontroller_train import MetacontrollerTrainer
from temporal.models.transformer import CausalTransformer
from temporal.data.dataset import TrajectoryDataset
from temporal.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train metacontroller")
    parser.add_argument("--config", type=str, default="configs/metacontroller.yaml")
    parser.add_argument("--base-model", type=str, default="checkpoints/base_model_final.pt")
    parser.add_argument("--data", type=str, default="data/pretraining_data.npz")
    parser.add_argument("--kl-alpha", type=float, default=None,
                        help="Override KL alpha from config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.kl_alpha is not None:
        config.metacontroller.kl_alpha = args.kl_alpha

    # Load base model
    base_model = CausalTransformer(
        obs_dim=config.base_model.obs_dim,
        num_actions=config.base_model.num_actions,
        embed_dim=config.base_model.embed_dim,
        num_layers=config.base_model.num_layers,
        num_heads=config.base_model.num_heads,
        head_dim=config.base_model.head_dim,
        mlp_dim=config.base_model.mlp_dim,
    )
    ckpt = torch.load(args.base_model, map_location="cpu", weights_only=True)
    base_model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded base model")

    dataset = TrajectoryDataset(args.data, max_length=config.base_model.max_seq_len)

    trainer = MetacontrollerTrainer(
        base_model, config.metacontroller, config.base_model, device=args.device,
    )
    mc = trainer.train(dataset, checkpoint_dir=config.checkpoint_dir)
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
