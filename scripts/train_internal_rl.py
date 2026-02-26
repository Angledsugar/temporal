#!/usr/bin/env python3
"""Stage 3: Train Internal RL policy."""

import argparse
import torch

from temporal.training.internal_rl_train import InternalRLTrainer
from temporal.models.transformer import CausalTransformer
from temporal.models.metacontroller import MetaController
from temporal.envs.tasks import POST_TRAINING_TASK
from temporal.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train Internal RL")
    parser.add_argument("--config", type=str, default="configs/internal_rl.yaml")
    parser.add_argument("--base-model", type=str, default="checkpoints/base_model_final.pt")
    parser.add_argument("--metacontroller", type=str, default="checkpoints/metacontroller_final.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

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

    # Load metacontroller
    metacontroller = MetaController(
        embed_dim=config.base_model.embed_dim,
        latent_dim=config.metacontroller.latent_dim,
        gru_dim=config.metacontroller.gru_dim,
        seq_embed_dim=config.metacontroller.seq_embed_dim,
        encoder_hidden=config.metacontroller.encoder_hidden,
        decoder_hidden=config.metacontroller.decoder_hidden,
        controller_rank=config.metacontroller.controller_rank,
    )
    mc_ckpt = torch.load(args.metacontroller, map_location="cpu", weights_only=True)
    metacontroller.load_state_dict(mc_ckpt["model_state_dict"])

    print(f"Post-training task: {POST_TRAINING_TASK}")

    trainer = InternalRLTrainer(
        base_model=base_model,
        metacontroller=metacontroller,
        config=config.internal_rl,
        base_config=config.base_model,
        mc_config=config.metacontroller,
        task=POST_TRAINING_TASK,
        device=args.device,
    )
    policy = trainer.train(checkpoint_dir=config.checkpoint_dir)
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
