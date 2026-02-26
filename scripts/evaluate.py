#!/usr/bin/env python3
"""Run all evaluations: linear probing, beta analysis, RL evaluation."""

import argparse
import torch

from internalrl.models.transformer import CausalTransformer
from internalrl.models.metacontroller import MetaController
from internalrl.models.rl_policy import CausalSSMPolicy
from internalrl.data.dataset import TrajectoryDataset
from internalrl.evaluation.linear_probing import run_linear_probing
from internalrl.evaluation.beta_analysis import analyze_beta_alignment
from internalrl.evaluation.rl_evaluation import evaluate_policy
from internalrl.envs.tasks import POST_TRAINING_TASK
from internalrl.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument("--config", type=str, default="configs/base_model.yaml")
    parser.add_argument("--base-model", type=str, default="checkpoints/base_model_final.pt")
    parser.add_argument("--metacontroller", type=str, default=None)
    parser.add_argument("--rl-policy", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/pretraining_data.npz")
    parser.add_argument("--eval", nargs="+", default=["probing", "beta", "rl"],
                        choices=["probing", "beta", "rl"])
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

    dataset = TrajectoryDataset(args.data, max_length=config.base_model.max_seq_len)

    # Stage 1: Linear Probing
    if "probing" in args.eval:
        print("\n=== Stage 1 Verification: Linear Probing ===")
        results = run_linear_probing(
            base_model, dataset, device=args.device, save_dir="results",
        )
        for layer, acc in sorted(results.items()):
            print(f"  Layer {layer}: {acc:.4f}")

    # Stage 2: Beta Analysis
    if "beta" in args.eval and args.metacontroller:
        print("\n=== Stage 2 Verification: Beta Analysis ===")
        mc = MetaController(
            embed_dim=config.base_model.embed_dim,
            latent_dim=config.metacontroller.latent_dim,
            gru_dim=config.metacontroller.gru_dim,
            seq_embed_dim=config.metacontroller.seq_embed_dim,
            encoder_hidden=config.metacontroller.encoder_hidden,
            decoder_hidden=config.metacontroller.decoder_hidden,
            controller_rank=config.metacontroller.controller_rank,
        )
        mc_ckpt = torch.load(args.metacontroller, map_location="cpu", weights_only=True)
        mc.load_state_dict(mc_ckpt["model_state_dict"])

        metrics = analyze_beta_alignment(
            base_model, mc, dataset,
            controlled_layer=config.metacontroller.controlled_layer,
            device=args.device, save_dir="results",
        )
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Stage 3: RL Evaluation
    if "rl" in args.eval and args.rl_policy and args.metacontroller:
        print("\n=== Stage 3 Verification: RL Evaluation ===")
        mc = MetaController(
            embed_dim=config.base_model.embed_dim,
            latent_dim=config.metacontroller.latent_dim,
            gru_dim=config.metacontroller.gru_dim,
            seq_embed_dim=config.metacontroller.seq_embed_dim,
            encoder_hidden=config.metacontroller.encoder_hidden,
            decoder_hidden=config.metacontroller.decoder_hidden,
            controller_rank=config.metacontroller.controller_rank,
        )
        mc_ckpt = torch.load(args.metacontroller, map_location="cpu", weights_only=True)
        mc.load_state_dict(mc_ckpt["model_state_dict"])

        policy = CausalSSMPolicy(
            embed_dim=config.base_model.embed_dim,
            latent_dim=config.metacontroller.latent_dim,
            hidden_dim=config.internal_rl.policy_embed_dim,
            num_layers=config.internal_rl.policy_depth,
        )
        rl_ckpt = torch.load(args.rl_policy, map_location="cpu", weights_only=True)
        policy.load_state_dict(rl_ckpt["model_state_dict"])

        results = evaluate_policy(
            policy, base_model, mc, POST_TRAINING_TASK,
            controlled_layer=config.metacontroller.controlled_layer,
            device=args.device, num_episodes=100,
        )
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
