#!/usr/bin/env python3
"""Run the complete pipeline: data collection → training → evaluation.

Usage:
    uv run python scripts/run_all.py --quick  # Quick test with small config
    uv run python scripts/run_all.py           # Full training
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from internalrl.data.collector import collect_trajectories
from internalrl.data.dataset import TrajectoryDataset
from internalrl.envs.tasks import PRETRAINING_TASKS, POST_TRAINING_TASK
from internalrl.models.transformer import CausalTransformer
from internalrl.models.metacontroller import MetaController
from internalrl.models.rl_policy import CausalSSMPolicy
from internalrl.training.pretrain import BaseModelTrainer
from internalrl.training.metacontroller_train import MetacontrollerTrainer
from internalrl.training.internal_rl_train import InternalRLTrainer
from internalrl.evaluation.linear_probing import run_linear_probing
from internalrl.evaluation.beta_analysis import analyze_beta_alignment
from internalrl.evaluation.rl_evaluation import evaluate_policy
from internalrl.utils.config import Config, BaseModelConfig, MetacontrollerConfig, InternalRLConfig


def get_quick_config() -> Config:
    """Get a small config for quick testing."""
    cfg = Config()

    # Smaller base model
    cfg.base_model.num_layers = 4
    cfg.base_model.embed_dim = 128
    cfg.base_model.num_heads = 4
    cfg.base_model.head_dim = 32
    cfg.base_model.mlp_dim = 256
    cfg.base_model.train_steps = 2000
    cfg.base_model.batch_size = 64
    cfg.base_model.max_seq_len = 100

    # Smaller metacontroller
    cfg.metacontroller.controlled_layer = 2  # L/2
    cfg.metacontroller.train_steps = 1000
    cfg.metacontroller.batch_size = 64

    # Smaller RL
    cfg.internal_rl.policy_embed_dim = 128
    cfg.internal_rl.train_steps = 500
    cfg.internal_rl.batch_size = 32

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run full Internal RL pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick test with small config")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-data", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    device = args.device
    config = get_quick_config() if args.quick else Config()

    data_path = Path("data/pretraining_data.npz")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # =====================
    # Step 0: Collect Data
    # =====================
    if not args.skip_data and not data_path.exists():
        print("\n" + "=" * 60)
        print("STEP 0: Collecting expert trajectory data")
        print("=" * 60)
        episodes = 500 if args.quick else 10000
        collect_trajectories(
            tasks=PRETRAINING_TASKS,
            episodes_per_task=episodes,
            epsilon=0.0,
            seed=config.seed,
            save_path=data_path,
        )

    dataset = TrajectoryDataset(str(data_path), max_length=config.base_model.max_seq_len)
    print(f"Dataset: {len(dataset)} trajectories")

    # =====================
    # Stage 1: Base Model
    # =====================
    base_model_path = ckpt_dir / "base_model_final.pt"
    if not args.skip_stage1:
        print("\n" + "=" * 60)
        print("STAGE 1: Training base autoregressive model")
        print("=" * 60)
        trainer = BaseModelTrainer(config.base_model, device=device)
        base_model = trainer.train(dataset, checkpoint_dir=ckpt_dir)
    else:
        base_model = CausalTransformer(
            obs_dim=config.base_model.obs_dim,
            num_actions=config.base_model.num_actions,
            embed_dim=config.base_model.embed_dim,
            num_layers=config.base_model.num_layers,
            num_heads=config.base_model.num_heads,
            head_dim=config.base_model.head_dim,
            mlp_dim=config.base_model.mlp_dim,
        )
        if base_model_path.exists():
            ckpt = torch.load(base_model_path, map_location="cpu", weights_only=True)
            base_model.load_state_dict(ckpt["model_state_dict"])
        base_model = base_model.to(device)

    # Stage 1 Verification: Linear Probing
    print("\n--- Stage 1 Verification: Linear Probing ---")
    probing_steps = 1000 if args.quick else 8000
    probe_results = run_linear_probing(
        base_model, dataset, train_steps=probing_steps,
        batch_size=64 if args.quick else 512,
        device=device, save_dir=results_dir,
    )

    # =====================
    # Stage 2: Metacontroller
    # =====================
    mc_path = ckpt_dir / "metacontroller_final.pt"
    if not args.skip_stage2:
        print("\n" + "=" * 60)
        print("STAGE 2: Training metacontroller")
        print("=" * 60)
        mc_trainer = MetacontrollerTrainer(
            base_model, config.metacontroller, config.base_model, device=device,
        )
        metacontroller = mc_trainer.train(dataset, checkpoint_dir=ckpt_dir)
    else:
        metacontroller = MetaController(
            embed_dim=config.base_model.embed_dim,
            latent_dim=config.metacontroller.latent_dim,
            gru_dim=config.metacontroller.gru_dim,
            seq_embed_dim=config.metacontroller.seq_embed_dim,
            encoder_hidden=config.metacontroller.encoder_hidden,
            decoder_hidden=config.metacontroller.decoder_hidden,
            controller_rank=config.metacontroller.controller_rank,
        )
        if mc_path.exists():
            mc_ckpt = torch.load(mc_path, map_location="cpu", weights_only=True)
            metacontroller.load_state_dict(mc_ckpt["model_state_dict"])
        metacontroller = metacontroller.to(device)

    # Stage 2 Verification: Beta Analysis
    print("\n--- Stage 2 Verification: Beta Analysis ---")
    beta_metrics = analyze_beta_alignment(
        base_model, metacontroller, dataset,
        controlled_layer=config.metacontroller.controlled_layer,
        device=device, save_dir=results_dir,
    )

    # =====================
    # Stage 3: Internal RL
    # =====================
    if not args.skip_stage3:
        print("\n" + "=" * 60)
        print("STAGE 3: Training Internal RL policy")
        print("=" * 60)
        rl_trainer = InternalRLTrainer(
            base_model=base_model,
            metacontroller=metacontroller,
            config=config.internal_rl,
            base_config=config.base_model,
            mc_config=config.metacontroller,
            task=POST_TRAINING_TASK,
            device=device,
        )
        policy = rl_trainer.train(checkpoint_dir=ckpt_dir)

        # Stage 3 Verification
        print("\n--- Stage 3 Verification: RL Evaluation ---")
        eval_results = evaluate_policy(
            policy, base_model, metacontroller, POST_TRAINING_TASK,
            controlled_layer=config.metacontroller.controlled_layer,
            device=device,
            num_episodes=50 if args.quick else 100,
        )
        print(f"  Success rate: {eval_results['success_rate']:.4f}")
        print(f"  Avg abstract steps: {eval_results['avg_abstract_steps']:.1f}")
        print(f"  Avg raw steps: {eval_results['avg_raw_steps']:.1f}")

    # =====================
    # Summary
    # =====================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - Summary")
    print("=" * 60)
    print("\nStage 1 - Linear Probing Accuracy:")
    for layer, acc in sorted(probe_results.items()):
        marker = " ✓" if acc > 0.4 else ""
        print(f"  Layer {layer}: {acc:.4f}{marker}")
    print(f"\nStage 2 - Beta Analysis:")
    for k, v in beta_metrics.items():
        print(f"  {k}: {v:.4f}")
    if not args.skip_stage3:
        print(f"\nStage 3 - Internal RL:")
        for k, v in eval_results.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
