#!/usr/bin/env python3
"""Phase 3: Internal RL training on the internal MDP.

Usage:
    python scripts/train_phase3.py \
        --config temporal/configs/phase3_internal_rl.yaml \
        --expert checkpoints/phase1/action_expert.pt \
        --meta checkpoints/expert_distill/metacontroller.pt \
        --env sim_manipulation \
        --output checkpoints/phase3/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from temporal.envs.internal_env import InternalEnv
from temporal.models.action_expert import ActionExpertWrapper
from temporal.models.internal_rl_policy import CausalRLPolicy
from temporal.models.metacontroller import MetaController
from temporal.training.phase3_trainer import Phase3Trainer
from temporal.utils.metrics import temporal_contraction_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_sim_env(env_name: str) -> gym.Env:
    """Create simulation environment.

    TODO: Replace with actual sim environment (ManiSkill3, SIMPLER, etc.)
    """
    return gym.make(env_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Internal RL")
    parser.add_argument("--config", type=str, default="temporal/configs/phase3_internal_rl.yaml")
    parser.add_argument("--expert", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym env name (placeholder)")
    parser.add_argument("--output", type=str, default="checkpoints/phase3/")
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
    logger.info("Loaded frozen action expert")

    # Load frozen MetaController (decoder only used)
    metacontroller = MetaController(
        n_e=config["model"]["n_e"],
        n_z=config["model"]["n_z"],
        rank=config["model"].get("rank", 32),
    ).to(device)
    meta_ckpt = torch.load(args.meta, map_location=device)
    metacontroller.load_state_dict(meta_ckpt["model_state_dict"])
    metacontroller.eval()
    for p in metacontroller.parameters():
        p.requires_grad = False
    logger.info("Loaded frozen MetaController")

    # Simulation environment
    sim_env = make_sim_env(args.env)

    # Internal MDP environment
    internal_env = InternalEnv(
        action_expert=expert,
        metacontroller=metacontroller,
        sim_env=sim_env,
        beta_threshold=config["env"]["beta_threshold"],
        max_primitive_steps_per_z=config["env"]["max_primitive_steps_per_z"],
    )

    # RL Policy
    policy = CausalRLPolicy(
        n_e=config["model"]["n_e"],
        n_z=config["model"]["n_z"],
        hidden=config["model"]["policy_hidden"],
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # Trainer
    trainer = Phase3Trainer(
        policy=policy,
        env=internal_env,
        lr=config["training"]["lr"],
        gamma=config["training"]["gamma"],
        device=device,
    )

    num_episodes = config["training"]["num_episodes"]
    eval_every = config["training"].get("eval_every", 50)
    save_every = config["training"].get("save_every", 100)

    best_reward = -float("inf")

    for ep in range(num_episodes):
        ep_reward, ep_len, info = trainer.train_episode()

        if (ep + 1) % 10 == 0:
            prim_steps = info.get("total_primitive_steps", ep_len)
            tcr = temporal_contraction_ratio(prim_steps, ep_len)
            logger.info(
                f"Episode {ep+1}/{num_episodes} | "
                f"Reward: {ep_reward:.2f} | Steps (z): {ep_len} | "
                f"Prim steps: {prim_steps} | TCR: {tcr:.1f}x"
            )

        if (ep + 1) % eval_every == 0:
            eval_reward = trainer.evaluate(num_episodes=5)
            logger.info(f"  Eval reward: {eval_reward:.2f}")

            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = output_dir / "rl_policy_best.pt"
                torch.save({"model_state_dict": policy.state_dict()}, best_path)
                logger.info(f"  New best! Saved: {best_path}")

        if (ep + 1) % save_every == 0:
            ckpt_path = output_dir / f"rl_policy_ep{ep+1}.pt"
            torch.save(
                {"episode": ep + 1, "model_state_dict": policy.state_dict()},
                ckpt_path,
            )

    final_path = output_dir / "rl_policy.pt"
    torch.save({"model_state_dict": policy.state_dict()}, final_path)
    logger.info(f"Phase 3 complete. Final: {final_path}, Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
