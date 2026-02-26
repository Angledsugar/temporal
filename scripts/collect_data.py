#!/usr/bin/env python3
"""Collect expert trajectory data for pretraining."""

import argparse
from pathlib import Path

from temporal.data.collector import collect_trajectories
from temporal.envs.tasks import PRETRAINING_TASKS


def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    parser.add_argument("--episodes-per-task", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/pretraining_data.npz")
    args = parser.parse_args()

    print(f"Collecting data for {len(PRETRAINING_TASKS)} tasks, "
          f"{args.episodes_per_task} episodes each...")

    collect_trajectories(
        tasks=PRETRAINING_TASKS,
        episodes_per_task=args.episodes_per_task,
        epsilon=args.epsilon,
        seed=args.seed,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
