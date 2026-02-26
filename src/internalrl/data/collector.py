"""Expert trajectory collection for pretraining data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..envs.expert import ExpertPolicy
from ..envs.gridworld import GridworldPinpad
from ..envs.tasks import GRID_SIZE, MAX_STEPS, NUM_COLORS, NUM_WALLS


def collect_trajectories(
    tasks: list[list[int]],
    episodes_per_task: int = 10000,
    epsilon: float = 0.0,
    seed: int = 0,
    save_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Collect expert trajectories for a set of tasks.

    Args:
        tasks: List of tasks (each a list of color indices).
        episodes_per_task: Number of episodes per task.
        epsilon: Expert noise level.
        seed: Random seed.
        save_path: If provided, save to this .npz path.

    Returns:
        Dictionary with arrays for observations, actions, subgoals, lengths.
    """
    env = GridworldPinpad(
        grid_size=GRID_SIZE,
        num_colors=NUM_COLORS,
        num_walls=NUM_WALLS,
        max_steps=MAX_STEPS,
        seed=seed,
    )
    expert = ExpertPolicy(epsilon=epsilon, seed=seed + 1)

    all_obs = []
    all_act = []
    all_sg = []
    all_lengths = []
    success_count = 0
    total_count = 0

    for task_idx, task in enumerate(tasks):
        for ep in tqdm(
            range(episodes_per_task),
            desc=f"Task {task_idx+1}/{len(tasks)}",
            leave=False,
        ):
            traj = expert.generate_trajectory(env, task)
            if traj is None:
                continue

            all_obs.append(traj["observations"])
            all_act.append(traj["actions"])
            all_sg.append(traj["subgoals"])
            all_lengths.append(traj["length"])
            if traj["success"]:
                success_count += 1
            total_count += 1

    print(f"Collected {total_count} trajectories, "
          f"success rate: {success_count / max(total_count, 1):.3f}")

    # Use object arrays to handle variable-length trajectories
    result = {
        "observations": np.array(all_obs, dtype=object),
        "actions": np.array(all_act, dtype=object),
        "subgoals": np.array(all_sg, dtype=object),
        "lengths": np.array(all_lengths, dtype=np.int64),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **result)
        print(f"Saved to {save_path}")

    return result
