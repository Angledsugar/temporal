"""Stage 3 Verification: Internal RL evaluation.

Evaluates the trained RL policy on the post-training task and
compares against baselines.

Expected result (Fig 8):
- Internal RL achieves high success rate
- Raw action RL (GRPO) fails within 1M episodes
- Internal RL without temporal abstraction (β=1) has high initial
  success but fails at credit assignment
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..models.rl_policy import CausalSSMPolicy
from ..models.transformer import CausalTransformer
from ..models.metacontroller import MetaController
from ..envs.internal_env import InternalRLEnv
from ..envs.gridworld import GridworldPinpad
from ..envs.tasks import GRID_SIZE, NUM_COLORS, NUM_WALLS, MAX_STEPS


def evaluate_policy(
    policy: CausalSSMPolicy,
    base_model: CausalTransformer,
    metacontroller: MetaController,
    task: list[int],
    controlled_layer: int = 3,
    beta_threshold: float = 0.5,
    num_episodes: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate trained RL policy on a task.

    Returns:
        Dictionary with:
            success_rate: fraction of successful episodes
            avg_reward: average total reward
            avg_abstract_steps: average number of abstract steps per episode
            avg_raw_steps: average raw environment steps
    """
    policy.eval()

    successes = 0
    total_rewards = []
    abstract_steps_list = []
    raw_steps_list = []

    controller_decoder = metacontroller.get_decoder()
    switching_unit = metacontroller.get_switching_unit()

    for ep in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        gridworld = GridworldPinpad(
            grid_size=GRID_SIZE, num_colors=NUM_COLORS,
            num_walls=NUM_WALLS, max_steps=MAX_STEPS, seed=ep + 10000,
        )
        env = InternalRLEnv(
            gridworld=gridworld,
            base_model=base_model,
            controller_decoder=controller_decoder,
            switching_unit=switching_unit,
            controlled_layer=controlled_layer,
            beta_threshold=beta_threshold,
            device=device,
        )

        e, reward, done, state = env.init(task)
        total_reward = 0.0
        abstract_steps = 0
        policy_states = None

        with torch.no_grad():
            while not done:
                _, _, z, policy_states = policy.step(e, policy_states)
                e, reward, done, state = env.step(z, state)
                total_reward += reward
                abstract_steps += 1

        if total_reward > 0:
            successes += 1
        total_rewards.append(total_reward)
        abstract_steps_list.append(abstract_steps)
        raw_steps_list.append(state.raw_steps)

    return {
        "success_rate": successes / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_abstract_steps": np.mean(abstract_steps_list),
        "avg_raw_steps": np.mean(raw_steps_list),
    }


def plot_learning_curves(
    results: dict[str, list[float]],
    save_path: str | Path = "results/learning_curves.png",
    title: str = "Internal RL: Post-training Task Success Rate",
) -> None:
    """Plot learning curves for comparison.

    Args:
        results: Dict mapping method name to list of success rates
                 over training episodes.
        save_path: Path to save figure.
        title: Plot title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Internal RL": "green", "Raw action RL": "blue",
              "w/o temporal abstraction": "orange", "CompILE": "purple"}

    for method, curve in results.items():
        color = colors.get(method, "gray")
        episodes = np.arange(len(curve))
        # Smooth with moving average
        window = max(1, len(curve) // 50)
        smoothed = np.convolve(curve, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=method, color=color, linewidth=2)

    ax.set_xlabel("Number of Episodes")
    ax.set_ylabel("Post-training Success Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curves to {save_path}")
