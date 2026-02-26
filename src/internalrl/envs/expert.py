"""Expert policy for Gridworld-Pinpad via BFS + epsilon-noise.

Generates trajectories by computing shortest paths to each subgoal
in sequence, with optional random action noise.
"""

from __future__ import annotations

import numpy as np

from .gridworld import GridworldPinpad


class ExpertPolicy:
    """BFS-based expert policy with epsilon-noise.

    At each timestep, computes the optimal action via BFS shortest path
    to the current subgoal. With probability epsilon, replaces the action
    with a random non-terminating action.

    Args:
        epsilon: Probability of taking a random action (default 0.0).
        seed: Random seed.
    """

    def __init__(self, epsilon: float = 0.0, seed: int | None = None):
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)

    def get_action(self, env: GridworldPinpad) -> int:
        """Get expert action for the current state.

        Args:
            env: The gridworld environment (accesses internal state).

        Returns:
            Optimal (or epsilon-noisy) action.
        """
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.randint(0, 4)

        # Current subgoal color
        if env.current_subgoal_idx >= len(env.task):
            return self.rng.randint(0, 4)

        target_color = env.task[env.current_subgoal_idx]
        target_pos = env.color_positions[target_color]

        # Avoid other colored cells (stepping on wrong color ends the episode)
        avoid = set(range(env.num_colors)) - {target_color}
        path = env.shortest_path(env.agent_pos, target_pos, avoid_colors=avoid)

        if len(path) == 0:
            # Already at target or unreachable
            return self.rng.randint(0, 4)

        return path[0]

    def generate_trajectory(
        self, env: GridworldPinpad, task: list[int]
    ) -> dict[str, np.ndarray] | None:
        """Generate a single expert trajectory.

        Args:
            env: The gridworld environment.
            task: Task to solve (sequence of color indices).

        Returns:
            Dictionary with:
                - observations: (T+1, obs_dim) float32
                - actions: (T,) int64
                - subgoals: (T,) int64 - ground truth subgoal index at each step
                - rewards: (T,) float32
                - success: bool
                - length: int (actual trajectory length)
            or None if trajectory generation fails.
        """
        obs, _ = env.reset(task=task)

        observations = [obs]
        actions = []
        subgoals = []
        rewards = []

        for _ in range(env.max_steps):
            subgoals.append(env.current_subgoal_idx)
            action = self.get_action(env)
            obs, reward, terminated, truncated, _ = env.step(action)

            actions.append(action)
            observations.append(obs)
            rewards.append(reward)

            if terminated or truncated:
                break

        length = len(actions)
        success = reward > 0

        return {
            "observations": np.array(observations, dtype=np.float32),  # (L+1, obs_dim)
            "actions": np.array(actions, dtype=np.int64),              # (L,)
            "subgoals": np.array(subgoals, dtype=np.int64),            # (L,)
            "rewards": np.array(rewards, dtype=np.float32),            # (L,)
            "success": success,
            "length": length,
            "task": task,
        }
