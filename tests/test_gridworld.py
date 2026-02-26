"""Tests for Gridworld-Pinpad environment."""

import numpy as np
import pytest

from temporal.envs.gridworld import GridworldPinpad
from temporal.envs.expert import ExpertPolicy
from temporal.envs.tasks import (
    GRID_SIZE, NUM_COLORS, NUM_WALLS, MAX_STEPS, OBS_DIM,
    PRETRAINING_TASKS, POST_TRAINING_TASK,
)


class TestGridworld:
    def test_reset(self):
        env = GridworldPinpad(seed=42)
        obs, info = env.reset(task=[0, 1, 2])
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert info["current_subgoal_idx"] == 0

    def test_obs_dim(self):
        expected = GRID_SIZE * GRID_SIZE * (NUM_WALLS + NUM_COLORS + 1)
        assert OBS_DIM == expected == 637

    def test_step_valid(self):
        env = GridworldPinpad(seed=42)
        env.reset(task=[0, 1])
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_wall_collision(self):
        """Agent should not move into walls."""
        env = GridworldPinpad(seed=42)
        env.reset(task=[0])
        # Try many random actions - should never end up on a wall
        for _ in range(50):
            env.step(np.random.randint(0, 4))
            if env.done:
                break
            assert env.agent_pos not in env.wall_positions

    def test_episode_termination_max_steps(self):
        env = GridworldPinpad(max_steps=10, seed=42)
        env.reset(task=[0, 1, 2, 3, 4, 5, 6, 7])  # Very long task
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps <= 10

    def test_shortest_path(self):
        env = GridworldPinpad(seed=42)
        env.reset(task=[0])
        # Path from agent to first color should exist
        target = env.color_positions[0]
        path = env.shortest_path(env.agent_pos, target)
        assert len(path) > 0 or env.agent_pos == target

    def test_no_overlap_positions(self):
        """Colors, walls, and agent should not overlap."""
        env = GridworldPinpad(seed=42)
        env.reset(task=[0])
        all_positions = set()
        for pos in env.color_positions.values():
            assert pos not in all_positions
            all_positions.add(pos)
        for pos in env.wall_positions:
            assert pos not in all_positions
            all_positions.add(pos)
        assert env.agent_pos not in all_positions


class TestExpert:
    def test_expert_generates_trajectory(self):
        env = GridworldPinpad(seed=42)
        expert = ExpertPolicy(epsilon=0.0, seed=42)
        traj = expert.generate_trajectory(env, task=[0, 1])
        assert traj is not None
        assert traj["observations"].shape[0] == traj["length"] + 1
        assert traj["actions"].shape[0] == traj["length"]
        assert traj["subgoals"].shape[0] == traj["length"]

    def test_expert_success_rate(self):
        """Expert with epsilon=0 should have high success rate."""
        successes = 0
        n_trials = 50
        for i in range(n_trials):
            env = GridworldPinpad(seed=i)
            expert = ExpertPolicy(epsilon=0.0, seed=i + 1000)
            traj = expert.generate_trajectory(env, task=[0, 1])
            if traj and traj["success"]:
                successes += 1
        success_rate = successes / n_trials
        # Should be very high for simple 2-color task
        assert success_rate > 0.8, f"Expected >80% success, got {success_rate:.1%}"

    def test_pretraining_tasks_valid(self):
        """All pretraining tasks should have valid color indices."""
        for task in PRETRAINING_TASKS:
            for color in task:
                assert 0 <= color < NUM_COLORS

    def test_post_training_task_valid(self):
        for color in POST_TRAINING_TASK:
            assert 0 <= color < NUM_COLORS
        assert len(POST_TRAINING_TASK) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
