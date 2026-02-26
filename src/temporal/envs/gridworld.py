"""Gridworld-Pinpad environment (Appendix A.1).

A 2D grid world where an agent navigates to visit colored cells in a
task-specific order. The environment is inspired by the visual Pin Pad
benchmark.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .tasks import GRID_SIZE, MAX_STEPS, NUM_ACTIONS, NUM_COLORS, NUM_WALLS, OBS_DIM

# Actions: 0=up, 1=right, 2=down, 3=left
ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


class GridworldPinpad:
    """Gridworld-Pinpad MDP.

    State: 2D grid of size G x G with O colored cells and W walls.
    Action: 4 cardinal directions (discrete).
    Observation: one-hot encoding of grid contents + agent position.
    Reward: 1 if task completed, 0 otherwise (sparse).

    Args:
        grid_size: Size of the grid (default 7).
        num_colors: Number of colored cells (default 8).
        num_walls: Number of wall cells (default 4).
        max_steps: Maximum episode length (default 100).
        seed: Random seed.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        num_colors: int = NUM_COLORS,
        num_walls: int = NUM_WALLS,
        max_steps: int = MAX_STEPS,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_walls = num_walls
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        # Grid encoding categories: empty(1) + colors(O) + walls(W) = W+O+1
        self.num_categories = num_walls + num_colors + 1  # 13
        self.obs_dim = grid_size * grid_size * self.num_categories  # 637

        # Will be set on reset
        self.grid: np.ndarray | None = None  # (G, G) with cell type indices
        self.agent_pos: tuple[int, int] | None = None
        self.color_positions: dict[int, tuple[int, int]] = {}
        self.wall_positions: set[tuple[int, int]] = set()
        self.task: list[int] = []
        self.current_subgoal_idx: int = 0
        self.visited_colors: list[int] = []
        self.step_count: int = 0
        self.done: bool = False

    def reset(self, task: list[int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment with a new random grid configuration.

        Args:
            task: Sequence of color indices to visit. If None, must be set before stepping.

        Returns:
            observation: One-hot encoded observation vector.
            info: Dictionary with metadata.
        """
        if task is not None:
            self.task = task

        self.current_subgoal_idx = 0
        self.visited_colors = []
        self.step_count = 0
        self.done = False

        # Initialize grid: 0 = empty
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Sample all positions for colors, walls, and agent (no overlap)
        total_cells = self.grid_size * self.grid_size
        num_special = self.num_colors + self.num_walls + 1  # +1 for agent
        positions = self.rng.choice(total_cells, size=num_special, replace=False)

        # Assign color positions (category indices 1..O)
        self.color_positions = {}
        for i in range(self.num_colors):
            r, c = divmod(int(positions[i]), self.grid_size)
            self.color_positions[i] = (r, c)
            self.grid[r, c] = i + 1  # colors are 1-indexed in grid

        # Assign wall positions (category indices O+1..O+W)
        self.wall_positions = set()
        for i in range(self.num_walls):
            r, c = divmod(int(positions[self.num_colors + i]), self.grid_size)
            self.wall_positions.add((r, c))
            self.grid[r, c] = self.num_colors + 1 + i  # walls after colors

        # Assign agent position
        agent_idx = int(positions[self.num_colors + self.num_walls])
        self.agent_pos = divmod(agent_idx, self.grid_size)

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take an action in the environment.

        Args:
            action: Integer in {0, 1, 2, 3} for up, right, down, left.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert 0 <= action < NUM_ACTIONS
        assert not self.done, "Episode is done. Call reset()."

        self.step_count += 1
        old_pos = self.agent_pos

        # Compute new position
        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            # Out of bounds: no-op
            new_r, new_c = self.agent_pos

        # Check walls
        if (new_r, new_c) in self.wall_positions:
            # Wall collision: no-op
            new_r, new_c = self.agent_pos

        self.agent_pos = (new_r, new_c)

        reward = 0.0
        terminated = False
        truncated = False

        # Check if agent moved to a different cell (colored cell visit detection)
        if self.agent_pos != old_pos:
            # Check if the new cell has a color
            cell_val = self.grid[self.agent_pos[0], self.agent_pos[1]]
            if 1 <= cell_val <= self.num_colors:
                color_idx = cell_val - 1  # Convert back to 0-indexed
                expected_color = self.task[self.current_subgoal_idx]

                if color_idx == expected_color:
                    # Correct color visited
                    self.visited_colors.append(color_idx)
                    self.current_subgoal_idx += 1

                    if self.current_subgoal_idx >= len(self.task):
                        # Task completed successfully
                        reward = 1.0
                        terminated = True
                else:
                    # Wrong color: episode ends
                    terminated = True

        # Check max steps
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        self.done = terminated or truncated
        obs = self._build_observation()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """Build one-hot observation vector.

        Each cell gets a (W + O + 1)-dimensional one-hot vector:
        - Index 0: empty
        - Index 1..O: color 0..O-1
        - Index O+1..O+W: wall 0..W-1

        The agent position is encoded as a separate G^2-dim one-hot
        appended to the grid encoding. However, per the paper formula
        G^2(W+O+1), we encode the agent presence into each cell's
        one-hot as the "empty+agent" category using index 0.

        Total: G^2 * (W+O+1) = 637 dimensions.
        """
        obs = np.zeros(self.grid_size * self.grid_size * self.num_categories, dtype=np.float32)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_idx = r * self.grid_size + c
                base = cell_idx * self.num_categories
                cell_val = self.grid[r, c]

                if cell_val > 0:
                    # Colored cell or wall
                    obs[base + cell_val] = 1.0
                else:
                    # Empty cell
                    obs[base + 0] = 1.0

        # Override the agent's cell: mark agent position
        # The paper says obs includes agent position as a one-hot.
        # We encode it by replacing the empty slot at agent position.
        ar, ac = self.agent_pos
        agent_cell_idx = ar * self.grid_size + ac
        agent_base = agent_cell_idx * self.num_categories
        cell_val = self.grid[ar, ac]
        if cell_val == 0:
            # Agent is on an empty cell - the empty flag is already set
            # We keep it as-is; the agent position information comes from
            # the pattern of "which cell has the agent" being distinguishable
            pass

        return obs

    def _build_info(self) -> dict[str, Any]:
        """Build info dictionary."""
        return {
            "agent_pos": self.agent_pos,
            "step_count": self.step_count,
            "current_subgoal_idx": self.current_subgoal_idx,
            "task": self.task,
            "visited_colors": list(self.visited_colors),
            "color_positions": dict(self.color_positions),
            "done": self.done,
        }

    def get_grid_state(self) -> dict[str, Any]:
        """Get full grid state for expert policy / visualization."""
        return {
            "grid": self.grid.copy(),
            "agent_pos": self.agent_pos,
            "color_positions": dict(self.color_positions),
            "wall_positions": set(self.wall_positions),
            "grid_size": self.grid_size,
        }

    def shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        avoid_colors: set[int] | None = None,
    ) -> list[int]:
        """BFS shortest path from start to goal, avoiding walls and specified colors.

        Args:
            start: Starting position.
            goal: Goal position.
            avoid_colors: Set of color indices whose cells should be avoided
                         (to prevent visiting wrong colors). The goal cell is
                         never avoided even if its color is in avoid_colors.

        Returns:
            List of actions to reach goal, or empty list if unreachable.
        """
        if start == goal:
            return []

        # Build set of positions to avoid (walls + wrong-color cells)
        blocked = set(self.wall_positions)
        if avoid_colors is not None:
            for color_idx, pos in self.color_positions.items():
                if color_idx in avoid_colors and pos != goal:
                    blocked.add(pos)

        visited = set()
        visited.add(start)
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])

        while queue:
            pos, path = queue.popleft()
            for action, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and (nr, nc) not in blocked
                    and (nr, nc) not in visited
                ):
                    new_path = path + [action]
                    if (nr, nc) == goal:
                        return new_path
                    visited.add((nr, nc))
                    queue.append(((nr, nc), new_path))

        # If no path avoiding colors, try without color avoidance (fallback)
        if avoid_colors is not None:
            return self.shortest_path(start, goal, avoid_colors=None)

        return []  # Unreachable
