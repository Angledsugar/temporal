"""Internal MDP for Phase 3 (Internal RL).

The key insight: we redefine the MDP so that the state and action
spaces live INSIDE the action expert's representations.

    State  = e_{t,l}  (residual stream at layer l)
    Action = z_t      (controller code, dim = n_z)
    Reward = 1 if subtask success, 0 otherwise

The "environment dynamics" are the composition of:
    (i)   decoder producing U_t from z_t
    (ii)  frozen action expert generating motor commands
    (iii) physical simulator executing those commands

From the RL agent's perspective, all of this is a black box.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class InternalEnv(gym.Env):
    """Internal MDP for Phase 3.

    The action expert + physical world are subsumed into the environment.
    The RL agent only sees residual stream states and emits controller codes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        action_expert: Any,
        metacontroller: Any,
        sim_env: gym.Env,
        beta_threshold: float = 0.5,
        max_primitive_steps_per_z: int = 50,
    ):
        super().__init__()
        self.expert = action_expert
        self.meta = metacontroller
        self.sim = sim_env
        self.beta_threshold = beta_threshold
        self.max_prim = max_primitive_steps_per_z

        n_e = action_expert.width   # 1024
        n_z = metacontroller.n_z    # 32

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_e,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(n_z,), dtype=np.float32
        )

        self.e_t: np.ndarray | None = None

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs, info = self.sim.reset()
        self.e_t = self._get_residual_stream(obs)
        return self.e_t, info

    def step(
        self, z_t: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute z_t: the action expert runs until beta_t fires
        or max primitive steps reached.

        This achieves TEMPORAL CONTRACTION: if a trajectory of T
        primitive steps has M switch points (M << T), the RL policy
        makes only M decisions.

        Args:
            z_t: (n_z,) -- controller code from RL policy.

        Returns:
            e_next: (n_e,)  -- next residual stream state.
            reward: float   -- accumulated reward during this z_t.
            terminated: bool
            truncated: bool
            info: dict with "primitive_steps" count.
        """
        total_reward = 0.0
        prim_steps = 0
        terminated = False
        truncated = False

        for _ in range(self.max_prim):
            # Apply controller to residual stream
            # e'_t = e_t + U_t @ e_t
            e_controlled = self._apply_control(self.e_t, z_t)

            # Action expert decodes to motor command
            action = self._decode_action(e_controlled)

            # Step physics
            obs, reward, terminated, truncated, info = self.sim.step(action)
            total_reward += reward
            prim_steps += 1

            # Update residual stream
            self.e_t = self._get_residual_stream(obs)

            # Check switching signal (beta_t binarised)
            beta_t = self._compute_beta(self.e_t, z_t)
            if beta_t > self.beta_threshold or terminated:
                break

        info["primitive_steps"] = prim_steps
        return self.e_t, total_reward, terminated, truncated, info

    def _get_residual_stream(self, obs: Any) -> np.ndarray:
        """Extract residual stream e_{t,l} from observation."""
        # TODO: Implement with actual action expert
        # return self.expert.extract_residual_stream(obs)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _apply_control(
        self, e_t: np.ndarray, z_t: np.ndarray
    ) -> np.ndarray:
        """Apply MetaController's additive control."""
        # TODO: self.meta.decoder.apply_control(e_t, z_t)
        return e_t

    def _decode_action(self, e_controlled: np.ndarray) -> np.ndarray:
        """Decode controlled residual stream to motor command."""
        # TODO: self.expert.decode_action(e_controlled)
        return np.zeros(self.sim.action_space.shape, dtype=np.float32)

    def _compute_beta(
        self, e_t: np.ndarray, z_t: np.ndarray
    ) -> float:
        """Compute switching probability beta_t."""
        # TODO: self.meta.compute_beta(e_t, z_t)
        return 0.0
