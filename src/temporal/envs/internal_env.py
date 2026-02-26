"""Internal RL environment wrapper (Algorithms 1-2 in paper).

Wraps the base model + metacontroller decoder as an "environment"
for Internal RL. The RL agent sees residual stream activations and
acts in the latent code space z.

Key idea: The base model + metacontroller decoder + gridworld is
treated as a single environment. The RL policy only needs to produce
latent codes z, and the metacontroller decoder converts them to
residual stream interventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gridworld import GridworldPinpad
from ..models.transformer import CausalTransformer
from ..models.metacontroller import ControllerDecoder, SwitchingUnit


@dataclass
class InternalState:
    """Internal state maintained across abstract timesteps."""
    # Base model KV caches
    kv_caches_lower: list[dict] = field(default_factory=list)  # layers 0..l-1
    kv_caches_upper: list[dict] = field(default_factory=list)  # layers l..L-1
    # Switching unit state
    h_switch: torch.Tensor | None = None
    # Previous z for switching
    z_prev: torch.Tensor | None = None
    # Accumulated reward within abstract action
    reward_acc: float = 0.0
    # Current residual activation
    e_current: torch.Tensor | None = None
    # Gridworld step count
    raw_steps: int = 0


class InternalRLEnv:
    """Internal RL environment (Algorithm 1-2).

    The RL policy produces latent codes z_t. This environment:
    1. Decodes z_t into a linear controller U_t via the hypernetwork
    2. Applies U_t to the base model's residual stream
    3. Samples actions from the modified base model
    4. Steps the gridworld environment
    5. Repeats until the switching unit triggers β_t ≈ 1 (switch)

    Args:
        gridworld: The gridworld environment.
        base_model: Frozen base autoregressive model.
        controller_decoder: Trained controller decoder from metacontroller.
        switching_unit: Trained switching unit from metacontroller.
        controlled_layer: Layer at which to apply control.
        beta_threshold: Threshold for binarizing β_t.
        device: Torch device.
    """

    def __init__(
        self,
        gridworld: GridworldPinpad,
        base_model: CausalTransformer,
        controller_decoder: ControllerDecoder,
        switching_unit: SwitchingUnit,
        controlled_layer: int = 3,
        beta_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.gridworld = gridworld
        self.base_model = base_model
        self.controller_decoder = controller_decoder
        self.switching_unit = switching_unit
        self.controlled_layer = controlled_layer
        self.beta_threshold = beta_threshold
        self.device = device

        # Freeze all components
        for model in [base_model, controller_decoder, switching_unit]:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

    def init(
        self, task: list[int]
    ) -> tuple[torch.Tensor, float, bool, InternalState]:
        """Initialize episode (Algorithm 2).

        Args:
            task: Task sequence of color indices.

        Returns:
            e: (1, embed_dim) initial residual activation.
            reward: Initial reward (0).
            done: Whether episode is done.
            state: Initial internal state.
        """
        obs, _ = self.gridworld.reset(task=task)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # (1, 1, obs_dim)

        # Forward through base model up to controlled layer
        with torch.no_grad():
            e, kv_lower = self.base_model.forward_up_to_layer(
                obs_tensor, self.controlled_layer
            )

        state = InternalState(
            kv_caches_lower=kv_lower,
            kv_caches_upper=[],
            e_current=e.squeeze(1),  # (1, embed_dim)
            z_prev=torch.zeros(1, self.controller_decoder.net[0].in_features, device=self.device),
        )

        return state.e_current, 0.0, False, state

    @torch.no_grad()
    def step(
        self,
        z: torch.Tensor,
        state: InternalState,
    ) -> tuple[torch.Tensor, float, bool, InternalState]:
        """Execute one abstract action (Algorithm 1).

        The latent code z is held fixed until the switching unit triggers.
        Multiple raw actions may be taken in the gridworld.

        Args:
            z: (1, latent_dim) latent controller code.
            state: Current internal state.

        Returns:
            e: (1, embed_dim) new residual activation (observation for RL).
            reward: Accumulated reward during this abstract action.
            done: Whether episode ended.
            new_state: Updated internal state.
        """
        reward_acc = 0.0
        e = state.e_current
        done = False

        # Keep applying the same controller until β triggers
        while True:
            # 1. Apply controller: e' = e + U @ e
            e_controlled = self.controller_decoder.apply_controller(
                e.unsqueeze(1), z.unsqueeze(1)
            ).squeeze(1)  # (1, embed_dim)

            # 2. Forward through remaining base model layers to get action
            result, _ = self.base_model.forward_from_layer(
                e_controlled.unsqueeze(1), self.controlled_layer
            )
            action_logits = result["action_logits"].squeeze(1)  # (1, num_actions)

            # 3. Sample action from base model
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()

            # 4. Step gridworld
            obs, reward, terminated, truncated, _ = self.gridworld.step(action)
            reward_acc += reward
            state.raw_steps += 1

            if terminated or truncated:
                done = True
                break

            # 5. Process new observation through lower layers
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)

            e_new, _ = self.base_model.forward_up_to_layer(
                obs_tensor, self.controlled_layer
            )
            e = e_new.squeeze(1)  # (1, embed_dim)

            # 6. Check switching unit
            # For simplicity, use a fixed GRU hidden of zeros (could maintain state)
            h_dummy = torch.zeros(1, self.switching_unit.net[0].in_features - e.shape[-1] - z.shape[-1],
                                  device=self.device)
            # Actually, switching unit takes (e, h, z_prev)
            # We need h from the GRU — for simplicity, use zero h
            gru_dim = 32  # from config
            h_switch = torch.zeros(1, gru_dim, device=self.device)
            beta = self.switching_unit.step(e, h_switch, z)  # (1, 1)

            # Binarize β with threshold (Heaviside step function)
            if beta.item() >= self.beta_threshold:
                break

            # Safety check: don't loop forever
            if state.raw_steps >= self.gridworld.max_steps:
                done = True
                break

        # Update state
        state.e_current = e
        state.z_prev = z
        state.reward_acc = reward_acc

        return e, reward_acc, done, state
