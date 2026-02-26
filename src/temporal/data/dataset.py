"""Trajectory dataset for autoregressive sequence model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset of observation-action trajectories.

    Each item is a fixed-length trajectory padded/truncated to max_length.

    Args:
        data_path: Path to the .npz file containing trajectory data.
        max_length: Maximum sequence length (default 100).
    """

    def __init__(self, data_path: str | Path, max_length: int = 100):
        data = np.load(data_path, allow_pickle=True)
        self.observations = data["observations"]  # list of (L+1, obs_dim) arrays
        self.actions = data["actions"]             # list of (L,) arrays
        self.subgoals = data["subgoals"]           # list of (L,) arrays
        self.lengths = data["lengths"]             # (N,) actual lengths
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        length = int(self.lengths[idx])
        obs = self.observations[idx]    # (L+1, obs_dim)
        act = self.actions[idx]         # (L,)
        sg = self.subgoals[idx]         # (L,)

        # Pad to max_length
        obs_dim = obs.shape[1]
        T = self.max_length

        padded_obs = np.zeros((T + 1, obs_dim), dtype=np.float32)
        padded_act = np.zeros(T, dtype=np.int64)
        padded_sg = np.zeros(T, dtype=np.int64)
        mask = np.zeros(T, dtype=np.float32)

        actual_len = min(length, T)
        padded_obs[:actual_len + 1] = obs[:actual_len + 1]
        padded_act[:actual_len] = act[:actual_len]
        padded_sg[:actual_len] = sg[:actual_len]
        mask[:actual_len] = 1.0

        return {
            "observations": torch.from_numpy(padded_obs),    # (T+1, obs_dim)
            "actions": torch.from_numpy(padded_act),          # (T,)
            "subgoals": torch.from_numpy(padded_sg),          # (T,)
            "mask": torch.from_numpy(mask),                    # (T,)
            "length": torch.tensor(actual_len, dtype=torch.long),
        }
