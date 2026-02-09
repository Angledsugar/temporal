"""Human manipulation data loaders for Phase 1 pretraining.

Supports Something-Something V2, Ego4D hand subset, and UniHand.
Actions are retargeted to canonical end-effector representation
via retargeting.py before being fed to the action expert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HumanMotionDataset(Dataset):
    """Generic human manipulation dataset.

    Loads trajectories of (subtask_text, proprioception, action) tuples
    extracted from human demonstration videos. Human hand joint angles
    are pre-retargeted to canonical EE representation.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        max_length: int = 256,
        action_chunk_size: int = 50,
        target_fps: int = 50,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length
        self.action_chunk_size = action_chunk_size
        self.target_fps = target_fps

        self.sequences = self._load_split()

    def _load_split(self) -> list[Path]:
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                names = [line.strip() for line in f if line.strip()]
            return [self.data_root / "trajectories" / n for n in names]
        # Fallback: glob all .npz files
        traj_dir = self.data_root / "trajectories"
        if traj_dir.exists():
            return sorted(traj_dir.glob("*.npz"))
        return []

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.sequences[idx]

        # Expected .npz format:
        #   "actions":        (T, action_dim)  -- canonical EE actions
        #   "proprioception": (T, proprio_dim) -- proprioceptive state
        #   "text":           str              -- subtask description
        data = np.load(path, allow_pickle=True)
        actions = data["actions"].astype(np.float32)
        proprio = data["proprioception"].astype(np.float32)
        text = str(data.get("text", ""))

        T = len(actions)

        # Truncate or pad to max_length
        if T > self.max_length:
            start = np.random.randint(0, T - self.max_length)
            actions = actions[start : start + self.max_length]
            proprio = proprio[start : start + self.max_length]
            T = self.max_length
        elif T < self.max_length:
            pad_len = self.max_length - T
            actions = np.concatenate(
                [actions, np.zeros((pad_len, actions.shape[1]), dtype=np.float32)]
            )
            proprio = np.concatenate(
                [proprio, np.zeros((pad_len, proprio.shape[1]), dtype=np.float32)]
            )

        return {
            "actions": torch.from_numpy(actions),
            "proprioception": torch.from_numpy(proprio),
            "text": text,
            "length": T,
        }


def create_dataloader(
    data_root: str | Path,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    dataset = HumanMotionDataset(data_root, split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
