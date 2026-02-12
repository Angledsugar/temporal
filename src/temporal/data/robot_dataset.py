"""Robot demonstration dataset loader for Phase 3 evaluation.

Used for fine-tuning adaptation and evaluation in simulation
(SIMPLER / ManiSkill3). Not used for Phase 1 pretraining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RobotDemoDataset(Dataset):
    """Robot demonstration dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        max_length: int = 256,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length
        self.sequences = self._load_split()

    def _load_split(self) -> list[Path]:
        traj_dir = self.data_root / self.split
        if traj_dir.exists():
            return sorted(traj_dir.glob("*.npz"))
        return []

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        data = np.load(self.sequences[idx], allow_pickle=True)
        actions = torch.from_numpy(data["actions"].astype(np.float32))
        proprio = torch.from_numpy(data["proprioception"].astype(np.float32))
        text = str(data.get("text", ""))

        T = len(actions)
        if T > self.max_length:
            actions = actions[: self.max_length]
            proprio = proprio[: self.max_length]
            T = self.max_length

        return {
            "actions": actions,
            "proprioception": proprio,
            "text": text,
            "length": T,
        }


def create_robot_dataloader(
    data_root: str | Path,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    dataset = RobotDemoDataset(data_root, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
