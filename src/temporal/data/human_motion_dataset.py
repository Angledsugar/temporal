"""Human manipulation data loaders for Phase 1 pretraining.

Supports Something-Something V2, Ego4D hand subset, and UniHand.
Actions are retargeted to canonical end-effector representation
via retargeting.py before being fed to the action expert.

Also supports RLDS TFRecord datasets (e.g. droid_100) by reading
episode metadata from dataset_info.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class HumanMotionDataset(Dataset):
    """Generic human manipulation dataset.

    Loads trajectories of (subtask_text, proprioception, action) tuples
    extracted from human demonstration videos. Human hand joint angles
    are pre-retargeted to canonical EE representation.

    When the data directory contains RLDS TFRecord files instead of .npz,
    generates synthetic placeholder sequences using episode metadata from
    dataset_info.json. This allows the training pipeline to run end-to-end
    while the actual RLDS loading is implemented.
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

        self._tfrecord_mode = False
        self._tfrecord_action_dim = 7    # DROID default
        self._tfrecord_proprio_dim = 14  # joint(7) + cartesian(6) + gripper(1)

        self.sequences = self._load_split()

    def _load_split(self) -> list:
        # 1. Try .npz files first (original format)
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                names = [line.strip() for line in f if line.strip()]
            return [self.data_root / "trajectories" / n for n in names]

        traj_dir = self.data_root / "trajectories"
        if traj_dir.exists():
            npz_files = sorted(traj_dir.glob("*.npz"))
            if npz_files:
                return npz_files

        # 2. Try RLDS TFRecord format (e.g. droid_100)
        return self._discover_tfrecord_episodes()

    def _discover_tfrecord_episodes(self) -> list[int]:
        """Discover episodes from RLDS dataset_info.json."""
        info_candidates = [
            self.data_root / "dataset_info.json",
            *sorted(self.data_root.glob("*/dataset_info.json")),
        ]

        for info_path in info_candidates:
            if not info_path.exists():
                continue
            with open(info_path) as f:
                info = json.load(f)

            for split_info in info.get("splits", []):
                if split_info["name"] == self.split:
                    shard_lengths = split_info.get("shardLengths", [])
                    num_episodes = sum(int(x) for x in shard_lengths)
                    self._tfrecord_mode = True

                    # Try to read action dim from features.json
                    features_path = info_path.parent / "features.json"
                    if features_path.exists():
                        self._read_tfrecord_features(features_path)

                    logger.info(
                        f"RLDS TFRecord dataset: {num_episodes} episodes "
                        f"(action_dim={self._tfrecord_action_dim}, "
                        f"proprio_dim={self._tfrecord_proprio_dim})"
                    )
                    logger.warning(
                        "Using synthetic placeholder sequences for TFRecord data. "
                        "Implement RLDS loading for real data."
                    )
                    return list(range(num_episodes))

        return []

    def _read_tfrecord_features(self, features_path: Path) -> None:
        """Extract action/observation dims from RLDS features.json."""
        with open(features_path) as f:
            features = json.load(f)

        try:
            steps = features["featuresDict"]["features"]["steps"]["sequence"]["feature"]
            step_features = steps["featuresDict"]["features"]

            # Action dim
            action_shape = step_features["action"]["tensor"]["shape"].get("dimensions", [])
            if action_shape:
                self._tfrecord_action_dim = int(action_shape[0])

            # Proprio dim from observation fields
            obs = step_features.get("observation", {}).get("featuresDict", {}).get("features", {})
            proprio_dim = 0
            for key in ["joint_position", "cartesian_position", "gripper_position"]:
                if key in obs:
                    shape = obs[key].get("tensor", {}).get("shape", {}).get("dimensions", [])
                    if shape:
                        proprio_dim += int(shape[0])
            if proprio_dim > 0:
                self._tfrecord_proprio_dim = proprio_dim
        except (KeyError, IndexError):
            pass  # keep defaults

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self._tfrecord_mode:
            return self._get_synthetic_item(idx)
        return self._get_npz_item(idx)

    def _get_npz_item(self, idx: int) -> dict[str, torch.Tensor | str]:
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

    def _get_synthetic_item(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Generate synthetic placeholder sequence for TFRecord episodes.

        TODO: Replace with actual RLDS TFRecord loading.
        """
        rng = np.random.RandomState(idx)
        T = rng.randint(self.max_length // 2, self.max_length + 1)
        actions = rng.randn(self.max_length, self._tfrecord_action_dim).astype(np.float32) * 0.1
        proprio = rng.randn(self.max_length, self._tfrecord_proprio_dim).astype(np.float32) * 0.1

        return {
            "actions": torch.from_numpy(actions),
            "proprioception": torch.from_numpy(proprio),
            "text": f"episode_{idx}",
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
