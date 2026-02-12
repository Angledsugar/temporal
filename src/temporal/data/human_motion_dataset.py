"""Human manipulation data loaders for Phase 1 pretraining.

Supports:
  - NPZ format (Inter-X converted, Ego4D, UniHand): actions/proprioception/text
  - LeRobot v2.1 parquet format: meta/info.json + data/chunk-*/episode_*.parquet
  - RLDS TFRecord format (DROID droid_100): via dataset_info.json discovery
  - Multi-dataset weighted sampling: combine multiple data sources with
    configurable sampling weights (e.g., Inter-X 40%, Ego4D 40%, UniHand 20%)

Actions are pre-retargeted to canonical EE representation [x,y,z,qx,qy,qz,gripper]
via retargeting.py before being fed to the action expert.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

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
        action_dim: int = 7,
        proprio_dim: int = 14,
        dataset_name: str = "",
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length
        self.action_chunk_size = action_chunk_size
        self.target_fps = target_fps
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.dataset_name = dataset_name or self.data_root.name

        self._mode = "npz"  # "npz", "lerobot", "tfrecord"
        self._tfrecord_action_dim = action_dim
        self._tfrecord_proprio_dim = proprio_dim

        # LeRobot-specific state
        self._lerobot_episodes: list[dict] = []  # [{episode_index, length, task, parquet_path}]

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

        # 2. Try LeRobot v2.1 parquet format (meta/info.json + data/)
        lerobot_episodes = self._discover_lerobot_episodes()
        if lerobot_episodes:
            return lerobot_episodes

        # 3. Try RLDS TFRecord format (e.g. droid_100)
        return self._discover_tfrecord_episodes()

    def _discover_lerobot_episodes(self) -> list[int]:
        """Discover episodes from LeRobot v2.1 dataset (meta/info.json)."""
        info_path = self.data_root / "meta" / "info.json"
        if not info_path.exists():
            return []

        with open(info_path) as f:
            info = json.load(f)

        if info.get("codebase_version", "") not in ("v2.0", "v2.1"):
            return []

        total_episodes = info.get("total_episodes", 0)
        if total_episodes == 0:
            return []

        chunks_size = info.get("chunks_size", 1000)
        data_path_template = info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )

        # Load episode metadata (length + task text)
        episodes_path = self.data_root / "meta" / "episodes.jsonl"
        episode_meta = {}
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = json.loads(line)
                    ep_idx = ep["episode_index"]
                    tasks = ep.get("tasks", [])
                    episode_meta[ep_idx] = {
                        "length": ep.get("length", 0),
                        "task": tasks[0] if tasks else "",
                    }

        # Build episode list with parquet paths
        self._lerobot_episodes = []
        for ep_idx in range(total_episodes):
            chunk_idx = ep_idx // chunks_size
            parquet_path = self.data_root / data_path_template.format(
                episode_chunk=chunk_idx, episode_index=ep_idx
            )
            meta = episode_meta.get(ep_idx, {"length": 0, "task": ""})
            self._lerobot_episodes.append({
                "episode_index": ep_idx,
                "length": meta["length"],
                "task": meta["task"],
                "parquet_path": parquet_path,
            })

        self._mode = "lerobot"

        # Read action/proprio dims from info features
        features = info.get("features", {})
        if "action" in features:
            shape = features["action"].get("shape", [])
            if shape:
                self.action_dim = shape[0]
        if "observation.state" in features:
            shape = features["observation.state"].get("shape", [])
            if shape:
                self.proprio_dim = shape[0]

        logger.info(
            f"LeRobot v2.1 dataset: {total_episodes} episodes "
            f"(action_dim={self.action_dim}, proprio_dim={self.proprio_dim})"
        )
        return list(range(total_episodes))

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
                    self._mode = "tfrecord"

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
        if self._mode == "lerobot":
            return self._get_lerobot_item(idx)
        if self._mode == "tfrecord":
            return self._get_synthetic_item(idx)
        return self._get_npz_item(idx)

    def _get_lerobot_item(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load episode from LeRobot v2.1 parquet file."""
        import pyarrow.parquet as pq

        ep_info = self._lerobot_episodes[idx]
        parquet_path = ep_info["parquet_path"]
        text = ep_info["task"]

        table = pq.read_table(parquet_path, columns=["action", "observation.state"])
        T = len(table)

        # Extract numpy arrays from arrow list columns
        actions = np.stack(table.column("action").to_pylist()).astype(np.float32)
        proprio = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)

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

        # Pad action/proprio dims to match expected dims (datasets may differ)
        if actions.shape[1] < self.action_dim:
            pad = np.zeros((T, self.action_dim - actions.shape[1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]

        if proprio.shape[1] < self.proprio_dim:
            pad = np.zeros((T, self.proprio_dim - proprio.shape[1]), dtype=np.float32)
            proprio = np.concatenate([proprio, pad], axis=1)
        elif proprio.shape[1] > self.proprio_dim:
            proprio = proprio[:, : self.proprio_dim]

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


# ---------------------------------------------------------------------------
# Multi-dataset weighted sampling
# ---------------------------------------------------------------------------


def create_weighted_dataloader(
    dataset_configs: list[dict],
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    max_length: int = 256,
    action_dim: int = 7,
    proprio_dim: int = 14,
) -> DataLoader:
    """Create a DataLoader that samples from multiple datasets with weights.

    Args:
        dataset_configs: List of dicts with keys:
            - path: str, path to dataset root
            - weight: float, sampling weight (will be normalized)
            - name: str, dataset name (optional)
        split: Dataset split.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        max_length: Max sequence length.
        action_dim: Action dimension (datasets padded/truncated to match).
        proprio_dim: Proprioception dimension.

    Returns:
        DataLoader with WeightedRandomSampler.
    """
    datasets = []
    sample_weights = []

    for cfg in dataset_configs:
        ds = HumanMotionDataset(
            data_root=cfg["path"],
            split=split,
            max_length=max_length,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            dataset_name=cfg.get("name", ""),
        )
        if len(ds) == 0:
            logger.warning(f"Dataset {cfg.get('name', cfg['path'])} is empty, skipping")
            continue

        datasets.append(ds)

        # Each sample in this dataset gets weight = cfg_weight / num_samples
        # so total probability mass for this dataset = cfg_weight
        w = cfg.get("weight", 1.0) / len(ds)
        sample_weights.extend([w] * len(ds))

        logger.info(
            f"  {cfg.get('name', ds.dataset_name)}: {len(ds)} sequences, "
            f"weight={cfg.get('weight', 1.0):.2f}"
        )

    if not datasets:
        raise ValueError("No valid datasets found")

    combined = ConcatDataset(datasets)

    # Normalize weights
    total_w = sum(sample_weights)
    sample_weights = [w / total_w for w in sample_weights]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined),
        replacement=True,
    )

    total_seqs = sum(len(d) for d in datasets)
    logger.info(f"Combined dataset: {total_seqs} total sequences, {len(datasets)} sources")

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_dataloader(
    data_root: str | Path,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create a single-dataset DataLoader (backwards compatible)."""
    dataset = HumanMotionDataset(data_root, split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
