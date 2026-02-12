"""Inter-X dataset converter for TempoRAL training pipeline.

Converts Inter-X human-human interaction data into:
  1. NPZ format (actions/proprioception/text) for direct pipeline use
  2. LeRobot v2.1 format (parquet + metadata) for Hub sharing

Inter-X source data:
  - skeletons-002/skeletons/{ID}/P1.npy, P2.npy  →  (T, 64, 3) at 120fps
  - motions-003/motions/{ID}/P1.npz, P2.npz      →  SMPL-X parameters at 120fps
  - texts/texts/{ID}.txt                          →  3 text descriptions per sequence
  - annots/                                       →  action labels, interaction order
  - splits/                                       →  train/val/test splits
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from temporal.data.retargeting import interx_skeleton_to_ee, interx_skeleton_to_proprio

logger = logging.getLogger(__name__)


class InterXConverter:
    """Converts Inter-X dataset to TempoRAL-compatible formats."""

    def __init__(
        self,
        interx_root: str | Path,
        target_fps: int = 30,
        hand_mode: str = "actor",
        gripper_threshold: float = 0.04,
    ):
        """
        Args:
            interx_root: Path to Inter-X dataset root.
            target_fps: Output FPS (source is 120fps).
            hand_mode: How to handle two-person data.
                "actor" - only the actor (from interaction_order).
                "both"  - both P1 and P2 as separate episodes.
                "p1"    - always P1 only.
            gripper_threshold: Threshold for gripper closed detection (meters).
        """
        self.root = Path(interx_root)
        self.target_fps = target_fps
        self.hand_mode = hand_mode
        self.gripper_threshold = gripper_threshold
        self.source_fps = 120
        self.downsample_ratio = self.source_fps // self.target_fps

        # Locate data directories
        self.skeleton_dir = self._find_dir("skeletons-002/skeletons", "skeletons")
        self.motion_dir = self._find_dir("motions-003/motions", "motions")
        self.text_dir = self._find_dir("texts/texts", "texts")

        # Load metadata
        self.action_labels = self._load_action_labels()
        self.interaction_order = self._load_interaction_order()
        self.splits = self._load_splits()

    def _find_dir(self, primary: str, fallback: str) -> Path:
        """Find data directory, trying primary path then fallback."""
        path = self.root / primary
        if path.exists():
            return path
        path = self.root / fallback
        if path.exists():
            return path
        raise FileNotFoundError(f"Cannot find {primary} or {fallback} in {self.root}")

    def _load_action_labels(self) -> list[str]:
        path = self.root / "annots" / "action_setting.txt"
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _load_interaction_order(self) -> dict[str, int]:
        path = self.root / "annots" / "interaction_order.pkl"
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    def _load_splits(self) -> dict[str, list[str]]:
        splits = {}
        for split_name in ["train", "val", "test", "all"]:
            path = self.root / "splits" / f"{split_name}.txt"
            if path.exists():
                splits[split_name] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return splits

    def _parse_action_id(self, seq_id: str) -> int:
        """Extract action ID from sequence name like G001T000A003R016."""
        parts = seq_id.split("A")
        if len(parts) >= 2:
            return int(parts[1].split("R")[0])
        return -1

    def get_action_label(self, seq_id: str) -> str:
        aid = self._parse_action_id(seq_id)
        if 0 <= aid < len(self.action_labels):
            return self.action_labels[aid]
        return ""

    def load_skeleton(self, seq_id: str, person: str = "P1") -> np.ndarray | None:
        """Load skeleton data (T, 64, 3) at source fps."""
        path = self.skeleton_dir / seq_id / f"{person}.npy"
        if not path.exists():
            return None
        return np.load(path)  # (T, 64, 3), float64

    def load_text(self, seq_id: str) -> list[str]:
        """Load text descriptions (3 per sequence)."""
        path = self.text_dir / f"{seq_id}.txt"
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _actor_person(self, seq_id: str) -> str:
        """Return 'P1' or 'P2' depending on who is the actor."""
        order = self.interaction_order.get(seq_id, 0)
        return "P1" if order == 0 else "P2"

    def _reactor_person(self, seq_id: str) -> str:
        order = self.interaction_order.get(seq_id, 0)
        return "P2" if order == 0 else "P1"

    def convert_sequence(
        self,
        seq_id: str,
        person: str = "P1",
        side: str = "right",
    ) -> dict | None:
        """Convert a single sequence+person to canonical format.

        Returns:
            dict with keys: actions(T,7), proprioception(T,14), text(str),
                  action_label(str), seq_id(str), person(str)
            or None if data is missing/too short.
        """
        skeleton = self.load_skeleton(seq_id, person)
        if skeleton is None:
            return None

        # Downsample 120fps → target_fps
        skeleton = skeleton[:: self.downsample_ratio].astype(np.float32)
        T = skeleton.shape[0]

        if T < 3:
            logger.warning(f"Skipping {seq_id}/{person}: too short ({T} frames)")
            return None

        # Retarget to canonical EE
        actions = interx_skeleton_to_ee(
            skeleton, side=side, gripper_threshold=self.gripper_threshold
        )
        proprio = interx_skeleton_to_proprio(
            skeleton, side=side
        )

        # Text: use first description
        texts = self.load_text(seq_id)
        text = texts[0] if texts else self.get_action_label(seq_id)

        return {
            "actions": actions,
            "proprioception": proprio,
            "text": text,
            "action_label": self.get_action_label(seq_id),
            "seq_id": seq_id,
            "person": person,
        }

    def _get_episodes_for_sequence(self, seq_id: str) -> list[tuple[str, str]]:
        """Return list of (person, side) pairs to convert for a sequence."""
        if self.hand_mode == "both":
            return [("P1", "right"), ("P1", "left"), ("P2", "right"), ("P2", "left")]
        elif self.hand_mode == "p1":
            return [("P1", "right")]
        else:  # actor mode
            actor = self._actor_person(seq_id)
            return [(actor, "right")]

    # ------------------------------------------------------------------
    # NPZ conversion
    # ------------------------------------------------------------------

    def convert_to_npz(
        self,
        output_dir: str | Path,
        split: str | None = None,
        max_sequences: int | None = None,
    ) -> dict[str, int]:
        """Convert Inter-X to NPZ files compatible with HumanMotionDataset.

        Output structure:
            output_dir/
            ├── train.txt, val.txt, test.txt
            └── trajectories/
                ├── {seq_id}_{person}_{side}.npz
                └── ...

        Returns:
            Dict with counts per split.
        """
        output_dir = Path(output_dir)
        traj_dir = output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        counts = {}

        for split_name, seq_ids in self.splits.items():
            if split_name == "all":
                continue
            if split is not None and split_name != split:
                continue

            names = []
            ids = seq_ids[:max_sequences] if max_sequences else seq_ids

            for seq_id in tqdm(ids, desc=f"NPZ {split_name}"):
                for person, side in self._get_episodes_for_sequence(seq_id):
                    result = self.convert_sequence(seq_id, person, side)
                    if result is None:
                        continue

                    fname = f"{seq_id}_{person}_{side}.npz"
                    np.savez_compressed(
                        traj_dir / fname,
                        actions=result["actions"],
                        proprioception=result["proprioception"],
                        text=result["text"],
                    )
                    names.append(fname)

            # Write split file
            split_file = output_dir / f"{split_name}.txt"
            split_file.write_text("\n".join(names) + "\n")
            counts[split_name] = len(names)
            logger.info(f"{split_name}: {len(names)} episodes")

        return counts

    # ------------------------------------------------------------------
    # LeRobot v2.1 conversion
    # ------------------------------------------------------------------

    def convert_to_lerobot(
        self,
        output_dir: str | Path,
        repo_id: str = "temporal/interx",
        split: str | None = None,
        max_sequences: int | None = None,
    ) -> None:
        """Convert Inter-X to LeRobot v2.1 dataset format.

        Uses LeRobotDataset.create() API to build a proper dataset with
        parquet files, metadata, and statistics.
        """
        import shutil

        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        output_dir = Path(output_dir)
        # LeRobot requires the directory to NOT exist (it creates it)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "axes": [
                        "shoulder_elevation",
                        "shoulder_azimuth",
                        "shoulder_twist",
                        "elbow_flexion",
                        "wrist_flexion",
                        "wrist_deviation",
                        "wrist_rotation",
                        "wrist_x",
                        "wrist_y",
                        "wrist_z",
                        "wrist_roll",
                        "wrist_pitch",
                        "wrist_yaw",
                        "gripper_opening",
                    ]
                },
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "axes": ["x", "y", "z", "qx", "qy", "qz", "gripper"]
                },
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self.target_fps,
            root=output_dir,
            robot_type="human_interx",
            features=features,
            use_videos=False,
        )

        all_seq_ids = []
        for split_name, seq_ids in self.splits.items():
            if split_name == "all":
                continue
            if split is not None and split_name != split:
                continue
            ids = seq_ids[:max_sequences] if max_sequences else seq_ids
            all_seq_ids.extend(ids)

        for seq_id in tqdm(all_seq_ids, desc="LeRobot"):
            for person, side in self._get_episodes_for_sequence(seq_id):
                result = self.convert_sequence(seq_id, person, side)
                if result is None:
                    continue

                T = result["actions"].shape[0]
                for t in range(T):
                    frame = {
                        "observation.state": result["proprioception"][t],
                        "action": result["actions"][t],
                        "task": result["text"],
                    }
                    dataset.add_frame(frame)

                dataset.save_episode()

        logger.info(
            f"LeRobot dataset saved to {output_dir}: "
            f"{dataset.num_episodes} episodes, {dataset.num_frames} frames"
        )

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    def compute_stats(
        self,
        seq_ids: list[str] | None = None,
        max_sequences: int = 100,
    ) -> dict:
        """Compute basic stats over a subset of sequences for diagnostics."""
        if seq_ids is None:
            seq_ids = self.splits.get("train", self.splits.get("all", []))
        seq_ids = seq_ids[:max_sequences]

        all_actions = []
        all_proprio = []
        lengths = []

        for seq_id in tqdm(seq_ids, desc="Stats"):
            for person, side in self._get_episodes_for_sequence(seq_id):
                result = self.convert_sequence(seq_id, person, side)
                if result is None:
                    continue
                all_actions.append(result["actions"])
                all_proprio.append(result["proprioception"])
                lengths.append(result["actions"].shape[0])

        if not all_actions:
            return {}

        actions = np.concatenate(all_actions, axis=0)
        proprio = np.concatenate(all_proprio, axis=0)
        lengths_arr = np.array(lengths)

        return {
            "num_episodes": len(lengths),
            "total_frames": int(actions.shape[0]),
            "sequence_length": {
                "min": int(lengths_arr.min()),
                "max": int(lengths_arr.max()),
                "mean": float(lengths_arr.mean()),
                "median": float(np.median(lengths_arr)),
            },
            "actions": {
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
            },
            "proprioception": {
                "mean": proprio.mean(axis=0).tolist(),
                "std": proprio.std(axis=0).tolist(),
                "min": proprio.min(axis=0).tolist(),
                "max": proprio.max(axis=0).tolist(),
            },
        }
