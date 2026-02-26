"""Real dataset for GROOT N1.6 using LeRobot format from HuggingFace.

Loads LeRobot-format datasets (parquet + video) and converts to
Groot wrapper input format, matching DummyGrootDataset's output schema.

Usage:
    # From local GROOT demo data
    ds = GrootLeRobotDataset(local_path="./Isaac-GR00T/demo_data/cube_to_bowl_5")

    # From HuggingFace
    ds = GrootLeRobotDataset(repo_id="IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from temporal.vla.data.lerobot_utils import (
    compute_global_index_map,
    decode_video_frame,
    download_lerobot_dataset,
    get_action_chunk,
    load_episode_parquet,
    load_lerobot_metadata,
    normalize_minmax,
)

logger = logging.getLogger(__name__)


class GrootLeRobotDataset(Dataset):
    """LeRobot dataset adapted for Groot wrapper input format.

    Loads data from a local LeRobot-format directory or downloads from
    HuggingFace Hub. Produces batches matching GrootWrapper.extract_residual()
    input format.

    Args:
        repo_id: HuggingFace repo ID. Ignored if local_path is provided.
        local_path: Path to a local LeRobot-format dataset directory.
        action_horizon: Number of future action steps per chunk.
        image_size: Target image resolution (H, W).
        max_text_len: Maximum tokenized text length.
        tokenizer_name: HuggingFace tokenizer for text encoding.
        normalize: Whether to apply min-max normalization to state/actions.
    """

    def __init__(
        self,
        repo_id: str = "",
        local_path: str | None = None,
        action_horizon: int = 16,
        image_size: tuple[int, int] = (224, 224),
        max_text_len: int = 64,
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
        normalize: bool = True,
    ):
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.normalize = normalize

        # Resolve dataset path
        if local_path:
            self.dataset_path = Path(local_path)
        elif repo_id:
            self.dataset_path = download_lerobot_dataset(repo_id)
        else:
            raise ValueError("Either repo_id or local_path must be provided")

        # Load metadata
        meta = load_lerobot_metadata(self.dataset_path)
        self.info = meta["info"]
        self.episodes = meta["episodes"]
        self.tasks = meta["tasks"]
        self.stats = meta["stats"]
        self.modality = meta["modality"]

        # Extract dimensions from info
        features = self.info.get("features", {})
        self.state_dim = features.get("observation.state", {}).get("shape", [6])[0]
        self.action_dim = features.get("action", {}).get("shape", [6])[0]

        # Detect video keys from modality.json or features
        self.video_keys = self._detect_video_keys()

        # Build global index map
        self.index_map = compute_global_index_map(self.episodes)

        # Lazy-load tokenizer
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

        # Episode cache (cache last loaded episode)
        self._cached_ep_idx: int | None = None
        self._cached_df = None

        logger.info(
            f"GrootLeRobotDataset: {len(self)} steps from "
            f"{len(self.episodes)} episodes, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"video_keys={self.video_keys}"
        )

    def _detect_video_keys(self) -> list[str]:
        """Detect video/camera keys from modality.json or info features."""
        if self.modality and "video" in self.modality:
            keys = []
            for name, spec in self.modality["video"].items():
                original = spec.get("original_key", f"observation.images.{name}")
                keys.append(original)
            return keys

        # Fallback: scan features for video types
        features = self.info.get("features", {})
        return [k for k, v in features.items() if v.get("dtype") == "video"]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        return self._tokenizer

    def _load_episode(self, ep_idx: int):
        """Load episode DataFrame with caching."""
        if self._cached_ep_idx != ep_idx:
            self._cached_df = load_episode_parquet(
                self.dataset_path, ep_idx, self.info
            )
            self._cached_ep_idx = ep_idx
        return self._cached_df

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, step_idx = self.index_map[idx]
        df = self._load_episode(ep_idx)
        row = df.iloc[step_idx]

        # --- State ---
        state = np.array(row["observation.state"], dtype=np.float32)
        if self.normalize and self.stats and "observation.state" in self.stats:
            s = self.stats["observation.state"]
            state = normalize_minmax(
                state,
                np.array(s["min"], dtype=np.float32),
                np.array(s["max"], dtype=np.float32),
            )

        # --- Actions (chunked) ---
        actions = get_action_chunk(df, step_idx, "action", self.action_horizon)
        if self.normalize and self.stats and "action" in self.stats:
            s = self.stats["action"]
            actions = normalize_minmax(
                actions,
                np.array(s["min"], dtype=np.float32),
                np.array(s["max"], dtype=np.float32),
            )

        # --- Images ---
        images = {}
        for i, video_key in enumerate(self.video_keys):
            cam_name = f"cam_{i}_rgb"
            video_path = self._get_video_path(ep_idx, video_key)
            if video_path.exists():
                images[cam_name] = decode_video_frame(
                    video_path, step_idx, self.image_size
                )
            else:
                images[cam_name] = torch.zeros(3, *self.image_size)

        # --- Text ---
        ep_meta = self.episodes[ep_idx]
        if ep_meta.get("tasks"):
            text = ep_meta["tasks"][0]
        elif "task_index" in row:
            task_idx = int(row["task_index"])
            text = self.tasks.get(task_idx, "manipulation task")
        else:
            text = "manipulation task"

        # --- Tokenize text ---
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "images": images,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "text": text,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0).bool(),
        }

    def _get_video_path(self, ep_idx: int, video_key: str) -> Path:
        """Construct video file path for a given episode and camera key."""
        chunks_size = self.info.get("chunks_size", 1000)
        chunk_id = ep_idx // chunks_size
        video_path_template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        rel_path = video_path_template.format(
            episode_chunk=chunk_id,
            video_key=video_key,
            episode_index=ep_idx,
        )
        return self.dataset_path / rel_path
