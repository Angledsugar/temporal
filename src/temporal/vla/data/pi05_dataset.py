"""Real dataset for π0.5 using LeRobot format from HuggingFace.

Loads LeRobot-format datasets (parquet + images/video) and converts to
π0.5 wrapper input format, matching DummyPi05Dataset's output schema.

Usage:
    # From HuggingFace (LIBERO example)
    ds = Pi05LeRobotDataset(repo_id="lerobot/libero_object_no_noops")

    # From local path
    ds = Pi05LeRobotDataset(local_path="./data/my_lerobot_dataset")
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
    normalize_zscore,
)

logger = logging.getLogger(__name__)


class Pi05LeRobotDataset(Dataset):
    """LeRobot dataset adapted for π0.5 wrapper input format.

    Loads data from a local LeRobot-format directory or downloads from
    HuggingFace Hub. Produces batches matching Pi05Wrapper.extract_residual()
    input format.

    Args:
        repo_id: HuggingFace repo ID. Ignored if local_path is provided.
        local_path: Path to a local LeRobot-format dataset directory.
        action_horizon: Number of future action steps per chunk.
        image_size: Target image resolution (H, W).
        max_token_len: Maximum tokenized prompt length.
        pad_state_dim: Pad state to this dimension (0 = no padding).
        pad_action_dim: Pad actions to this dimension (0 = no padding).
        tokenizer_name: HuggingFace tokenizer for prompt encoding.
        normalize: Whether to apply z-score normalization to state/actions.
        image_key_map: Mapping from dataset image columns to model camera names.
            Default auto-detects.
    """

    def __init__(
        self,
        repo_id: str = "",
        local_path: str | None = None,
        action_horizon: int = 50,
        image_size: tuple[int, int] = (224, 224),
        max_token_len: int = 48,
        pad_state_dim: int = 0,
        pad_action_dim: int = 0,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        normalize: bool = True,
        image_key_map: dict[str, str] | None = None,
    ):
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.max_token_len = max_token_len
        self.pad_state_dim = pad_state_dim
        self.pad_action_dim = pad_action_dim
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

        # Extract dimensions from info
        features = self.info.get("features", {})
        self.state_dim = self._detect_state_dim(features)
        self.action_dim = self._detect_action_dim(features)

        # Detect image/video columns and build key map
        self.image_key_map = image_key_map or self._auto_detect_image_keys(features)

        # Detect state/action column names
        self.state_col = self._detect_column(features, ["observation.state", "observation/state", "state"])
        self.action_col = self._detect_column(features, ["actions", "action"])
        self.prompt_col = self._detect_column(features, ["task", "prompt", "language_instruction"], required=False)

        # Build global index map
        self.index_map = compute_global_index_map(self.episodes)

        # Lazy-load tokenizer
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

        # Episode cache
        self._cached_ep_idx: int | None = None
        self._cached_df = None

        logger.info(
            f"Pi05LeRobotDataset: {len(self)} steps from "
            f"{len(self.episodes)} episodes, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"images={list(self.image_key_map.keys())}"
        )

    @staticmethod
    def _detect_state_dim(features: dict) -> int:
        for key in ["observation.state", "observation/state", "state"]:
            if key in features:
                shape = features[key].get("shape", [])
                if shape:
                    return shape[0] if isinstance(shape[0], int) else shape[-1]
        return 8  # LIBERO default

    @staticmethod
    def _detect_action_dim(features: dict) -> int:
        for key in ["actions", "action"]:
            if key in features:
                shape = features[key].get("shape", [])
                if shape:
                    return shape[0] if isinstance(shape[0], int) else shape[-1]
        return 7  # LIBERO default

    @staticmethod
    def _detect_column(features: dict, candidates: list[str], required: bool = True) -> str | None:
        for c in candidates:
            if c in features:
                return c
        if required:
            raise KeyError(f"None of {candidates} found in dataset features: {list(features.keys())}")
        return None

    def _auto_detect_image_keys(self, features: dict) -> dict[str, str]:
        """Auto-detect image/video columns and map to model camera names.

        Returns:
            Dict mapping dataset column name -> model camera name.
        """
        image_features = {}
        for key, spec in features.items():
            dtype = spec.get("dtype", "")
            if dtype in ("image", "video") or "image" in key.lower():
                image_features[key] = key
        if not image_features:
            logger.warning("No image features detected in dataset")
            return {}

        # Map to π0.5 expected names
        result = {}
        for i, (col_name, _) in enumerate(image_features.items()):
            cam_name = f"cam_{i}_rgb"
            result[col_name] = cam_name
        return result

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        return self._tokenizer

    def _load_episode(self, ep_idx: int):
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
        state = np.array(row[self.state_col], dtype=np.float32)
        if self.normalize and self.stats and self.state_col in self.stats:
            s = self.stats[self.state_col]
            state = normalize_zscore(
                state,
                np.array(s["mean"], dtype=np.float32),
                np.array(s["std"], dtype=np.float32),
            )
        # Pad state if needed
        if self.pad_state_dim > 0 and len(state) < self.pad_state_dim:
            state = np.pad(state, (0, self.pad_state_dim - len(state)))

        # --- Actions (chunked) ---
        actions = get_action_chunk(df, step_idx, self.action_col, self.action_horizon)
        if self.normalize and self.stats and self.action_col in self.stats:
            s = self.stats[self.action_col]
            actions = normalize_zscore(
                actions,
                np.array(s["mean"], dtype=np.float32),
                np.array(s["std"], dtype=np.float32),
            )
        # Pad actions if needed
        if self.pad_action_dim > 0 and actions.shape[1] < self.pad_action_dim:
            pad_width = ((0, 0), (0, self.pad_action_dim - actions.shape[1]))
            actions = np.pad(actions, pad_width)

        # --- Images ---
        images = {}
        image_masks = {}
        for col_name, cam_name in self.image_key_map.items():
            features_spec = self.info.get("features", {}).get(col_name, {})
            dtype = features_spec.get("dtype", "")

            if dtype == "video":
                # Video: decode frame
                video_path = self._get_video_path(ep_idx, col_name)
                if video_path.exists():
                    images[cam_name] = decode_video_frame(
                        video_path, step_idx, self.image_size
                    )
                    image_masks[cam_name] = torch.tensor(True)
                else:
                    images[cam_name] = torch.zeros(3, *self.image_size)
                    image_masks[cam_name] = torch.tensor(False)
            elif dtype == "image":
                # Image stored in parquet as bytes or path
                img_data = row.get(col_name)
                if img_data is not None:
                    images[cam_name] = self._parse_image(img_data)
                    image_masks[cam_name] = torch.tensor(True)
                else:
                    images[cam_name] = torch.zeros(3, *self.image_size)
                    image_masks[cam_name] = torch.tensor(False)
            else:
                images[cam_name] = torch.zeros(3, *self.image_size)
                image_masks[cam_name] = torch.tensor(False)

        # --- Prompt ---
        ep_meta = self.episodes[ep_idx]
        if self.prompt_col and self.prompt_col in row:
            prompt = str(row[self.prompt_col])
        elif ep_meta.get("tasks"):
            prompt = ep_meta["tasks"][0]
        elif "task_index" in row:
            task_idx = int(row["task_index"])
            prompt = self.tasks.get(task_idx, "manipulation task")
        else:
            prompt = "manipulation task"

        # --- Tokenize prompt ---
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "images": images,
            "image_masks": image_masks,
            "tokenized_prompt": encoded["input_ids"].squeeze(0),
            "tokenized_prompt_mask": encoded["attention_mask"].squeeze(0).bool(),
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
        }

    def _get_video_path(self, ep_idx: int, video_key: str) -> Path:
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

    def _parse_image(self, img_data) -> torch.Tensor:
        """Parse image data from parquet (numpy array, bytes, or path)."""
        import torchvision.transforms.functional as TF

        if isinstance(img_data, np.ndarray):
            if img_data.ndim == 3 and img_data.shape[2] == 3:
                # (H, W, 3) → (3, H, W)
                tensor = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0
            elif img_data.ndim == 3 and img_data.shape[0] == 3:
                # Already (3, H, W)
                tensor = torch.from_numpy(img_data).float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
            else:
                return torch.zeros(3, *self.image_size)
            return TF.resize(tensor, list(self.image_size), antialias=True)
        elif isinstance(img_data, (bytes, bytearray)):
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]))
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        elif isinstance(img_data, dict) and "path" in img_data:
            from PIL import Image

            img = Image.open(img_data["path"]).convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]))
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        else:
            return torch.zeros(3, *self.image_size)
