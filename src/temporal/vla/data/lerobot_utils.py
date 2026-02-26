"""Common utilities for loading LeRobot-format datasets.

Shared by both GrootLeRobotDataset and Pi05LeRobotDataset.
Supports local directories and HuggingFace Hub downloads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


def download_lerobot_dataset(
    repo_id: str,
    local_dir: str | None = None,
    repo_type: str = "dataset",
) -> Path:
    """Download a LeRobot dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot").
        local_dir: Optional local directory to save to. Uses HF cache if None.
        repo_type: Repository type (default: "dataset").

    Returns:
        Path to the downloaded dataset directory.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
    )
    logger.info(f"Downloaded dataset {repo_id} to {path}")
    return Path(path)


def load_lerobot_metadata(dataset_path: Path) -> dict:
    """Parse LeRobot metadata files (info.json, episodes.jsonl, tasks.jsonl, stats.json).

    Args:
        dataset_path: Root directory of the LeRobot dataset.

    Returns:
        dict with keys: info, episodes, tasks, stats, modality (if exists).
    """
    meta_dir = dataset_path / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"meta/ directory not found in {dataset_path}")

    # info.json
    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    # episodes.jsonl
    episodes = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    # tasks.jsonl
    tasks = {}
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    tasks[entry["task_index"]] = entry["task"]

    # stats.json
    stats = None
    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    # modality.json (GROOT-specific)
    modality = None
    modality_path = meta_dir / "modality.json"
    if modality_path.exists():
        with open(modality_path) as f:
            modality = json.load(f)

    return {
        "info": info,
        "episodes": episodes,
        "tasks": tasks,
        "stats": stats,
        "modality": modality,
    }


def load_episode_parquet(dataset_path: Path, episode_idx: int, info: dict) -> pd.DataFrame:
    """Load a single episode's parquet file.

    Args:
        dataset_path: Root directory of the LeRobot dataset.
        episode_idx: Episode index.
        info: Parsed info.json dict.

    Returns:
        DataFrame with episode data.
    """
    chunks_size = info.get("chunks_size", 1000)
    chunk_id = episode_idx // chunks_size
    data_path_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel_path = data_path_template.format(
        episode_chunk=chunk_id, episode_index=episode_idx
    )
    parquet_path = dataset_path / rel_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def decode_video_frame(
    video_path: Path,
    frame_idx: int,
    size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Decode a single video frame and resize.

    Args:
        video_path: Path to MP4 video file.
        frame_idx: Frame index to extract.
        size: Target (H, W) size.

    Returns:
        Tensor of shape (3, H, W) in [0, 1] range.
    """
    try:
        import torchvision.io as tio

        # Read specific frames using torchvision
        video, _, _ = tio.read_video(
            str(video_path),
            pts_unit="sec",
            output_format="TCHW",
        )
        if frame_idx >= len(video):
            frame_idx = len(video) - 1
        frame = video[frame_idx].float() / 255.0  # (3, H, W)
    except Exception:
        # Fallback: use PIL + imageio
        try:
            import imageio.v3 as iio

            frames = iio.imread(str(video_path), plugin="pyav")
            if frame_idx >= len(frames):
                frame_idx = len(frames) - 1
            frame_np = frames[frame_idx]  # (H, W, 3) uint8
            frame = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        except Exception:
            # Last resort: return zeros
            logger.warning(f"Failed to decode frame {frame_idx} from {video_path}")
            return torch.zeros(3, *size)

    # Resize to target size
    frame = TF.resize(frame, list(size), antialias=True)
    return frame


def compute_global_index_map(episodes: list[dict]) -> list[tuple[int, int]]:
    """Build global index → (episode_idx, step_idx) mapping.

    Args:
        episodes: List of episode metadata dicts with 'episode_index' and 'length'.

    Returns:
        List where index i maps to (episode_index, step_within_episode).
    """
    index_map = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        for step in range(ep_len):
            index_map.append((ep_idx, step))
    return index_map


def normalize_minmax(value: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Min-max normalization to [-1, 1] range (GROOT default)."""
    range_ = np.maximum(vmax - vmin, 1e-8)
    return 2.0 * (value - vmin) / range_ - 1.0


def normalize_zscore(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization (OpenPI default)."""
    std = np.maximum(std, 1e-8)
    return (value - mean) / std


def get_action_chunk(
    df: pd.DataFrame,
    step_idx: int,
    action_col: str,
    action_horizon: int,
) -> np.ndarray:
    """Extract an action chunk from current step to step + action_horizon.

    Pads by repeating last action if near episode end.

    Args:
        df: Episode DataFrame.
        step_idx: Current step index within episode.
        action_col: Column name for actions.
        action_horizon: Number of future steps.

    Returns:
        Array of shape (action_horizon, action_dim).
    """
    actions = []
    ep_len = len(df)
    for t in range(action_horizon):
        idx = min(step_idx + t, ep_len - 1)
        a = np.array(df.iloc[idx][action_col], dtype=np.float32)
        actions.append(a)
    return np.stack(actions)
