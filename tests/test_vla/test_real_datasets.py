"""Tests for real LeRobot dataset adapters (Groot + π0.5).

Uses Isaac-GR00T demo data (cube_to_bowl_5) for integration tests.
Tests lerobot_utils independently with mock data.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from temporal.vla.data.lerobot_utils import (
    compute_global_index_map,
    get_action_chunk,
    load_lerobot_metadata,
    normalize_minmax,
    normalize_zscore,
)


# ------------------------------------------------------------------ #
# Fixtures: create a minimal mock LeRobot dataset on disk             #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_lerobot_dir(tmp_path: Path):
    """Create a minimal LeRobot-format dataset directory for testing."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps({"episode_index": 0, "tasks": ["pick up object"], "length": 10}) + "\n")
        f.write(json.dumps({"episode_index": 1, "tasks": ["place object"], "length": 10}) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "pick up object"}) + "\n")

    # stats.json
    stats = {
        "observation.state": {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "min": [-1.0] * 6,
            "max": [1.0] * 6,
        },
        "action": {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "min": [-1.0] * 6,
            "max": [1.0] * 6,
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    # Create parquet files
    for ep_idx in range(2):
        rows = []
        for step in range(10):
            rows.append({
                "observation.state": np.random.randn(6).astype(np.float32).tolist(),
                "action": np.random.randn(6).astype(np.float32).tolist(),
                "timestamp": float(step) / 10.0,
                "episode_index": ep_idx,
                "index": ep_idx * 10 + step,
                "task_index": 0,
            })
        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet")

    return tmp_path


# ------------------------------------------------------------------ #
# lerobot_utils tests                                                 #
# ------------------------------------------------------------------ #

class TestLeRobotUtils:
    def test_load_metadata(self, mock_lerobot_dir):
        meta = load_lerobot_metadata(mock_lerobot_dir)
        assert meta["info"]["robot_type"] == "test_robot"
        assert len(meta["episodes"]) == 2
        assert meta["tasks"][0] == "pick up object"
        assert "observation.state" in meta["stats"]

    def test_load_metadata_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lerobot_metadata(tmp_path / "nonexistent")

    def test_compute_global_index_map(self):
        episodes = [
            {"episode_index": 0, "length": 5},
            {"episode_index": 1, "length": 3},
        ]
        index_map = compute_global_index_map(episodes)
        assert len(index_map) == 8
        assert index_map[0] == (0, 0)
        assert index_map[4] == (0, 4)
        assert index_map[5] == (1, 0)
        assert index_map[7] == (1, 2)

    def test_normalize_minmax(self):
        value = np.array([0.5, -0.5])
        vmin = np.array([-1.0, -1.0])
        vmax = np.array([1.0, 1.0])
        result = normalize_minmax(value, vmin, vmax)
        np.testing.assert_allclose(result, [0.5, -0.5])

    def test_normalize_zscore(self):
        value = np.array([2.0, -1.0])
        mean = np.array([0.0, 0.0])
        std = np.array([1.0, 1.0])
        result = normalize_zscore(value, mean, std)
        np.testing.assert_allclose(result, [2.0, -1.0])

    def test_get_action_chunk(self, mock_lerobot_dir):
        from temporal.vla.data.lerobot_utils import load_episode_parquet
        meta = load_lerobot_metadata(mock_lerobot_dir)
        df = load_episode_parquet(mock_lerobot_dir, 0, meta["info"])
        chunk = get_action_chunk(df, 0, "action", 4)
        assert chunk.shape == (4, 6)

    def test_get_action_chunk_end_padding(self, mock_lerobot_dir):
        """Action chunk at end of episode should pad by repeating last action."""
        from temporal.vla.data.lerobot_utils import load_episode_parquet
        meta = load_lerobot_metadata(mock_lerobot_dir)
        df = load_episode_parquet(mock_lerobot_dir, 0, meta["info"])
        chunk = get_action_chunk(df, 8, "action", 4)
        assert chunk.shape == (4, 6)
        # Last two actions should be the same (repeated)
        np.testing.assert_array_equal(chunk[2], chunk[3])


# ------------------------------------------------------------------ #
# GrootLeRobotDataset tests                                          #
# ------------------------------------------------------------------ #

class TestGrootLeRobotDataset:
    def test_load_from_mock(self, mock_lerobot_dir):
        """Test GrootLeRobotDataset with mock data (no video)."""
        # Add modality.json for Groot
        modality = {
            "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
            "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
            "video": {},
            "annotation": {},
        }
        with open(mock_lerobot_dir / "meta" / "modality.json", "w") as f:
            json.dump(modality, f)

        from temporal.vla.data.groot_dataset import GrootLeRobotDataset
        ds = GrootLeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            tokenizer_name="bert-base-uncased",  # small tokenizer for test
        )
        assert len(ds) == 20

        sample = ds[0]
        assert "state" in sample
        assert "actions" in sample
        assert "text" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert sample["state"].shape == (6,)
        assert sample["actions"].shape == (4, 6)
        assert isinstance(sample["text"], str)

    def test_output_format_matches_dummy(self, mock_lerobot_dir):
        """Ensure output keys match DummyGrootDataset."""
        modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
        with open(mock_lerobot_dir / "meta" / "modality.json", "w") as f:
            json.dump(modality, f)

        from temporal.vla.data.groot_dataset import GrootLeRobotDataset
        from temporal.vla.data.dummy_dataset import DummyGrootDataset

        real_ds = GrootLeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            tokenizer_name="bert-base-uncased",
        )
        dummy_ds = DummyGrootDataset(
            num_samples=5,
            state_dim=6,
            action_dim=6,
            action_horizon=4,
        )

        real_sample = real_ds[0]
        dummy_sample = dummy_ds[0]

        # Same top-level keys
        assert set(real_sample.keys()) == set(dummy_sample.keys())


# ------------------------------------------------------------------ #
# Pi05LeRobotDataset tests                                           #
# ------------------------------------------------------------------ #

class TestPi05LeRobotDataset:
    def test_load_from_mock(self, mock_lerobot_dir):
        """Test Pi05LeRobotDataset with mock data (no images)."""
        from temporal.vla.data.pi05_dataset import Pi05LeRobotDataset
        ds = Pi05LeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            tokenizer_name="bert-base-uncased",
        )
        assert len(ds) == 20

        sample = ds[0]
        assert "state" in sample
        assert "actions" in sample
        assert "images" in sample
        assert "image_masks" in sample
        assert "tokenized_prompt" in sample
        assert "tokenized_prompt_mask" in sample
        assert sample["state"].shape == (6,)
        assert sample["actions"].shape == (4, 6)

    def test_output_format_matches_dummy(self, mock_lerobot_dir):
        """Ensure output keys match DummyPi05Dataset."""
        from temporal.vla.data.pi05_dataset import Pi05LeRobotDataset
        from temporal.vla.data.dummy_dataset import DummyPi05Dataset

        real_ds = Pi05LeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            tokenizer_name="bert-base-uncased",
        )
        dummy_ds = DummyPi05Dataset(
            num_samples=5,
            state_dim=6,
            action_dim=6,
            action_horizon=4,
        )

        real_sample = real_ds[0]
        dummy_sample = dummy_ds[0]

        # Same top-level keys
        assert set(real_sample.keys()) == set(dummy_sample.keys())

    def test_state_padding(self, mock_lerobot_dir):
        """Test that state is padded when pad_state_dim > actual dim."""
        from temporal.vla.data.pi05_dataset import Pi05LeRobotDataset
        ds = Pi05LeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            pad_state_dim=32,
            tokenizer_name="bert-base-uncased",
        )
        sample = ds[0]
        assert sample["state"].shape == (32,)

    def test_action_padding(self, mock_lerobot_dir):
        """Test that actions are padded when pad_action_dim > actual dim."""
        from temporal.vla.data.pi05_dataset import Pi05LeRobotDataset
        ds = Pi05LeRobotDataset(
            local_path=str(mock_lerobot_dir),
            action_horizon=4,
            pad_action_dim=32,
            tokenizer_name="bert-base-uncased",
        )
        sample = ds[0]
        assert sample["actions"].shape == (4, 32)


# ------------------------------------------------------------------ #
# Integration test with real GROOT demo data                          #
# ------------------------------------------------------------------ #

GROOT_DEMO = Path("Isaac-GR00T/demo_data/cube_to_bowl_5")


def _groot_demo_has_real_data() -> bool:
    """Check if GROOT demo data is available and not just LFS pointers."""
    if not GROOT_DEMO.exists():
        return False
    parquet = GROOT_DEMO / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet.exists():
        return False
    # LFS pointer files are small ASCII; real parquet files are larger binary
    return parquet.stat().st_size > 1000


@pytest.mark.skipif(
    not _groot_demo_has_real_data(),
    reason="Isaac-GR00T demo data not available or is LFS pointer",
)
class TestGrootDemoData:
    def test_load_real_demo(self):
        """Test loading actual GROOT demo dataset."""
        from temporal.vla.data.groot_dataset import GrootLeRobotDataset
        ds = GrootLeRobotDataset(
            local_path=str(GROOT_DEMO),
            action_horizon=16,
            tokenizer_name="bert-base-uncased",
        )
        assert len(ds) > 0

        sample = ds[0]
        assert sample["state"].shape[0] == 6  # SO101: 6-dim state
        assert sample["actions"].shape == (16, 6)
        assert "images" in sample
        assert isinstance(sample["text"], str)
        assert len(sample["text"]) > 0
