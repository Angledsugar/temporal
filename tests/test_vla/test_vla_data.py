"""Tests for VLA dummy datasets."""

import pytest
import torch

from internalrl.vla.data.dummy_dataset import (
    DummyGrootDataset,
    DummyPi05Dataset,
    DummyResidualDataset,
)


class TestDummyResidualDataset:
    """Test the simplest residual-only dataset."""

    def test_length(self):
        ds = DummyResidualDataset(num_samples=10)
        assert len(ds) == 10

    def test_item_shapes(self):
        ds = DummyResidualDataset(
            seq_len=50, embed_dim=2048, action_dim=32, action_horizon=50
        )
        item = ds[0]
        assert item["residual"].shape == (50, 2048)
        assert item["actions"].shape == (50, 32)
        assert item["noise"].shape == (50, 32)
        assert item["time"].dim() == 0  # scalar

    def test_custom_dims(self):
        ds = DummyResidualDataset(
            seq_len=20, embed_dim=64, action_dim=7, action_horizon=16
        )
        item = ds[0]
        assert item["residual"].shape == (20, 64)
        assert item["actions"].shape == (16, 7)


class TestDummyPi05Dataset:
    """Test π0.5 format dataset."""

    def test_length(self):
        ds = DummyPi05Dataset(num_samples=5)
        assert len(ds) == 5

    def test_item_format(self):
        ds = DummyPi05Dataset(
            num_images=2,
            image_size=(224, 224),
            max_token_len=48,
            state_dim=32,
            action_dim=32,
            action_horizon=50,
        )
        item = ds[0]

        # Images
        assert "images" in item
        assert len(item["images"]) == 2
        for name, img in item["images"].items():
            assert img.shape == (3, 224, 224)

        # Image masks
        assert "image_masks" in item
        for name, mask in item["image_masks"].items():
            assert mask.dtype == torch.bool

        # Language tokens
        assert item["tokenized_prompt"].shape == (48,)
        assert item["tokenized_prompt_mask"].shape == (48,)
        assert item["tokenized_prompt_mask"].dtype == torch.bool

        # State and actions
        assert item["state"].shape == (32,)
        assert item["actions"].shape == (50, 32)

    def test_single_camera(self):
        ds = DummyPi05Dataset(num_images=1)
        item = ds[0]
        assert len(item["images"]) == 1


class TestDummyGrootDataset:
    """Test Groot format dataset."""

    def test_length(self):
        ds = DummyGrootDataset(num_samples=7)
        assert len(ds) == 7

    def test_item_format(self):
        ds = DummyGrootDataset(
            num_images=1,
            image_size=(224, 224),
            state_dim=29,
            action_dim=29,
            action_horizon=16,
        )
        item = ds[0]

        # Images
        assert "images" in item
        for name, img in item["images"].items():
            assert img.shape == (3, 224, 224)

        # State and actions
        assert item["state"].shape == (29,)
        assert item["actions"].shape == (16, 29)

        # Text
        assert isinstance(item["text"], str)

        # Tokenized
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_different_action_dims(self):
        ds = DummyGrootDataset(action_dim=7, action_horizon=8)
        item = ds[0]
        assert item["actions"].shape == (8, 7)
