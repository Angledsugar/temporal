"""Dummy datasets for VLA pipeline structure verification.

Generates random tensors matching the input formats of π0.5 and Groot,
allowing end-to-end pipeline testing without real data or model weights.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyResidualDataset(Dataset):
    """Simplest dummy: random residual streams for MC-only testing.

    Bypasses the VLA model entirely — useful for testing metacontroller
    forward/backward without loading a multi-billion parameter model.

    Args:
        num_samples: Number of samples.
        seq_len: Sequence length (prefix tokens).
        embed_dim: VLM hidden dimension.
        action_dim: Action space dimension.
        action_horizon: Number of action steps per chunk.
    """

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 50,
        embed_dim: int = 2048,
        action_dim: int = 32,
        action_horizon: int = 50,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "residual": torch.randn(self.seq_len, self.embed_dim),
            "actions": torch.randn(self.action_horizon, self.action_dim),
            "noise": torch.randn(self.action_horizon, self.action_dim),
            "time": torch.rand(1).squeeze(0),
        }


class DummyPi05Dataset(Dataset):
    """Dummy dataset mimicking π0.5 input format.

    Produces random tensors matching the PI0Pytorch.forward() input format.

    Args:
        num_samples: Number of samples.
        num_images: Number of camera views.
        image_size: (H, W) resolution.
        max_token_len: Maximum language token length.
        state_dim: State vector dimension.
        action_dim: Action space dimension.
        action_horizon: Number of action steps per chunk.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_images: int = 1,
        image_size: tuple[int, int] = (224, 224),
        max_token_len: int = 48,
        state_dim: int = 32,
        action_dim: int = 32,
        action_horizon: int = 50,
    ):
        self.num_samples = num_samples
        self.num_images = num_images
        self.image_size = image_size
        self.max_token_len = max_token_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        H, W = self.image_size
        camera_names = [f"cam_{i}_rgb" for i in range(self.num_images)]

        images = {name: torch.randn(3, H, W) for name in camera_names}
        image_masks = {name: torch.tensor(True) for name in camera_names}

        return {
            "images": images,
            "image_masks": image_masks,
            "tokenized_prompt": torch.randint(0, 1000, (self.max_token_len,)),
            "tokenized_prompt_mask": torch.ones(self.max_token_len, dtype=torch.bool),
            "state": torch.randn(self.state_dim),
            "actions": torch.randn(self.action_horizon, self.action_dim),
        }


class DummyGrootDataset(Dataset):
    """Dummy dataset mimicking Groot N1.6 input format.

    Produces random tensors matching Gr00tN1d6 input format
    (images, states, actions, text).

    Args:
        num_samples: Number of samples.
        num_images: Number of camera views.
        image_size: (H, W) resolution.
        state_dim: State vector dimension.
        action_dim: Action space dimension.
        action_horizon: Number of action steps per chunk.
        max_text_len: Maximum text token length.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_images: int = 1,
        image_size: tuple[int, int] = (224, 224),
        state_dim: int = 29,
        action_dim: int = 29,
        action_horizon: int = 16,
        max_text_len: int = 64,
    ):
        self.num_samples = num_samples
        self.num_images = num_images
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        H, W = self.image_size
        camera_names = [f"cam_{i}_rgb" for i in range(self.num_images)]

        images = {name: torch.randn(3, H, W) for name in camera_names}

        return {
            "images": images,
            "state": torch.randn(self.state_dim),
            "actions": torch.randn(self.action_horizon, self.action_dim),
            "text": "pick up the object",
            "input_ids": torch.randint(0, 1000, (self.max_text_len,)),
            "attention_mask": torch.ones(self.max_text_len, dtype=torch.bool),
        }
