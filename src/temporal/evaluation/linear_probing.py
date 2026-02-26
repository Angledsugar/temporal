"""Stage 1 Verification: Linear probing of residual stream (Appendix C.2).

Trains linear classifiers on frozen residual stream activations to
decode the current subgoal at each timestep.

Expected result (Fig 3, A1):
- Accuracy increases from ~30% at layer 0 to ~50%+ at mid-depth (layer 4+)
- Accuracy peaks near layer 5 and may drop slightly at final layer 6
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from ..models.transformer import CausalTransformer
from ..data.dataset import TrajectoryDataset
from ..envs.tasks import NUM_COLORS


class LinearProbe(nn.Module):
    """Linear classifier for subgoal prediction from residual stream."""

    def __init__(self, embed_dim: int, num_subgoals: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_subgoals)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.linear(e)


def run_linear_probing(
    base_model: CausalTransformer,
    dataset: TrajectoryDataset,
    num_subgoals: int = NUM_COLORS,
    probe_layers: list[int] | None = None,
    train_steps: int = 8000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cuda",
    save_dir: str | Path = "results",
) -> dict[int, float]:
    """Run linear probing at each layer of the frozen base model.

    Args:
        base_model: Pretrained, frozen base model.
        dataset: Trajectory dataset with subgoal labels.
        num_subgoals: Number of subgoal classes.
        probe_layers: Which layers to probe (default: all).
        train_steps: Training steps per probe.
        batch_size: Batch size.
        lr: Learning rate.
        device: Torch device.
        save_dir: Directory for results.

    Returns:
        Dictionary mapping layer index to accuracy.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_model = base_model.to(device)
    base_model.eval()

    if probe_layers is None:
        probe_layers = list(range(base_model.num_layers + 1))  # 0..L

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    results = {}

    for layer in probe_layers:
        print(f"\nProbing layer {layer}...")
        probe = LinearProbe(base_model.embed_dim, num_subgoals).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

        probe.train()
        step = 0
        pbar = tqdm(total=train_steps, desc=f"Layer {layer}")
        total_correct = 0
        total_count = 0

        while step < train_steps:
            for batch in dataloader:
                if step >= train_steps:
                    break

                obs = batch["observations"][:, :-1].to(device)  # (B, T, obs_dim)
                subgoals = batch["subgoals"].to(device)          # (B, T)
                mask = batch["mask"].to(device)                   # (B, T)

                with torch.no_grad():
                    result = base_model(obs, return_residuals=True)
                    e = result["residuals"][layer]  # (B, T, embed_dim)

                logits = probe(e)  # (B, T, num_subgoals)
                loss = F.cross_entropy(
                    logits.reshape(-1, num_subgoals),
                    subgoals.reshape(-1),
                    reduction="none",
                ).reshape_as(subgoals)
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track accuracy
                preds = logits.argmax(dim=-1)
                correct = ((preds == subgoals) * mask).sum().item()
                total_correct += correct
                total_count += mask.sum().item()

                step += 1
                pbar.update(1)

                if step % 500 == 0:
                    acc = total_correct / max(total_count, 1)
                    pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{loss.item():.4f}")
                    total_correct = 0
                    total_count = 0

        pbar.close()

        # Final evaluation accuracy
        probe.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                obs = batch["observations"][:, :-1].to(device)
                subgoals = batch["subgoals"].to(device)
                mask = batch["mask"].to(device)

                result = base_model(obs, return_residuals=True)
                e = result["residuals"][layer]
                logits = probe(e)
                preds = logits.argmax(dim=-1)
                eval_correct += ((preds == subgoals) * mask).sum().item()
                eval_total += mask.sum().item()

        accuracy = eval_correct / max(eval_total, 1)
        results[layer] = accuracy
        print(f"Layer {layer}: accuracy = {accuracy:.4f}")

    # Plot results
    layers = sorted(results.keys())
    accs = [results[l] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(layers, accs, color="steelblue", alpha=0.8)
    ax.set_xlabel("Probing Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probing: Subgoal Decoding from Residual Stream")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    fig.savefig(save_dir / "linear_probing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to {save_dir / 'linear_probing.png'}")
    return results
