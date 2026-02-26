"""Stage 2 Verification: Switching gate (β_t) alignment analysis.

Analyzes whether the metacontroller's switching gate β_t aligns
with ground truth subgoal transitions.

Expected result (Fig 6, A3):
- β_t exhibits quasi-binary behavior (most values near 0 or 1)
- β_t ≈ 1 at timesteps where the ground truth subgoal changes
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..models.transformer import CausalTransformer
from ..models.metacontroller import MetaController
from ..data.dataset import TrajectoryDataset
from torch.utils.data import DataLoader


def analyze_beta_alignment(
    base_model: CausalTransformer,
    metacontroller: MetaController,
    dataset: TrajectoryDataset,
    controlled_layer: int = 3,
    num_trajectories: int = 10,
    device: str = "cuda",
    save_dir: str | Path = "results",
) -> dict[str, float]:
    """Analyze β_t alignment with ground truth subgoal transitions.

    Returns:
        Dictionary with metrics:
            binary_fraction: fraction of β values near 0 or 1
            alignment_precision: precision of β spikes at true transitions
            alignment_recall: recall of true transitions detected by β spikes
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_model = base_model.to(device).eval()
    metacontroller = metacontroller.to(device).eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_betas = []
    all_subgoals = []
    all_masks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_trajectories:
                break

            obs = batch["observations"][:, :-1].to(device)
            subgoals = batch["subgoals"].to(device)
            mask = batch["mask"].to(device)
            length = batch["length"].item()

            # Extract residual stream
            e, _ = base_model.forward_up_to_layer(obs, controlled_layer)

            # Run metacontroller
            mc_output = metacontroller(e)
            beta = mc_output["beta_seq"].squeeze(-1)  # (1, T)

            all_betas.append(beta[0, :length].cpu().numpy())
            all_subgoals.append(subgoals[0, :length].cpu().numpy())
            all_masks.append(mask[0, :length].cpu().numpy())

    # Compute metrics
    binary_fractions = []
    precisions = []
    recalls = []

    for beta, sg, m in zip(all_betas, all_subgoals, all_masks):
        valid = m > 0
        b = beta[valid]

        # Binary fraction: values near 0 or 1
        binary_frac = np.mean((b > 0.9) | (b < 0.1))
        binary_fractions.append(binary_frac)

        # Ground truth transitions: where subgoal changes
        sg_valid = sg[valid]
        true_switches = np.zeros_like(sg_valid, dtype=bool)
        true_switches[1:] = sg_valid[1:] != sg_valid[:-1]
        true_switches[0] = True  # First step is always a "switch"

        # Detected switches: β > 0.5
        detected_switches = b > 0.5

        # Precision: of detected switches, how many are true?
        if detected_switches.sum() > 0:
            precision = (detected_switches & true_switches).sum() / detected_switches.sum()
        else:
            precision = 0.0
        precisions.append(precision)

        # Recall: of true switches, how many are detected?
        if true_switches.sum() > 0:
            recall = (detected_switches & true_switches).sum() / true_switches.sum()
        else:
            recall = 0.0
        recalls.append(recall)

    metrics = {
        "binary_fraction": np.mean(binary_fractions),
        "alignment_precision": np.mean(precisions),
        "alignment_recall": np.mean(recalls),
    }

    # Plot example trajectories
    n_plot = min(3, len(all_betas))
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        ax = axes[i]
        beta = all_betas[i]
        sg = all_subgoals[i]
        T = len(beta)

        # Plot β_t
        ax.plot(range(T), beta, color="blue", alpha=0.7, label="β_t (switch gate)")

        # Color-code ground truth subgoals
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for t in range(T):
            ax.axvspan(t - 0.5, t + 0.5, color=colors[sg[t] % 10], alpha=0.15)

        # Mark true transition points
        for t in range(1, T):
            if sg[t] != sg[t - 1]:
                ax.axvline(t, color="red", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("β_t")
        if i == 0:
            ax.set_title("Switching Gate β_t vs Ground Truth Subgoal Transitions")
        if i == n_plot - 1:
            ax.set_xlabel("Time step")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "beta_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"β analysis metrics: {metrics}")
    print(f"Saved plot to {save_dir / 'beta_analysis.png'}")

    return metrics
