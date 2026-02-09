"""Visualization utilities for TempoRAL.

- beta_t switching patterns over time
- z_t t-SNE clustering
- Boundary overlay on action sequences
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_beta_sequence(
    beta_seq: np.ndarray,
    ground_truth_boundaries: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot beta_t switching pattern over time.

    Args:
        beta_seq: (T,) -- switching probabilities.
        ground_truth_boundaries: (T,) -- binary ground truth boundaries.
        save_path: Path to save figure. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(beta_seq, label="β_t (predicted)", color="blue", linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="threshold")

    if ground_truth_boundaries is not None:
        for t in np.where(ground_truth_boundaries > 0.5)[0]:
            ax.axvline(x=t, color="red", alpha=0.3, linewidth=1)
        ax.axvline(x=-1, color="red", alpha=0.3, label="ground truth boundary")

    ax.set_xlabel("Time step")
    ax.set_ylabel("β_t")
    ax.set_title("Switching Gate Pattern")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_z_tsne(
    z_seq: np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """t-SNE visualization of controller codes z_t.

    Args:
        z_seq: (N, n_z) -- controller codes from multiple trajectories.
        labels: (N,) -- cluster labels (e.g. subtask index).
        save_path: Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        return

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_seq)

    fig, ax = plt.subplots(figsize=(8, 8))
    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6)
        fig.colorbar(scatter, ax=ax, label="Subtask")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6)

    ax.set_title("Controller Code z_t (t-SNE)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
