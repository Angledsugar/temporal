"""Visualization utilities for gridworld and training results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..envs.gridworld import GridworldPinpad


COLOR_MAP = {
    0: "#FF0000",  # Red
    1: "#00FF00",  # Green
    2: "#0000FF",  # Blue
    3: "#FFFF00",  # Yellow
    4: "#FF00FF",  # Magenta
    5: "#00FFFF",  # Cyan
    6: "#FF8000",  # Orange
    7: "#8000FF",  # Purple
}


def visualize_grid(
    env: GridworldPinpad,
    save_path: str | Path | None = None,
    title: str = "Gridworld-Pinpad",
) -> None:
    """Visualize the current grid state."""
    fig, ax = plt.subplots(figsize=(6, 6))
    G = env.grid_size

    # Draw grid
    for r in range(G):
        for c in range(G):
            rect = plt.Rectangle((c, G - 1 - r), 1, 1, fill=True,
                                 facecolor="white", edgecolor="gray")
            ax.add_patch(rect)

    # Draw walls
    for (r, c) in env.wall_positions:
        rect = plt.Rectangle((c, G - 1 - r), 1, 1, fill=True,
                             facecolor="gray", edgecolor="black")
        ax.add_patch(rect)

    # Draw colored cells
    for color_idx, (r, c) in env.color_positions.items():
        rect = plt.Rectangle((c, G - 1 - r), 1, 1, fill=True,
                             facecolor=COLOR_MAP.get(color_idx, "#AAAAAA"),
                             edgecolor="black", alpha=0.7)
        ax.add_patch(rect)
        ax.text(c + 0.5, G - 1 - r + 0.5, str(color_idx),
                ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw agent
    ar, ac = env.agent_pos
    circle = plt.Circle((ac + 0.5, G - 1 - ar + 0.5), 0.3,
                        color="black", fill=True)
    ax.add_patch(circle)

    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
