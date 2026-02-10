#!/usr/bin/env python3
"""Evaluate TempoRAL: metrics computation and visualization.

Usage:
    python scripts/evaluate.py \
        --expert checkpoints/phase1/action_expert.pt \
        --meta checkpoints/expert_distill/metacontroller.pt \
        --policy checkpoints/phase3/rl_policy.pt \
        --data /path/to/eval_data \
        --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from temporal.models.action_expert import ActionExpertWrapper
from temporal.models.internal_rl_policy import CausalRLPolicy
from temporal.models.metacontroller import MetaController
from temporal.utils.metrics import (
    success_rate,
    switching_accuracy,
    switching_nmi,
    temporal_contraction_ratio,
)
from temporal.utils.visualization import plot_beta_sequence, plot_z_tsne

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_switching(
    metacontroller: MetaController,
    expert: ActionExpertWrapper,
    eval_data: list[dict],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Evaluate switching boundary detection."""
    all_nmi = []
    all_f1 = []
    all_tcr = []

    for i, sample in enumerate(eval_data):
        # TODO: Extract residual stream from actual data
        # e_seq = expert.extract_residual_stream(sample["observations"])
        T = sample.get("length", 100)
        n_e = 1024

        # Placeholder: random residual stream
        e_seq = torch.randn(1, T, n_e, device=device)

        with torch.no_grad():
            z_seq, _, beta_seq = metacontroller(e_seq)

        beta_np = beta_seq[0].cpu().numpy()
        gt_boundaries = sample.get("boundaries", np.zeros(T))

        # Metrics
        nmi = switching_nmi(beta_np, gt_boundaries)
        f1 = switching_accuracy(beta_np, gt_boundaries, tolerance=2)
        num_switches = int((beta_np > 0.5).sum())
        tcr = temporal_contraction_ratio(T, max(num_switches, 1))

        all_nmi.append(nmi)
        all_f1.append(f1)
        all_tcr.append(tcr)

        # Visualize first few
        if i < 5:
            plot_beta_sequence(
                beta_np,
                ground_truth_boundaries=gt_boundaries,
                save_path=output_dir / f"beta_seq_{i}.png",
            )

    return {
        "switching_nmi_mean": float(np.mean(all_nmi)),
        "switching_nmi_std": float(np.std(all_nmi)),
        "switching_f1_mean": float(np.mean(all_f1)),
        "switching_f1_std": float(np.std(all_f1)),
        "tcr_mean": float(np.mean(all_tcr)),
        "tcr_std": float(np.std(all_tcr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TempoRAL Evaluation")
    parser.add_argument("--expert", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--data", type=str, default=None, help="Evaluation data directory")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    expert = ActionExpertWrapper(width=1024, depth=18, intervention_layer=9).to(device)
    expert.load_state_dict(torch.load(args.expert, map_location=device)["model_state_dict"])
    expert.eval()

    metacontroller = MetaController(n_e=1024, n_z=32).to(device)
    metacontroller.load_state_dict(torch.load(args.meta, map_location=device)["model_state_dict"])
    metacontroller.eval()

    # TODO: Load actual evaluation data
    # For now, generate placeholder data
    eval_data = [
        {"length": 100, "boundaries": np.random.binomial(1, 0.05, size=100).astype(float)}
        for _ in range(20)
    ]

    logger.info(f"Evaluating on {len(eval_data)} sequences...")

    # Switching evaluation
    switching_results = evaluate_switching(
        metacontroller, expert, eval_data, device, output_dir
    )

    logger.info("=== Switching Evaluation ===")
    for k, v in switching_results.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save results
    results = {"switching": switching_results}
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
