"""Evaluation metrics for TempoRAL.

- Subtask success rate
- Switching accuracy (NMI with ground truth)
- Temporal contraction ratio
"""

from __future__ import annotations

import numpy as np


def success_rate(results: list[dict]) -> float:
    """Compute overall subtask success rate.

    Args:
        results: List of {"subtask": str, "success": bool} dicts.

    Returns:
        Rate in [0, 1].
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)


def switching_nmi(
    predicted_beta: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Normalized Mutual Information between predicted and GT boundaries.

    Args:
        predicted_beta: (T,) -- predicted switching probabilities.
        ground_truth: (T,) -- binary ground truth boundaries.
        threshold: Binarisation threshold for predicted_beta.

    Returns:
        NMI score in [0, 1].
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        return 0.0

    pred_binary = (predicted_beta > threshold).astype(int)
    gt_binary = (ground_truth > 0.5).astype(int)
    return float(normalized_mutual_info_score(gt_binary, pred_binary))


def switching_accuracy(
    predicted_beta: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    tolerance: int = 2,
) -> float:
    """Boundary detection accuracy with temporal tolerance.

    A predicted boundary at time t is considered correct if there
    exists a ground truth boundary within [t-tolerance, t+tolerance].

    Args:
        predicted_beta: (T,) -- predicted switching probabilities.
        ground_truth: (T,) -- binary ground truth boundaries.
        threshold: Binarisation threshold.
        tolerance: Temporal tolerance in timesteps.

    Returns:
        F1 score of boundary detection.
    """
    pred_times = set(np.where(predicted_beta > threshold)[0])
    gt_times = set(np.where(ground_truth > 0.5)[0])

    if not gt_times:
        return 1.0 if not pred_times else 0.0
    if not pred_times:
        return 0.0

    # True positives: predicted boundaries near a GT boundary
    tp = 0
    for pt in pred_times:
        for gt in gt_times:
            if abs(pt - gt) <= tolerance:
                tp += 1
                break

    precision = tp / len(pred_times) if pred_times else 0.0
    recall = tp / len(gt_times) if gt_times else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def temporal_contraction_ratio(
    total_primitive_steps: int,
    num_switches: int,
) -> float:
    """Compute temporal contraction ratio.

    Measures how much the search space is reduced by acting
    at switch points only instead of every timestep.

    Args:
        total_primitive_steps: T (total low-level steps).
        num_switches: M (number of z_t decisions).

    Returns:
        Ratio T/M. Higher = more temporal contraction.
    """
    if num_switches == 0:
        return float("inf")
    return total_primitive_steps / num_switches
