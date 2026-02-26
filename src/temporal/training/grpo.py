"""GRPO: Group Relative Policy Optimization (Appendix C.5.2).

PPO variant with relative advantage estimation instead of a learned
value function (critic). Used for Internal RL in abstract action space.

Objective (Eq. 10):
    E[min(r_t * A_τ, clip(r_t, 1-ε, 1+ε) * A_τ)]

Relative advantage:
    R̄ = mean(R(τ_i))
    σ_R = std(R(τ_i))
    A_τ = (R(τ) - R̄) / (σ_R + δ)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_relative_advantages(
    rewards: torch.Tensor, delta: float = 1e-3
) -> torch.Tensor:
    """Compute relative advantages for a batch of trajectories.

    Args:
        rewards: (B,) total reward per trajectory.
        delta: Small constant for numerical stability.

    Returns:
        advantages: (B,) relative advantages.
    """
    mean_r = rewards.mean()
    std_r = rewards.std()
    return (rewards - mean_r) / (std_r + delta)


def compute_grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.0,
    entropies: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO clipped surrogate loss.

    Args:
        log_probs_new: (B,) log-probs under current policy.
        log_probs_old: (B,) log-probs under old policy (detached).
        advantages: (B,) relative advantages.
        clip_epsilon: PPO clipping range.
        entropy_coeff: Entropy bonus coefficient.
        entropies: (B,) optional entropy values.

    Returns:
        loss: Scalar loss (to minimize).
        metrics: Dictionary with loss components.
    """
    # Importance ratio
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Entropy bonus
    entropy_loss = torch.tensor(0.0, device=policy_loss.device)
    if entropy_coeff > 0 and entropies is not None:
        entropy_loss = -entropy_coeff * entropies.mean()

    total_loss = policy_loss + entropy_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantage_mean": advantages.mean().item(),
    }
    return total_loss, metrics
