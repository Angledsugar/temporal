"""Phase 3: Internal RL for Subtask Granularity Optimisation.

Trains a causal RL policy in the abstract controller-code space.
The action expert and MetaController decoder are FROZEN.
Only the causal GRU policy (psi) is trained.

Policy gradient with relative advantage estimation:
    grad J = E[ sum_m A_m * grad ln pi_psi(z_{t_m} | e_{1:t_m}) ]

where {t_1, ..., t_M} are switch points (M << T).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from temporal.envs.internal_env import InternalEnv
from temporal.models.internal_rl_policy import CausalRLPolicy

logger = logging.getLogger(__name__)


@dataclass
class Phase3Config:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_episodes: int = 100_000
    max_steps_per_episode: int = 200
    log_every: int = 50
    eval_every: int = 1_000
    eval_episodes: int = 100
    save_every: int = 5_000
    output_dir: str = "checkpoints/phase3/"


class Phase3Trainer:
    """Internal RL training loop using policy gradient."""

    def __init__(
        self,
        policy: CausalRLPolicy,
        env: InternalEnv,
        config: Phase3Config,
    ):
        self.policy = policy
        self.env = env
        self.config = config
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )

    def collect_episode(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[torch.Tensor]]:
        """Collect one episode of experience.

        Returns:
            states:   list of (n_e,) tensors -- residual stream states.
            actions:  list of (n_z,) tensors -- controller codes.
            rewards:  list of floats         -- per-switch rewards.
            log_probs: list of scalar tensors -- log pi(z_t | e_{1:t}).
        """
        e_t, _ = self.env.reset()
        e_t = torch.tensor(e_t, dtype=torch.float32).unsqueeze(0)  # (1, n_e)
        h_prev = None

        states, actions, rewards, log_probs = [], [], [], []

        for _ in range(self.config.max_steps_per_episode):
            z_t, mu, std, value, h_prev = self.policy.sample(e_t, h_prev)
            lp = self.policy.log_prob(z_t, mu, std)

            # Step internal environment
            e_next, reward, terminated, truncated, info = self.env.step(
                z_t.squeeze(0).detach().numpy()
            )

            states.append(e_t.squeeze(0))
            actions.append(z_t.squeeze(0))
            rewards.append(reward)
            log_probs.append(lp.squeeze(0))

            if terminated or truncated:
                break

            e_t = torch.tensor(e_next, dtype=torch.float32).unsqueeze(0)

        return states, actions, rewards, log_probs

    def compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns with relative advantage (baseline subtracted).

        Args:
            rewards: Per-switch rewards.

        Returns:
            advantages: Baseline-subtracted returns.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.config.gamma * G
            returns.insert(0, G)

        # Relative advantage: subtract mean (simple baseline)
        mean_return = sum(returns) / max(len(returns), 1)
        advantages = [G - mean_return for G in returns]
        return advantages

    def train(self) -> None:
        """Main training loop."""
        self.policy.train()

        for episode in range(self.config.max_episodes):
            states, actions, rewards, log_probs = self.collect_episode()

            if not rewards:
                continue

            advantages = self.compute_returns(rewards)

            # Policy gradient loss
            # grad J = sum_m A_m * grad ln pi(z_{t_m} | e_{1:t_m})
            policy_loss = torch.tensor(0.0)
            for lp, adv in zip(log_probs, advantages):
                policy_loss = policy_loss - lp * adv

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Logging
            if episode % self.config.log_every == 0:
                total_reward = sum(rewards)
                num_switches = len(rewards)
                success = any(r > 0.5 for r in rewards)
                logger.info(
                    f"Episode {episode}: reward={total_reward:.2f} "
                    f"switches={num_switches} success={success}"
                )

            if episode % self.config.save_every == 0 and episode > 0:
                self._save_checkpoint(episode)

        self._save_checkpoint(self.config.max_episodes)
        logger.info("Phase 3 training complete.")

    def _save_checkpoint(self, episode: int) -> None:
        path = f"{self.config.output_dir}/rl_policy_ep{episode}.pt"
        torch.save(self.policy.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")
