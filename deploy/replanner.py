"""Runtime re-planning on beta_t trigger.

Monitors beta_t during execution and triggers VLM re-planning
when the action expert detects it cannot complete a subtask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from temporal.deploy.vlm_interface import VLMInterface

logger = logging.getLogger(__name__)


@dataclass
class ReplanConfig:
    max_replan_attempts: int = 3
    beta_threshold: float = 0.5
    early_fire_ratio: float = 0.3  # beta fires before 30% of expected duration


class Replanner:
    """Monitors execution and triggers re-planning on failure."""

    def __init__(self, vlm: VLMInterface, config: ReplanConfig):
        self.vlm = vlm
        self.config = config

    def should_replan(
        self,
        beta_t: float,
        elapsed_steps: int,
        expected_steps: int,
    ) -> bool:
        """Determine if re-planning is needed.

        Re-plan if beta_t fires prematurely (before expected completion).

        Args:
            beta_t: Current switching probability.
            elapsed_steps: Steps executed so far for this subtask.
            expected_steps: Expected steps for this subtask type.

        Returns:
            True if re-planning should be triggered.
        """
        if beta_t <= self.config.beta_threshold:
            return False

        # beta fired -- check if it's premature
        progress = elapsed_steps / max(expected_steps, 1)
        if progress < self.config.early_fire_ratio:
            logger.info(
                f"Early beta fire at {progress:.0%} progress "
                f"(beta={beta_t:.2f}). Triggering re-plan."
            )
            return True

        return False

    def replan(self, failed_subtask: str, attempt: int = 0) -> list[str]:
        """Request VLM to decompose failed subtask into finer steps.

        Args:
            failed_subtask: Description of the failed subtask.
            attempt: Current re-plan attempt number.

        Returns:
            refined: List of finer subtask descriptions.
        """
        if attempt >= self.config.max_replan_attempts:
            logger.warning(
                f"Max replan attempts ({self.config.max_replan_attempts}) "
                f"reached for '{failed_subtask}'. Skipping."
            )
            return []

        context = (
            f"Attempt {attempt + 1}/{self.config.max_replan_attempts}. "
            f"The action expert's internal boundary detector fired prematurely, "
            f"indicating this subtask exceeds its single-chunk capability."
        )
        return self.vlm.refine(failed_subtask, context)
