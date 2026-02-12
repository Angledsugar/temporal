"""Full TempoRAL deployment pipeline.

VLM -> [subtask_1, ..., subtask_N] -> Action Expert + MetaController

Integrates:
    - Capability-aware VLM prompting (learned prior)
    - Runtime beta_t monitoring (adaptation)
    - Re-planning on failure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from temporal.deploy.capability_prompt import build_capability_prompt
from temporal.deploy.replanner import Replanner, ReplanConfig
from temporal.deploy.vlm_interface import VLMInterface, VLMConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    beta_threshold: float = 0.5
    max_steps_per_subtask: int = 200
    replan_on_failure: bool = True
    max_replan_attempts: int = 3


class TempoRALPipeline:
    """Full deployment pipeline: VLM + Action Expert + MetaController."""

    def __init__(
        self,
        vlm: VLMInterface,
        action_expert: Any,
        metacontroller: Any,
        rl_policy: Any,
        config: PipelineConfig,
    ):
        self.vlm = vlm
        self.expert = action_expert
        self.meta = metacontroller
        self.policy = rl_policy
        self.config = config
        self.replanner = Replanner(
            vlm, ReplanConfig(max_replan_attempts=config.max_replan_attempts)
        )
        self.capability_prompt = build_capability_prompt()

    def execute(self, instruction: str, env: Any) -> dict[str, Any]:
        """Execute a full instruction.

        Args:
            instruction: Natural language instruction.
            env: Physical or simulated environment.

        Returns:
            result: Execution summary with success status.
        """
        # 1. VLM generates all subtasks at once
        subtasks = self.vlm.decompose(instruction, self.capability_prompt)
        logger.info(f"Instruction: '{instruction}' -> {len(subtasks)} subtasks")

        results = []

        for i, subtask in enumerate(subtasks):
            logger.info(f"Executing subtask {i+1}/{len(subtasks)}: '{subtask}'")
            success = self._execute_subtask(subtask, env)

            if not success and self.config.replan_on_failure:
                # Runtime re-planning
                refined = self.replanner.replan(subtask)
                for j, sub in enumerate(refined):
                    logger.info(f"  Re-plan {j+1}/{len(refined)}: '{sub}'")
                    sub_success = self._execute_subtask(sub, env)
                    results.append({"subtask": sub, "success": sub_success})
            else:
                results.append({"subtask": subtask, "success": success})

        total_success = all(r["success"] for r in results)
        return {
            "instruction": instruction,
            "subtasks": results,
            "success": total_success,
        }

    def _execute_subtask(self, subtask: str, env: Any) -> bool:
        """Execute a single subtask with beta_t monitoring.

        Args:
            subtask: Subtask description.
            env: Environment.

        Returns:
            True if subtask completed successfully.
        """
        # TODO: Implement with actual models
        # obs = env.get_observation()
        # e_t = self.expert.extract_residual_stream(obs, subtask)
        # h_prev = None
        #
        # for step in range(self.config.max_steps_per_subtask):
        #     z_t, _, _, _, h_prev = self.policy.sample(e_t, h_prev)
        #     e_controlled = self.meta.decoder.apply_control(e_t, z_t)
        #     action = self.expert.decode_action(e_controlled)
        #     obs, reward, done, _, info = env.step(action)
        #     e_t = self.expert.extract_residual_stream(obs, subtask)
        #
        #     beta_t = self.meta.compute_beta(e_t, z_t)
        #     if beta_t > self.config.beta_threshold:
        #         return True
        #     if done:
        #         break
        #
        # return False
        return False
