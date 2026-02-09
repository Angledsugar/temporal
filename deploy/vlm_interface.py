"""External VLM API interface for subtask generation.

Supports multiple VLM providers (Gemma 3, Gemini, GPT).
The VLM generates ALL subtasks at once given an instruction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLMConfig:
    provider: str = "google"      # google / openai / local
    model: str = "gemini-pro"
    temperature: float = 0.3
    max_subtasks: int = 20


class VLMInterface:
    """Interface to external VLM for subtask decomposition."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._client = None

    def _init_client(self) -> None:
        """Initialise API client based on provider."""
        if self.config.provider == "google":
            # TODO: import google.generativeai
            pass
        elif self.config.provider == "openai":
            # TODO: import openai
            pass

    def decompose(
        self, instruction: str, capability_prompt: str
    ) -> list[str]:
        """Decompose instruction into subtasks (all at once).

        Args:
            instruction: User instruction, e.g. "make a cup of coffee".
            capability_prompt: Capability-aware system prompt from Phase 3.

        Returns:
            subtasks: Ordered list of subtask descriptions.
        """
        system_prompt = (
            "You are a robot task planner. Decompose the given instruction "
            "into an ordered list of subtasks.\n\n"
            f"CAPABILITY CONSTRAINTS:\n{capability_prompt}\n\n"
            "Output ONLY a numbered list. No explanations."
        )

        user_prompt = f"Instruction: {instruction}"

        # TODO: Call actual VLM API
        # response = self._client.generate(system_prompt, user_prompt)
        # return self._parse_subtasks(response)

        # Placeholder
        return [f"subtask_{i}" for i in range(3)]

    def refine(self, original: str, context: str) -> list[str]:
        """Re-plan a failed subtask into finer steps.

        Called when beta_t fires prematurely during execution,
        indicating the subtask exceeds the action expert's capability.

        Args:
            original: The failed subtask description.
            context: Failure context for the VLM.

        Returns:
            refined_subtasks: Finer decomposition of the failed subtask.
        """
        prompt = (
            f"The following subtask could not be completed by the robot:\n"
            f"  \"{original}\"\n\n"
            f"Context: {context}\n\n"
            f"Decompose this into 2-3 simpler steps that the robot can execute."
        )

        # TODO: Call actual VLM API
        return [f"{original}_part_{i}" for i in range(2)]

    @staticmethod
    def _parse_subtasks(response: str) -> list[str]:
        """Parse numbered list from VLM response."""
        lines = response.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering: "1. pick up cup" -> "pick up cup"
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)
        return subtasks
