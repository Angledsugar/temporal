"""Generates capability-aware system prompt for the VLM.

Distills the learned capability profile from Phase 3 into
natural language constraints that guide the VLM to generate
subtasks at the right granularity.
"""

from __future__ import annotations


def build_capability_prompt(
    avg_subtask_duration: float = 3.0,
    max_objects_per_subtask: int = 1,
    supported_primitives: list[str] | None = None,
) -> str:
    """Build capability-aware prompt from learned profile.

    Args:
        avg_subtask_duration: Average duration (seconds) of a successful subtask.
        max_objects_per_subtask: Max objects the expert can handle per subtask.
        supported_primitives: List of manipulation primitives the expert can do.

    Returns:
        prompt: Natural language capability description.
    """
    if supported_primitives is None:
        supported_primitives = [
            "reach", "grasp", "release", "lift", "place",
            "push", "pull", "rotate (<=90 degrees)",
        ]

    primitives_str = ", ".join(supported_primitives)

    return (
        f"Each subtask should describe a single contact-phase manipulation "
        f"primitive completable in {avg_subtask_duration:.0f} seconds or less.\n"
        f"Supported primitives: {primitives_str}.\n"
        f"Each subtask should involve at most {max_objects_per_subtask} object(s).\n"
        f"Examples of GOOD subtasks: 'grasp the cup handle', 'place cup on saucer'.\n"
        f"Examples of BAD subtasks: 'make coffee' (too abstract), "
        f"'rotate wrist 15 degrees' (too fine).\n"
        f"Do NOT combine multiple contact phases into one subtask."
    )
