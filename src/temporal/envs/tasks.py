"""Task definitions for Gridworld-Pinpad (Table A1 in paper).

Each task is a sequence of color indices [c0, c1, ..., cL] that the agent
must visit in order. Colors are numbered 0-7.

Abstract subgoals are pairs: 0-1, 2-3, 4-5, 6-7.
"""

# Pretraining tasks (Table A1, page 12)
# 18 tasks of length 5-6 using abstract subgoal pairs
PRETRAINING_TASKS: list[list[int]] = [
    [0, 1, 4, 5, 0, 1],
    [0, 1, 4, 5, 2, 3],
    [0, 1, 6, 7, 2, 3],
    [2, 3, 0, 1, 4, 5],
    [2, 3, 6, 7, 2, 3],
    [2, 3, 6, 7, 4, 5],
    [4, 5, 0, 1, 4, 5],
    [4, 5, 0, 1, 6, 7],
    [4, 5, 2, 3, 6, 7],
    [6, 7, 2, 3, 0, 1],
    [6, 7, 2, 3, 6, 7],
    [6, 7, 4, 5, 0, 1],
    [0, 1, 6, 7, 4, 5],
    [2, 3, 0, 1, 6, 7],
    [4, 5, 2, 3, 0, 1],
    [6, 7, 4, 5, 2, 3],
]

# Post-training task (page 12): longer compositional task not seen during pretraining
# 0-1-2-3-4-5-6-7-0-1-2-3
POST_TRAINING_TASK: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]

# Environment constants (Section A.1.2)
GRID_SIZE = 7       # G
NUM_COLORS = 8      # O
NUM_WALLS = 4       # W
MAX_STEPS = 100     # T
NUM_ACTIONS = 4     # 4 cardinal directions

# Observation dimension: G^2 * (W + O + 1) = 49 * 13 = 637
OBS_DIM = GRID_SIZE * GRID_SIZE * (NUM_WALLS + NUM_COLORS + 1)

# Abstract subgoal pairs (for analysis/verification)
SUBGOAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
NUM_SUBGOALS = len(SUBGOAL_PAIRS)  # 4 abstract subgoals
