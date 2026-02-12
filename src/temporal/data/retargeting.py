"""Human hand -> canonical end-effector retargeting.

Converts human hand joint representations (e.g. MANO parameters)
to a canonical 7-DoF end-effector representation:
    [x, y, z, qx, qy, qz, qw]  (position + quaternion + gripper)

This enables the action expert to learn from human data
despite kinematic differences.
"""

from __future__ import annotations

import numpy as np


def fingertip_to_ee(
    hand_joints: np.ndarray,
    gripper_threshold: float = 0.04,
) -> np.ndarray:
    """Convert hand joint positions to canonical EE representation.

    Uses the midpoint between thumb tip and index finger tip as
    the end-effector position, and the distance between them as
    the gripper state.

    Args:
        hand_joints: (T, num_joints, 3) -- 3D joint positions.
            Expected joint ordering: [wrist, ..., thumb_tip(4), index_tip(8), ...]
        gripper_threshold: Distance threshold for binary gripper state.

    Returns:
        ee_actions: (T, 7) -- [dx, dy, dz, dqx, dqy, dqz, gripper]
            Delta position, delta orientation (axis-angle), gripper open/close.
    """
    T = hand_joints.shape[0]
    ee_actions = np.zeros((T, 7), dtype=np.float32)

    # Thumb tip and index tip indices (typical MANO ordering)
    thumb_tip_idx = 4
    index_tip_idx = 8

    for t in range(T):
        thumb = hand_joints[t, thumb_tip_idx]
        index = hand_joints[t, index_tip_idx]

        # EE position = midpoint of thumb and index
        ee_pos = (thumb + index) / 2.0

        # Gripper = distance between thumb and index
        grip_dist = np.linalg.norm(thumb - index)
        gripper = 1.0 if grip_dist < gripper_threshold else 0.0

        if t > 0:
            prev_thumb = hand_joints[t - 1, thumb_tip_idx]
            prev_index = hand_joints[t - 1, index_tip_idx]
            prev_pos = (prev_thumb + prev_index) / 2.0
            ee_actions[t, :3] = ee_pos - prev_pos  # delta position
        ee_actions[t, 6] = gripper

    return ee_actions


def mano_to_ee(
    mano_params: np.ndarray,
    gripper_threshold: float = 0.04,
) -> np.ndarray:
    """Convert MANO parameters to canonical EE representation.

    Args:
        mano_params: (T, 51) -- MANO pose (45) + shape (6) params.
        gripper_threshold: Threshold for gripper state.

    Returns:
        ee_actions: (T, 7) -- canonical EE actions.
    """
    # TODO: Implement MANO forward kinematics -> joint positions -> fingertip_to_ee
    raise NotImplementedError("Requires MANO model for forward kinematics")
