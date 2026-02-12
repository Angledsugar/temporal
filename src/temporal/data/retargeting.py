"""Human hand -> canonical end-effector retargeting.

Converts human hand joint representations (e.g. MANO parameters,
Inter-X OptiTrack skeletons) to a canonical 7-DoF end-effector
representation:
    [x, y, z, qx, qy, qz, gripper]  (position + quaternion + gripper)

This enables the action expert to learn from human data
despite kinematic differences.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Inter-X OptiTrack 64-joint skeleton layout
# ---------------------------------------------------------------------------
# Pelvis/Legs:  0(pelvis), 1-5(right leg), 6-10(left leg)
# Spine/Head:   11(spine), 12(chest), 13-14(head)
# Right arm:    15(shoulder), 16(elbow)
# Right hand:   17(wrist), 18-20(thumb ×3), 21-24(index ×4),
#               25-28(middle ×4), 29-32(ring ×4), 33-36(pinky ×4)
# Left arm:     37-38(shoulder), 39(elbow)
# Left hand:    40(wrist), 41-44(thumb ×4), 45-48(index ×4),
#               49-52(middle ×4), 53-56(ring ×4), 57-60(pinky ×4)
# Face:         61-63
# ---------------------------------------------------------------------------

# Right hand
INTERX_R_SHOULDER = 15
INTERX_R_ELBOW = 16
INTERX_R_WRIST = 17
INTERX_R_THUMB_TIP = 20
INTERX_R_INDEX_TIP = 24
INTERX_R_MIDDLE_TIP = 28

# Left hand
INTERX_L_SHOULDER = 38
INTERX_L_ELBOW = 39
INTERX_L_WRIST = 40
INTERX_L_THUMB_TIP = 44
INTERX_L_INDEX_TIP = 48
INTERX_L_MIDDLE_TIP = 52

# Body landmarks
INTERX_PELVIS = 0
INTERX_SPINE = 11
INTERX_CHEST = 12


def _rotation_matrix_from_axes(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation matrix from forward and up vectors (right-hand rule)."""
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)
    return np.stack([forward, right, up], axis=-1)  # (3, 3)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def interx_skeleton_to_ee(
    skeleton: np.ndarray,
    side: str = "right",
    gripper_threshold: float = 0.04,
) -> np.ndarray:
    """Convert Inter-X 64-joint skeleton to canonical 7-DoF EE.

    Args:
        skeleton: (T, 64, 3) -- 3D joint positions in meters, Y-up.
        side: "right" or "left" hand.
        gripper_threshold: Distance (m) below which gripper is closed.

    Returns:
        ee: (T, 7) float32 -- [x, y, z, qx, qy, qz, gripper]
            Absolute wrist position, wrist orientation quaternion, gripper state.
    """
    T = skeleton.shape[0]
    ee = np.zeros((T, 7), dtype=np.float32)

    if side == "right":
        wrist_idx, elbow_idx = INTERX_R_WRIST, INTERX_R_ELBOW
        thumb_tip, index_tip, middle_tip = INTERX_R_THUMB_TIP, INTERX_R_INDEX_TIP, INTERX_R_MIDDLE_TIP
    else:
        wrist_idx, elbow_idx = INTERX_L_WRIST, INTERX_L_ELBOW
        thumb_tip, index_tip, middle_tip = INTERX_L_THUMB_TIP, INTERX_L_INDEX_TIP, INTERX_L_MIDDLE_TIP

    for t in range(T):
        wrist = skeleton[t, wrist_idx]
        elbow = skeleton[t, elbow_idx]
        thumb = skeleton[t, thumb_tip]
        index = skeleton[t, index_tip]
        middle = skeleton[t, middle_tip]

        # Position: wrist xyz
        ee[t, :3] = wrist

        # Orientation: build local frame from forearm direction + finger direction
        forearm = wrist - elbow  # forearm axis
        finger_dir = middle - wrist  # wrist → middle finger
        R = _rotation_matrix_from_axes(finger_dir, forearm)
        ee[t, 3:7] = _rotmat_to_quat(R)

    # Gripper: thumb-index distance
    thumb_pos = skeleton[:, thumb_tip]  # (T, 3)
    index_pos = skeleton[:, index_tip]  # (T, 3)
    grip_dist = np.linalg.norm(thumb_pos - index_pos, axis=1)  # (T,)
    ee[:, 6] = (grip_dist < gripper_threshold).astype(np.float32)

    return ee


def interx_skeleton_to_proprio(
    skeleton: np.ndarray,
    side: str = "right",
    max_grip_dist: float = 0.15,
) -> np.ndarray:
    """Extract 14-D proprioception from Inter-X skeleton.

    Proprioception layout (matching DROID convention):
        [0-6]:  Joint angles -- shoulder(3) + elbow(1) + wrist(3)
        [7-12]: Cartesian state -- wrist xyz(3) + wrist orientation euler(3)
        [13]:   Gripper opening (normalized thumb-index distance)

    Args:
        skeleton: (T, 64, 3) -- 3D joint positions.
        side: "right" or "left".
        max_grip_dist: Max thumb-index distance for normalization.

    Returns:
        proprio: (T, 14) float32
    """
    T = skeleton.shape[0]
    proprio = np.zeros((T, 14), dtype=np.float32)

    if side == "right":
        shoulder, elbow, wrist = INTERX_R_SHOULDER, INTERX_R_ELBOW, INTERX_R_WRIST
        thumb_tip, index_tip, middle_tip = INTERX_R_THUMB_TIP, INTERX_R_INDEX_TIP, INTERX_R_MIDDLE_TIP
    else:
        shoulder, elbow, wrist = INTERX_L_SHOULDER, INTERX_L_ELBOW, INTERX_L_WRIST
        thumb_tip, index_tip, middle_tip = INTERX_L_THUMB_TIP, INTERX_L_INDEX_TIP, INTERX_L_MIDDLE_TIP

    for t in range(T):
        s = skeleton[t, shoulder]
        e = skeleton[t, elbow]
        w = skeleton[t, wrist]
        m = skeleton[t, middle_tip]

        upper_arm = e - s
        forearm = w - e
        hand_dir = m - w

        # Joint angles (approximate from bone vectors)
        # Shoulder: 3 angles (elevation, azimuth, twist approximated via decomposition)
        spine_up = np.array([0.0, 1.0, 0.0])  # Y-up
        proprio[t, 0] = _angle_between(upper_arm, spine_up)  # elevation
        proprio[t, 1] = np.arctan2(upper_arm[0], upper_arm[2])  # azimuth in XZ
        proprio[t, 2] = np.arctan2(upper_arm[0], upper_arm[1])  # twist approx

        # Elbow: 1 angle (flexion)
        proprio[t, 3] = _angle_between(upper_arm, forearm)

        # Wrist: 3 angles (relative to forearm)
        proprio[t, 4] = _angle_between(forearm, hand_dir)
        wrist_cross = np.cross(forearm, hand_dir)
        proprio[t, 5] = np.arctan2(wrist_cross[1], wrist_cross[0]) if np.linalg.norm(wrist_cross) > 1e-8 else 0.0
        proprio[t, 6] = np.arctan2(wrist_cross[2], wrist_cross[0]) if np.linalg.norm(wrist_cross) > 1e-8 else 0.0

        # Cartesian: wrist position
        proprio[t, 7:10] = w

        # Cartesian: wrist orientation (euler from local frame)
        R = _rotation_matrix_from_axes(hand_dir, forearm)
        # Extract euler angles (ZYX convention)
        proprio[t, 10] = np.arctan2(R[2, 1], R[2, 2])   # roll
        proprio[t, 11] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))  # pitch
        proprio[t, 12] = np.arctan2(R[1, 0], R[0, 0])   # yaw

    # Gripper: normalized thumb-index distance
    thumb_pos = skeleton[:, thumb_tip]
    index_pos = skeleton[:, index_tip]
    grip_dist = np.linalg.norm(thumb_pos - index_pos, axis=1)
    proprio[:, 13] = np.clip(grip_dist / max_grip_dist, 0.0, 1.0)

    return proprio


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
