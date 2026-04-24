from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np


POSE_DIM = 6
STATE_DIM = 14
ACTION_DIM = 14


def _as_vector(values: Sequence[float] | np.ndarray, expected_dim: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (expected_dim,):
        raise ValueError(f"{name} must have shape ({expected_dim},), got {arr.shape}")
    return arr


def rpy_to_matrix(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = _as_vector(rpy, 3, name="rpy")
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    pitch = math.asin(float(np.clip(-mat[2, 0], -1.0, 1.0)))
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    else:
        # Gimbal-lock fallback: keep yaw well-defined and absorb the ambiguity into roll.
        roll = math.atan2(float(-mat[1, 2]), float(mat[1, 1]))
        yaw = 0.0

    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def rotvec_to_matrix(rotvec: Sequence[float] | np.ndarray) -> np.ndarray:
    vec = _as_vector(rotvec, 3, name="rotvec").astype(np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float32)

    axis = vec / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return matrix.astype(np.float32)


def compose_local_rpy(current_rpy: Sequence[float] | np.ndarray, delta_rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    current_rot = rpy_to_matrix(current_rpy)
    delta_rot = rpy_to_matrix(delta_rpy)
    return matrix_to_rpy(current_rot @ delta_rot)


def apply_delta_pose(
    current_pose: Sequence[float] | np.ndarray,
    delta_pose: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply a world-frame delta pose to an absolute pose in `[x, y, z, roll, pitch, yaw]` order.

    ARX teleop records the rotational delta as a rotation vector and applies it in the world frame.
    """
    current = _as_vector(current_pose, POSE_DIM, name="current_pose")
    delta = _as_vector(delta_pose, POSE_DIM, name="delta_pose")

    absolute_pose = np.empty(POSE_DIM, dtype=np.float32)
    absolute_pose[:3] = current[:3] + delta[:3]
    absolute_pose[3:] = matrix_to_rpy(rotvec_to_matrix(delta[3:]) @ rpy_to_matrix(current[3:]))
    return absolute_pose


@dataclasses.dataclass(frozen=True)
class DualArmEECommand:
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper: float
    right_gripper: float


def delta_action_to_absolute_command(
    state: Sequence[float] | np.ndarray,
    action: Sequence[float] | np.ndarray,
) -> DualArmEECommand:
    """Convert a 14D ARX delta-EE action into absolute dual-arm EE targets."""
    state_arr = _as_vector(state, STATE_DIM, name="state")
    action_arr = _as_vector(action, ACTION_DIM, name="action")

    return DualArmEECommand(
        left_pose=apply_delta_pose(state_arr[:6], action_arr[:6]),
        right_pose=apply_delta_pose(state_arr[6:12], action_arr[6:12]),
        left_gripper=float(np.clip(action_arr[12], 0.0, 1.0)),
        right_gripper=float(np.clip(action_arr[13], 0.0, 1.0)),
    )


def delta_action_chunk_to_absolute_commands(
    state: Sequence[float] | np.ndarray,
    actions: Sequence[Sequence[float]] | np.ndarray,
    *,
    action_horizon: int,
) -> list[DualArmEECommand]:
    """Convert the first `action_horizon` actions in a chunk into absolute EE commands.

    OpenPI delta actions are encoded relative to the current observation state, so every action in the
    returned chunk is anchored to the same input state.
    """
    if action_horizon <= 0:
        return []

    action_chunk = np.asarray(actions, dtype=np.float32)
    if action_chunk.ndim != 2 or action_chunk.shape[-1] < ACTION_DIM:
        raise ValueError(f"actions must have shape (T, >= {ACTION_DIM}), got {action_chunk.shape}")

    return [
        delta_action_to_absolute_command(state, action[:ACTION_DIM])
        for action in action_chunk[:action_horizon]
    ]
