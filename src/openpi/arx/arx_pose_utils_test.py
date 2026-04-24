import math

import numpy as np

from openpi.arx import arx_pose_utils


def test_apply_delta_pose_adds_translation_and_composes_world_rotvec():
    current_pose = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2], dtype=np.float32)
    delta_pose = np.asarray([0.05, -0.03, 0.02, 0.1, -0.2, 0.3], dtype=np.float32)

    absolute_pose = arx_pose_utils.apply_delta_pose(current_pose, delta_pose)

    np.testing.assert_allclose(absolute_pose[:3], current_pose[:3] + delta_pose[:3], atol=1e-6)
    expected_rot = arx_pose_utils.rotvec_to_matrix(delta_pose[3:]) @ arx_pose_utils.rpy_to_matrix(current_pose[3:])
    np.testing.assert_allclose(arx_pose_utils.rpy_to_matrix(absolute_pose[3:]), expected_rot, atol=1e-6)


def test_delta_action_chunk_to_absolute_commands_clips_gripper_and_slices_horizon():
    state = np.asarray(
        [
            0.10,
            0.20,
            0.30,
            0.00,
            0.00,
            0.00,
            -0.10,
            -0.20,
            0.40,
            0.00,
            0.00,
            0.00,
            0.25,
            0.75,
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, -0.02, 0.00, 0.01, 0.00, 0.00, 0.00, -1.0, 2.0],
            [0.02, 0.00, 0.00, 0.00, 0.00, 0.00, -0.03, 0.00, 0.02, 0.00, 0.00, 0.00, 0.4, 0.6],
            [0.03, 0.00, 0.00, 0.00, 0.00, 0.00, -0.04, 0.00, 0.03, 0.00, 0.00, 0.00, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    commands = arx_pose_utils.delta_action_chunk_to_absolute_commands(state, actions, action_horizon=2)

    assert len(commands) == 2
    np.testing.assert_allclose(commands[0].left_pose[:3], np.asarray([0.11, 0.20, 0.30], dtype=np.float32))
    np.testing.assert_allclose(commands[0].right_pose[:3], np.asarray([-0.12, -0.20, 0.41], dtype=np.float32))
    assert commands[0].left_gripper == 0.0
    assert commands[0].right_gripper == 1.0
    assert np.isclose(commands[1].left_gripper, 0.4)
    assert np.isclose(commands[1].right_gripper, 0.6)
