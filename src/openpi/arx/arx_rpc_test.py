import numpy as np

from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient
from openpi.arx.arx_ros2_rpc_server import ArxROS2RPCServer


class _FakeRPCClient:
    def __init__(self, responses=None, raise_methods=None):
        self.responses = responses or {}
        self.raise_methods = set(raise_methods or ())
        self.calls = []
        self.closed = False

    def __call__(self, method, *args):
        self.calls.append((method, args))
        if method in self.raise_methods:
            raise RuntimeError(f"boom in {method}")
        return self.responses.get(method)

    def close(self):
        self.closed = True


def test_rpc_client_deserializes_full_state_and_preserves_signature():
    fake_client = _FakeRPCClient(
        responses={
            "get_full_state": {
                "left_arm": {
                    "joint_positions": [1.0] * 7,
                    "joint_velocities": [2.0] * 7,
                    "joint_currents": [3.0] * 7,
                    "end_pose": [4.0] * 6,
                    "gripper": 0.25,
                },
                "right_arm": {
                    "joint_positions": [5.0] * 7,
                    "joint_velocities": [6.0] * 7,
                    "joint_currents": [7.0] * 7,
                    "end_pose": [8.0] * 6,
                    "gripper": 0.75,
                },
            },
            "set_dual_ee_poses": {"accepted": True, "executed": False, "sequence_id": 1},
        }
    )
    client = ArxROS2RPCClient(client=fake_client)

    state = client.get_full_state()
    ack = client.set_dual_ee_poses(np.arange(6), np.arange(6) + 10, 0.1, 0.2)

    assert state is not None
    assert ack == {"accepted": True, "executed": False, "sequence_id": 1}
    np.testing.assert_array_equal(state["left_arm"]["joint_positions"], np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(state["right_arm"]["end_pose"], np.full(6, 8.0, dtype=np.float32))
    assert fake_client.calls[-1] == (
        "set_dual_ee_poses",
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 0.1, 0.2),
    )


def test_rpc_client_handles_transport_errors():
    client = ArxROS2RPCClient(client=_FakeRPCClient(raise_methods={"get_full_state"}))
    assert client.get_full_state() is None


class _DummyBridge:
    def __init__(self):
        self.calls = []

    def is_connected(self):
        return True

    def set_dual_ee_poses_fast(self, left_pose, right_pose, left_gripper, right_gripper):
        self.calls.append(("set_dual_ee_poses_fast", left_pose, right_pose, left_gripper, right_gripper))

    def get_full_state_serialized(self):
        self.calls.append(("get_full_state_serialized",))
        return {"left_arm": {"end_pose": [0.0] * 6, "gripper": 0.0}}


def test_rpc_server_silent_mode_acknowledges_without_forwarding():
    server = ArxROS2RPCServer(command_mode="silent", start_workers=False)
    bridge = _DummyBridge()
    server.bridge = bridge

    ack = server.set_dual_ee_poses([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], 0.3, 0.4)
    state = server.get_full_state()

    assert ack["accepted"] is True
    assert ack["executed"] is False
    assert ack["left_pose"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert bridge.calls == [("get_full_state_serialized",)]
    assert state == {"left_arm": {"end_pose": [0.0] * 6, "gripper": 0.0}}


def test_rpc_server_execute_mode_forwards_dual_ee_commands_and_state_reads():
    server = ArxROS2RPCServer(command_mode="execute", start_workers=False)
    bridge = _DummyBridge()
    server.bridge = bridge

    ack = server.set_dual_ee_poses([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], 0.3, 0.4)
    state = server.get_full_state()

    assert ack["accepted"] is True
    assert ack["executed"] is True
    assert bridge.calls[0] == (
        "set_dual_ee_poses_fast",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        0.3,
        0.4,
    )
    assert bridge.calls[1] == ("get_full_state_serialized",)
    assert state == {"left_arm": {"end_pose": [0.0] * 6, "gripper": 0.0}}
