"""ARX LIFT2 ROS 2 ZeroRPC client vendored into openpi-franka."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import zerorpc
except ImportError:  # pragma: no cover - covered by dependency injection in tests.
    zerorpc = None

log = logging.getLogger(__name__)

NUM_ARM_JOINTS = 7
NUM_POSE_DIMS = 6
_RPC_TIMEOUT_S = 30
_RPC_HEARTBEAT_S = 20


class ArxROS2RPCClient:
    """ARX LIFT2 ROS 2 RPC client."""

    def __init__(
        self,
        ip: str = "localhost",
        port: int = 4242,
        *,
        client: Any | None = None,
        autoconnect: bool = True,
    ):
        self._addr = f"tcp://{ip}:{port}"
        self._client = None
        self.server = None

        if client is not None:
            self._client = client
            self.server = client
            return

        if not autoconnect:
            return

        if zerorpc is None:
            log.error("Failed to connect ZeroRPC client to %s: zerorpc is not installed", self._addr)
            return

        try:
            rpc_client = zerorpc.Client(timeout=_RPC_TIMEOUT_S, heartbeat=_RPC_HEARTBEAT_S)
            rpc_client.connect(self._addr)
            self._client = rpc_client
            self.server = rpc_client
            log.info("ZeroRPC client connected to %s", self._addr)
        except Exception as exc:  # pragma: no cover - depends on runtime transport.
            log.error("Failed to connect ZeroRPC client to %s: %s", self._addr, exc)

    def _call(self, method: str, *args):
        if self._client is None:
            raise RuntimeError("RPC client already closed or not initialized")
        try:
            return self._client(method, *args)
        except Exception as exc:
            raise RuntimeError(f"RPC call failed in {method}: {exc}") from exc

    def system_connect(self, timeout: float = 10.0) -> bool:
        try:
            return bool(self._call("system_connect", timeout))
        except Exception as exc:
            log.error("Error connecting to ARX LIFT2 system: %s", exc)
            return False

    def disconnect(self):
        try:
            self._call("disconnect")
            log.info("Disconnected from ARX LIFT2 system")
        except Exception as exc:
            log.warning("Error disconnecting (continuing): %s", exc)
        finally:
            self.close()

    def is_connected(self) -> bool:
        try:
            return bool(self._call("is_connected"))
        except Exception as exc:
            log.error("Error checking connection status: %s", exc)
            return False

    def get_command_mode(self) -> str | None:
        try:
            return str(self._call("get_command_mode"))
        except Exception as exc:
            log.error("Error getting command mode: %s", exc)
            return None

    def get_full_state(self):
        try:
            state = self._call("get_full_state")
            if state is None:
                return None

            deserialized = {}

            if "chassis" in state:
                deserialized["chassis"] = dict(state["chassis"])

            for side in ("left_arm", "right_arm"):
                if side in state:
                    arm = state[side]
                    deserialized[side] = {
                        "joint_positions": np.asarray(arm["joint_positions"], dtype=np.float32),
                        "joint_velocities": np.asarray(arm["joint_velocities"], dtype=np.float32),
                        "joint_currents": np.asarray(arm["joint_currents"], dtype=np.float32),
                        "end_pose": np.asarray(arm["end_pose"], dtype=np.float32),
                        "gripper": float(arm["gripper"]),
                    }

            return deserialized
        except Exception as exc:
            log.error("Error getting full state: %s", exc)
            return None

    def get_chassis_height(self) -> float:
        try:
            return float(self._call("get_chassis_height"))
        except Exception as exc:
            log.error("Error getting chassis height: %s", exc)
            return 0.0

    def set_chassis_height(self, height: float):
        try:
            self._call("set_chassis_height", float(height))
        except Exception as exc:
            log.error("Error setting chassis height: %s", exc)

    def set_chassis_velocity(self, vx: float, vy: float, wz: float):
        try:
            self._call("set_chassis_velocity", float(vx), float(vy), float(wz))
        except Exception as exc:
            log.error("Error setting chassis velocity: %s", exc)

    def get_left_joint_positions(self):
        try:
            return np.asarray(self._call("get_left_joint_positions"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting left joint positions: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_left_joint_velocities(self):
        try:
            return np.asarray(self._call("get_left_joint_velocities"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting left joint velocities: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_left_joint_currents(self):
        try:
            return np.asarray(self._call("get_left_joint_currents"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting left joint currents: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_left_end_pose(self):
        try:
            return np.asarray(self._call("get_left_end_pose"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting left end pose: %s", exc)
            return np.zeros(NUM_POSE_DIMS, dtype=np.float32)

    def get_left_gripper_position(self) -> float:
        try:
            return float(self._call("get_left_gripper_position"))
        except Exception as exc:
            log.error("Error getting left gripper position: %s", exc)
            return 0.0

    def set_left_joint_positions(self, positions):
        try:
            self._call("set_left_joint_positions", np.asarray(positions, dtype=np.float32).tolist())
        except Exception as exc:
            log.error("Error setting left joint positions: %s", exc)

    def set_left_gripper(self, position: float):
        try:
            self._call("set_left_gripper", float(position))
        except Exception as exc:
            log.error("Error setting left gripper: %s", exc)

    def get_right_joint_positions(self):
        try:
            return np.asarray(self._call("get_right_joint_positions"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting right joint positions: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_right_joint_velocities(self):
        try:
            return np.asarray(self._call("get_right_joint_velocities"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting right joint velocities: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_right_joint_currents(self):
        try:
            return np.asarray(self._call("get_right_joint_currents"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting right joint currents: %s", exc)
            return np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def get_right_end_pose(self):
        try:
            return np.asarray(self._call("get_right_end_pose"), dtype=np.float32)
        except Exception as exc:
            log.error("Error getting right end pose: %s", exc)
            return np.zeros(NUM_POSE_DIMS, dtype=np.float32)

    def get_right_gripper_position(self) -> float:
        try:
            return float(self._call("get_right_gripper_position"))
        except Exception as exc:
            log.error("Error getting right gripper position: %s", exc)
            return 0.0

    def set_right_joint_positions(self, positions):
        try:
            self._call("set_right_joint_positions", np.asarray(positions, dtype=np.float32).tolist())
        except Exception as exc:
            log.error("Error setting right joint positions: %s", exc)

    def set_right_gripper(self, position: float):
        try:
            self._call("set_right_gripper", float(position))
        except Exception as exc:
            log.error("Error setting right gripper: %s", exc)

    def set_dual_joint_positions(self, left_positions, right_positions):
        try:
            self._call(
                "set_dual_joint_positions",
                np.asarray(left_positions, dtype=np.float32).tolist(),
                np.asarray(right_positions, dtype=np.float32).tolist(),
            )
        except Exception as exc:
            log.error("Error setting dual joint positions: %s", exc)

    def set_full_command(self, left_positions, right_positions, vx, vy, wz, height):
        try:
            self._call(
                "set_full_command",
                np.asarray(left_positions, dtype=np.float32).tolist(),
                np.asarray(right_positions, dtype=np.float32).tolist(),
                float(vx),
                float(vy),
                float(wz),
                float(height),
            )
        except Exception as exc:
            log.error("Error setting full command: %s", exc)

    def set_left_ee_pose(self, pose, gripper: float = 0.0):
        try:
            self._call("set_left_ee_pose", np.asarray(pose, dtype=np.float32).tolist(), float(gripper))
        except Exception as exc:
            log.error("Error setting left ee pose: %s", exc)

    def set_right_ee_pose(self, pose, gripper: float = 0.0):
        try:
            self._call("set_right_ee_pose", np.asarray(pose, dtype=np.float32).tolist(), float(gripper))
        except Exception as exc:
            log.error("Error setting right ee pose: %s", exc)

    def set_dual_ee_poses(self, left_pose, right_pose, left_gripper: float = 0.0, right_gripper: float = 0.0):
        try:
            return self._call(
                "set_dual_ee_poses",
                np.asarray(left_pose, dtype=np.float32).tolist(),
                np.asarray(right_pose, dtype=np.float32).tolist(),
                float(left_gripper),
                float(right_gripper),
            )
        except Exception as exc:
            log.error("Error setting dual ee poses: %s", exc)
            return {
                "accepted": False,
                "executed": False,
                "sequence_id": -1,
                "left_pose": np.asarray(left_pose, dtype=np.float32).tolist(),
                "right_pose": np.asarray(right_pose, dtype=np.float32).tolist(),
                "left_gripper": float(left_gripper),
                "right_gripper": float(right_gripper),
                "message": str(exc),
            }

    def emergency_stop(self):
        try:
            self._call("emergency_stop")
        except Exception as exc:
            log.error("Error during emergency stop: %s", exc)

    def close(self):
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if close is not None:
                    close()
            except Exception as exc:
                log.error("Error closing RPC client: %s", exc)
            self._client = None
            self.server = None
