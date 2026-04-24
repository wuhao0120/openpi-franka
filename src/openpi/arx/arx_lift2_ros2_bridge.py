"""ARX LIFT2 ROS 2 bridge vendored into openpi-franka."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np

try:  # pragma: no cover - importability is validated indirectly in unit tests.
    import rclpy
    from arm_control.msg import JointControl
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotCmd
    from arx5_arm_msg.msg import RobotStatus
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    _ROS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised via lazy failure paths.
    rclpy = None
    JointControl = PosCmd = RobotCmd = RobotStatus = None
    _ROS_IMPORT_ERROR = exc

    class Node:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _raise_ros_import_error()

    class SingleThreadedExecutor:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _raise_ros_import_error()

    class QoSProfile:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _raise_ros_import_error()

    class ReliabilityPolicy:  # type: ignore[no-redef]
        RELIABLE = None

    class HistoryPolicy:  # type: ignore[no-redef]
        KEEP_LAST = None


logger = logging.getLogger(__name__)


def _raise_ros_import_error():
    raise RuntimeError(
        "ROS2 dependencies are unavailable. The ARX bridge requires `rclpy`, `arm_control`, and `arx5_arm_msg` "
        "to be installed on the robot-control machine."
    ) from _ROS_IMPORT_ERROR


def _require_ros2():
    if _ROS_IMPORT_ERROR is not None:
        _raise_ros_import_error()


@dataclass
class ArmState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_cur: np.ndarray
    end_pos: np.ndarray
    data_received: bool = False


class LiftBridge(Node):
    HEIGHT_SCALE = 41.54

    def __init__(self, node_name: str = "lift_bridge"):
        _require_ros2()
        super().__init__(node_name)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.body_sub = self.create_subscription(PosCmd, "/body_information", self._body_callback, qos_profile)
        self.body_cmd_pub = self.create_publisher(PosCmd, "/ARX_VR_L", qos_profile)

        self._state_lock = threading.Lock()
        self._chassis_height_raw = 0.0
        self._chassis_head_yaw = 0.0
        self._chassis_head_pitch = 0.0
        self._chassis_waist_pos = 0.0
        self._data_received = False

        self._snapshot = {"height": 0.0, "head_yaw": 0.0, "head_pitch": 0.0, "waist": 0.0}

        self._executor = None
        self._spin_thread = None
        self._stop_event = threading.Event()

    def _body_callback(self, msg: PosCmd):
        height_raw = msg.height
        head_yaw = msg.head_yaw
        head_pitch = msg.head_pit
        waist_pos = msg.temp_float_data[0] if len(msg.temp_float_data) > 0 else 0.0

        self._snapshot = {
            "height": float(height_raw / self.HEIGHT_SCALE),
            "head_yaw": float(head_yaw),
            "head_pitch": float(head_pitch),
            "waist": float(waist_pos),
        }

        with self._state_lock:
            self._chassis_height_raw = height_raw
            self._chassis_head_yaw = head_yaw
            self._chassis_head_pitch = head_pitch
            self._chassis_waist_pos = waist_pos
            self._data_received = True

    def start(self):
        if self._spin_thread is not None:
            return
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True, name="LiftBridgeSpin")
        self._spin_thread.start()

    def _spin_loop(self):
        while not self._stop_event.is_set():
            try:
                if rclpy.ok():
                    self._executor.spin_once(timeout_sec=0.01)
                else:
                    break
            except Exception as exc:
                if "RCLError" in str(type(exc)):
                    break
                logger.debug("Lift bridge spin loop error: %s", exc)

    def stop(self):
        if self._spin_thread is None:
            return
        self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        self._executor = None

    def is_connected(self) -> bool:
        return self._data_received

    def get_height(self) -> float:
        with self._state_lock:
            return float(self._chassis_height_raw / self.HEIGHT_SCALE)

    def get_head_yaw(self) -> float:
        with self._state_lock:
            return float(self._chassis_head_yaw)

    def get_head_pitch(self) -> float:
        with self._state_lock:
            return float(self._chassis_head_pitch)

    def get_waist_position(self) -> float:
        with self._state_lock:
            return float(self._chassis_waist_pos)

    def get_snapshot(self) -> dict[str, float]:
        return dict(self._snapshot)

    def set_chassis_cmd(self, vx: float, vy: float, wz: float, mode: int = 2):
        msg = PosCmd()
        msg.chx = vx
        msg.chy = vy
        msg.chz = wz
        msg.mode1 = mode
        snap = self._snapshot
        msg.height = snap["height"] * self.HEIGHT_SCALE
        msg.head_yaw = snap["head_yaw"]
        msg.head_pit = snap["head_pitch"]
        msg.temp_float_data = [snap["waist"]] + [0.0] * 5
        self.body_cmd_pub.publish(msg)

    def set_height(self, height: float):
        msg = PosCmd()
        msg.height = height * self.HEIGHT_SCALE
        snap = self._snapshot
        msg.head_yaw = snap["head_yaw"]
        msg.head_pit = snap["head_pitch"]
        msg.temp_float_data = [snap["waist"]] + [0.0] * 5
        msg.chx = 0.0
        msg.chy = 0.0
        msg.chz = 0.0
        msg.mode1 = 2
        self.body_cmd_pub.publish(msg)

    def set_chassis_full(self, vx: float, vy: float, wz: float, height: float, mode: int = 2):
        msg = PosCmd()
        msg.chx = vx
        msg.chy = vy
        msg.chz = wz
        msg.height = height * self.HEIGHT_SCALE
        msg.mode1 = mode
        snap = self._snapshot
        msg.head_yaw = snap["head_yaw"]
        msg.head_pit = snap["head_pitch"]
        msg.temp_float_data = [snap["waist"]] + [0.0] * 5
        self.body_cmd_pub.publish(msg)


class R5DualArmBridge(Node):
    NUM_ARM_JOINTS = 6
    NUM_GRIPPER_JOINTS = 1
    NUM_TOTAL_JOINTS = 7
    MODE_END_CONTROL = 4
    MODE_POSITION_CONTROL = 5

    def __init__(self, node_name: str = "r5_dual_arm_bridge", control_mode: str = "joint_control_v1"):
        _require_ros2()
        super().__init__(node_name)

        if control_mode not in ("joint_control_v1", "normal"):
            raise ValueError(f"Unsupported control_mode: {control_mode}")

        self.control_mode = control_mode
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_sub = self.create_subscription(
            RobotStatus,
            "/joint_information",
            self._left_arm_callback,
            qos_profile,
        )
        self.right_sub = self.create_subscription(
            RobotStatus,
            "/joint_information2",
            self._right_arm_callback,
            qos_profile,
        )

        if self.control_mode == "joint_control_v1":
            self.left_cmd_pub = self.create_publisher(JointControl, "/joint_control", qos_profile)
            self.right_cmd_pub = self.create_publisher(JointControl, "/joint_control2", qos_profile)
        else:
            self.left_robotcmd_pub = self.create_publisher(RobotCmd, "/arm_cmd_l", qos_profile)
            self.right_robotcmd_pub = self.create_publisher(RobotCmd, "/arm_cmd_r", qos_profile)

        self._state_lock = threading.Lock()
        self.left_state = ArmState(
            joint_pos=np.zeros(self.NUM_TOTAL_JOINTS),
            joint_vel=np.zeros(self.NUM_TOTAL_JOINTS),
            joint_cur=np.zeros(self.NUM_TOTAL_JOINTS),
            end_pos=np.zeros(6),
        )
        self.right_state = ArmState(
            joint_pos=np.zeros(self.NUM_TOTAL_JOINTS),
            joint_vel=np.zeros(self.NUM_TOTAL_JOINTS),
            joint_cur=np.zeros(self.NUM_TOTAL_JOINTS),
            end_pos=np.zeros(6),
        )
        self._left_snapshot = self._zero_arm_snapshot()
        self._right_snapshot = self._zero_arm_snapshot()

        self._executor = None
        self._spin_thread = None
        self._stop_event = threading.Event()

    def _zero_arm_snapshot(self) -> dict[str, Any]:
        return {
            "joint_positions": [0.0] * self.NUM_TOTAL_JOINTS,
            "joint_velocities": [0.0] * self.NUM_TOTAL_JOINTS,
            "joint_currents": [0.0] * self.NUM_TOTAL_JOINTS,
            "end_pose": [0.0] * 6,
            "gripper": 0.0,
        }

    def _build_arm_snapshot(self, msg: RobotStatus) -> dict[str, Any]:
        n = self.NUM_TOTAL_JOINTS
        return {
            "joint_positions": list(msg.joint_pos[:n]),
            "joint_velocities": list(msg.joint_vel[:n]),
            "joint_currents": list(msg.joint_cur[:n]),
            "end_pose": list(msg.end_pos),
            "gripper": float(msg.joint_pos[6] / 5.0) if len(msg.joint_pos) > 6 else 0.0,
        }

    def _left_arm_callback(self, msg: RobotStatus):
        self._left_snapshot = self._build_arm_snapshot(msg)
        with self._state_lock:
            self.left_state.joint_pos = np.asarray(msg.joint_pos[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.left_state.joint_vel = np.asarray(msg.joint_vel[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.left_state.joint_cur = np.asarray(msg.joint_cur[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.left_state.end_pos = np.asarray(msg.end_pos, dtype=np.float32)
            self.left_state.data_received = True

    def _right_arm_callback(self, msg: RobotStatus):
        self._right_snapshot = self._build_arm_snapshot(msg)
        with self._state_lock:
            self.right_state.joint_pos = np.asarray(msg.joint_pos[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.right_state.joint_vel = np.asarray(msg.joint_vel[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.right_state.joint_cur = np.asarray(msg.joint_cur[: self.NUM_TOTAL_JOINTS], dtype=np.float32)
            self.right_state.end_pos = np.asarray(msg.end_pos, dtype=np.float32)
            self.right_state.data_received = True

    def start(self):
        if self._spin_thread is not None:
            return
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True, name="R5DualBridgeSpin")
        self._spin_thread.start()

    def _spin_loop(self):
        while not self._stop_event.is_set():
            try:
                if rclpy.ok():
                    self._executor.spin_once(timeout_sec=0.01)
                else:
                    break
            except Exception as exc:
                if "RCLError" in str(type(exc)):
                    break
                logger.debug("R5 bridge spin loop error: %s", exc)

    def stop(self):
        if self._spin_thread is None:
            return
        self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        self._executor = None

    def is_connected(self) -> bool:
        return self.left_state.data_received and self.right_state.data_received

    def is_left_connected(self) -> bool:
        return self.left_state.data_received

    def is_right_connected(self) -> bool:
        return self.right_state.data_received

    def get_left_joint_positions(self) -> np.ndarray:
        with self._state_lock:
            return self.left_state.joint_pos.copy()

    def get_left_joint_velocities(self) -> np.ndarray:
        with self._state_lock:
            return self.left_state.joint_vel.copy()

    def get_left_joint_currents(self) -> np.ndarray:
        with self._state_lock:
            return self.left_state.joint_cur.copy()

    def get_left_end_pose(self) -> np.ndarray:
        with self._state_lock:
            return self.left_state.end_pos.copy()

    def get_left_gripper_position(self) -> float:
        with self._state_lock:
            return float(self.left_state.joint_pos[6] / 5.0)

    def get_right_joint_positions(self) -> np.ndarray:
        with self._state_lock:
            return self.right_state.joint_pos.copy()

    def get_right_joint_velocities(self) -> np.ndarray:
        with self._state_lock:
            return self.right_state.joint_vel.copy()

    def get_right_joint_currents(self) -> np.ndarray:
        with self._state_lock:
            return self.right_state.joint_cur.copy()

    def get_right_end_pose(self) -> np.ndarray:
        with self._state_lock:
            return self.right_state.end_pos.copy()

    def get_right_gripper_position(self) -> float:
        with self._state_lock:
            return float(self.right_state.joint_pos[6] / 5.0)

    def _publish_joint_robotcmd(self, publisher, positions: np.ndarray):
        msg = RobotCmd()
        msg.end_pos = [0.0] * 6
        msg.joint_pos = [float(v) for v in positions[:6]]
        msg.gripper = float(positions[6])
        msg.mode = self.MODE_POSITION_CONTROL
        publisher.publish(msg)

    def set_left_joint_positions(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        mode: int = 0,
    ):
        if positions.size != self.NUM_TOTAL_JOINTS:
            raise ValueError(f"positions must be {self.NUM_TOTAL_JOINTS}D")
        if self.control_mode == "joint_control_v1":
            msg = JointControl()
            msg.joint_pos = list(np.asarray(positions, dtype=np.float32)) + [0.0]
            if velocities is not None:
                msg.joint_vel = list(np.asarray(velocities, dtype=np.float32)) + [0.0]
            else:
                msg.joint_vel = [0.0] * 8
            msg.joint_cur = [0.0] * 8
            msg.mode = mode
            self.left_cmd_pub.publish(msg)
        else:
            self._publish_joint_robotcmd(self.left_robotcmd_pub, positions)

    def set_right_joint_positions(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        mode: int = 0,
    ):
        if positions.size != self.NUM_TOTAL_JOINTS:
            raise ValueError(f"positions must be {self.NUM_TOTAL_JOINTS}D")
        if self.control_mode == "joint_control_v1":
            msg = JointControl()
            msg.joint_pos = list(np.asarray(positions, dtype=np.float32)) + [0.0]
            if velocities is not None:
                msg.joint_vel = list(np.asarray(velocities, dtype=np.float32)) + [0.0]
            else:
                msg.joint_vel = [0.0] * 8
            msg.joint_cur = [0.0] * 8
            msg.mode = mode
            self.right_cmd_pub.publish(msg)
        else:
            self._publish_joint_robotcmd(self.right_robotcmd_pub, positions)

    def set_left_gripper(self, position: float, mode: int = 0):
        current = self.get_left_joint_positions()
        current[6] = position * 5.0
        self.set_left_joint_positions(current, mode=mode)

    def set_right_gripper(self, position: float, mode: int = 0):
        current = self.get_right_joint_positions()
        current[6] = position * 5.0
        self.set_right_joint_positions(current, mode=mode)

    def set_dual_joint_positions(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        left_velocities: Optional[np.ndarray] = None,
        right_velocities: Optional[np.ndarray] = None,
        mode: int = 0,
    ):
        self.set_left_joint_positions(left_positions, left_velocities, mode)
        self.set_right_joint_positions(right_positions, right_velocities, mode)

    def set_dual_joint_positions_fast(self, left_list: list[float], right_list: list[float], mode: int = 0):
        if self.control_mode == "joint_control_v1":
            left_msg = JointControl()
            left_msg.joint_pos = [float(x) for x in left_list] + [0.0]
            left_msg.joint_vel = [0.0] * 8
            left_msg.joint_cur = [0.0] * 8
            left_msg.mode = mode
            self.left_cmd_pub.publish(left_msg)

            right_msg = JointControl()
            right_msg.joint_pos = [float(x) for x in right_list] + [0.0]
            right_msg.joint_vel = [0.0] * 8
            right_msg.joint_cur = [0.0] * 8
            right_msg.mode = mode
            self.right_cmd_pub.publish(right_msg)
        else:
            self._publish_joint_robotcmd(self.left_robotcmd_pub, np.asarray(left_list, dtype=np.float32))
            self._publish_joint_robotcmd(self.right_robotcmd_pub, np.asarray(right_list, dtype=np.float32))

    def get_dual_snapshot(self) -> dict[str, Any]:
        return {
            "left_arm": dict(self._left_snapshot),
            "right_arm": dict(self._right_snapshot),
        }

    def _require_normal_mode(self):
        if self.control_mode != "normal":
            raise RuntimeError(
                "ee_pose control requires control_mode='normal' and open_double_arm_normal.launch.py"
            )

    def set_left_ee_pose(self, pose: np.ndarray, gripper: float = 0.0):
        self._require_normal_mode()
        msg = RobotCmd()
        msg.end_pos = [float(v) for v in pose[:6]]
        msg.joint_pos = [0.0] * 6
        msg.gripper = float(gripper * 5.0)
        msg.mode = self.MODE_END_CONTROL
        self.left_robotcmd_pub.publish(msg)

    def set_right_ee_pose(self, pose: np.ndarray, gripper: float = 0.0):
        self._require_normal_mode()
        msg = RobotCmd()
        msg.end_pos = [float(v) for v in pose[:6]]
        msg.joint_pos = [0.0] * 6
        msg.gripper = float(gripper * 5.0)
        msg.mode = self.MODE_END_CONTROL
        self.right_robotcmd_pub.publish(msg)

    def set_dual_ee_poses(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ):
        self.set_left_ee_pose(left_pose, left_gripper)
        self.set_right_ee_pose(right_pose, right_gripper)

    def set_dual_ee_poses_fast(
        self,
        left_pose: list[float],
        right_pose: list[float],
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ):
        self._require_normal_mode()

        left_msg = RobotCmd()
        left_msg.end_pos = [float(v) for v in left_pose[:6]]
        left_msg.joint_pos = [0.0] * 6
        left_msg.gripper = float(left_gripper * 5.0)
        left_msg.mode = self.MODE_END_CONTROL
        self.left_robotcmd_pub.publish(left_msg)

        right_msg = RobotCmd()
        right_msg.end_pos = [float(v) for v in right_pose[:6]]
        right_msg.joint_pos = [0.0] * 6
        right_msg.gripper = float(right_gripper * 5.0)
        right_msg.mode = self.MODE_END_CONTROL
        self.right_robotcmd_pub.publish(right_msg)


class ARXLift2Bridge:
    def __init__(
        self,
        lift_node_name: str = "arx_lift_bridge",
        arms_node_name: str = "arx_r5_dual_bridge",
        enable_lift: bool = False,
        enable_arms: bool = True,
        control_mode: str = "joint_control_v1",
    ):
        self.lift_bridge: Optional[LiftBridge] = None
        self.arms_bridge: Optional[R5DualArmBridge] = None
        self._is_connected = False
        self.enable_lift = enable_lift
        self.enable_arms = enable_arms
        self.control_mode = control_mode

        if self.enable_lift:
            self.lift_bridge = LiftBridge(node_name=lift_node_name)
        if self.enable_arms:
            self.arms_bridge = R5DualArmBridge(node_name=arms_node_name, control_mode=control_mode)

    def connect(self, timeout: float = 10.0):
        _require_ros2()
        if not rclpy.ok():
            rclpy.init()

        if self.enable_lift and self.lift_bridge is not None:
            self.lift_bridge.start()
        if self.enable_arms and self.arms_bridge is not None:
            self.arms_bridge.start()

        start_time = time.time()
        while time.time() - start_time < timeout:
            lift_ok = not self.enable_lift or self.lift_bridge.is_connected()
            arms_ok = not self.enable_arms or self.arms_bridge.is_connected()
            if lift_ok and arms_ok:
                self._is_connected = True
                return
            time.sleep(0.5)

        raise RuntimeError("Failed to connect to the ARX LIFT2 system")

    def disconnect(self):
        if self.lift_bridge is not None and self.enable_lift:
            if self.lift_bridge.is_connected():
                self.lift_bridge.set_chassis_cmd(0.0, 0.0, 0.0)
                time.sleep(0.2)
            self.lift_bridge.stop()

        if self.arms_bridge is not None and self.enable_arms:
            if self.arms_bridge.is_connected():
                left_joints = self.arms_bridge.get_left_joint_positions()
                right_joints = self.arms_bridge.get_right_joint_positions()
                self.arms_bridge.set_dual_joint_positions(left_joints, right_joints)
                time.sleep(0.2)
            self.arms_bridge.stop()

        self._is_connected = False

    def is_connected(self) -> bool:
        return self._is_connected

    def get_full_state(self) -> dict[str, Any]:
        state = {}
        if self.enable_lift and self.lift_bridge is not None:
            state["chassis"] = {
                "height": self.lift_bridge.get_height(),
                "head_yaw": self.lift_bridge.get_head_yaw(),
                "head_pitch": self.lift_bridge.get_head_pitch(),
                "waist": self.lift_bridge.get_waist_position(),
            }
        if self.enable_arms and self.arms_bridge is not None:
            state["left_arm"] = {
                "joint_positions": self.arms_bridge.get_left_joint_positions(),
                "joint_velocities": self.arms_bridge.get_left_joint_velocities(),
                "joint_currents": self.arms_bridge.get_left_joint_currents(),
                "end_pose": self.arms_bridge.get_left_end_pose(),
                "gripper": self.arms_bridge.get_left_gripper_position(),
            }
            state["right_arm"] = {
                "joint_positions": self.arms_bridge.get_right_joint_positions(),
                "joint_velocities": self.arms_bridge.get_right_joint_velocities(),
                "joint_currents": self.arms_bridge.get_right_joint_currents(),
                "end_pose": self.arms_bridge.get_right_end_pose(),
                "gripper": self.arms_bridge.get_right_gripper_position(),
            }
        return state

    def get_full_state_serialized(self) -> dict[str, Any]:
        state = {}
        if self.enable_lift and self.lift_bridge is not None:
            state["chassis"] = self.lift_bridge.get_snapshot()
        if self.enable_arms and self.arms_bridge is not None:
            state.update(self.arms_bridge.get_dual_snapshot())
        return state

    def get_chassis_height(self) -> float:
        if self.lift_bridge is None:
            raise RuntimeError("LIFT chassis is not enabled")
        return self.lift_bridge.get_height()

    def set_chassis_height(self, height: float):
        if self.lift_bridge is None:
            raise RuntimeError("LIFT chassis is not enabled")
        self.lift_bridge.set_height(height)

    def set_chassis_velocity(self, vx: float, vy: float, wz: float):
        if self.lift_bridge is None:
            raise RuntimeError("LIFT chassis is not enabled")
        self.lift_bridge.set_chassis_cmd(vx, vy, wz)

    def get_left_joint_positions(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_left_joint_positions()

    def get_right_joint_positions(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_right_joint_positions()

    def get_left_joint_velocities(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_left_joint_velocities()

    def get_left_joint_currents(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_left_joint_currents()

    def get_left_end_pose(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_left_end_pose()

    def get_left_gripper_position(self) -> float:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_left_gripper_position()

    def get_right_joint_velocities(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_right_joint_velocities()

    def get_right_joint_currents(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_right_joint_currents()

    def get_right_end_pose(self) -> np.ndarray:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_right_end_pose()

    def get_right_gripper_position(self) -> float:
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        return self.arms_bridge.get_right_gripper_position()

    def set_left_joint_positions(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, mode: int = 0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_left_joint_positions(positions, velocities, mode)

    def set_right_joint_positions(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, mode: int = 0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_right_joint_positions(positions, velocities, mode)

    def set_left_gripper(self, position: float, mode: int = 0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_left_gripper(position, mode)

    def set_right_gripper(self, position: float, mode: int = 0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_right_gripper(position, mode)

    def set_dual_joint_positions(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        left_velocities: Optional[np.ndarray] = None,
        right_velocities: Optional[np.ndarray] = None,
        mode: int = 0,
    ):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_dual_joint_positions(
            left_positions,
            right_positions,
            left_velocities,
            right_velocities,
            mode,
        )

    def set_left_ee_pose(self, pose: np.ndarray, gripper: float = 0.0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_left_ee_pose(pose, gripper)

    def set_right_ee_pose(self, pose: np.ndarray, gripper: float = 0.0):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_right_ee_pose(pose, gripper)

    def set_dual_ee_poses(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_dual_ee_poses(left_pose, right_pose, left_gripper, right_gripper)

    def set_dual_ee_poses_fast(
        self,
        left_pose: list[float],
        right_pose: list[float],
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ):
        if self.arms_bridge is None:
            raise RuntimeError("Dual arms are not enabled")
        self.arms_bridge.set_dual_ee_poses_fast(left_pose, right_pose, left_gripper, right_gripper)

    def set_full_command(
        self,
        left_positions: np.ndarray | list[float],
        right_positions: np.ndarray | list[float],
        vx: float,
        vy: float,
        wz: float,
        height: float,
    ):
        if self.enable_arms and self.arms_bridge is not None:
            if isinstance(left_positions, list):
                self.arms_bridge.set_dual_joint_positions_fast(  # type: ignore[arg-type]
                    left_positions,
                    right_positions,
                )
            else:
                self.arms_bridge.set_dual_joint_positions(left_positions, right_positions)
        if self.enable_lift and self.lift_bridge is not None:
            self.lift_bridge.set_chassis_full(vx, vy, wz, height)

    def emergency_stop(self):
        if self.enable_lift and self.lift_bridge is not None:
            self.lift_bridge.set_chassis_cmd(0.0, 0.0, 0.0)
        if self.enable_arms and self.arms_bridge is not None and self.arms_bridge.is_connected():
            left_joints = self.arms_bridge.get_left_joint_positions()
            right_joints = self.arms_bridge.get_right_joint_positions()
            self.arms_bridge.set_dual_joint_positions(left_joints, right_joints)
