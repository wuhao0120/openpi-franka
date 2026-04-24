from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client import image_tools

from openpi.arx.arx_pose_utils import DualArmEECommand
from openpi.arx.arx_pose_utils import delta_action_chunk_to_absolute_commands
from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient

log = logging.getLogger(__name__)

CAMERA_KEYS = ("head_image", "left_wrist_image", "right_wrist_image")
CAMERA_ENV_MAP = {
    "head_image": "OPENPI_ARX_HEAD_CAMERA",
    "left_wrist_image": "OPENPI_ARX_LEFT_WRIST_CAMERA",
    "right_wrist_image": "OPENPI_ARX_RIGHT_WRIST_CAMERA",
}


@dataclasses.dataclass(frozen=True)
class DualArmCommandResult:
    command: DualArmEECommand
    ack: dict[str, Any]


def state_from_full_state(full_state: Mapping[str, Any]) -> np.ndarray:
    """Convert an ARX RPC state dict into the 14D OpenPI ARX state vector."""
    left_arm = full_state["left_arm"]
    right_arm = full_state["right_arm"]
    return np.asarray(
        [
            *np.asarray(left_arm["end_pose"], dtype=np.float32).tolist(),
            *np.asarray(right_arm["end_pose"], dtype=np.float32).tolist(),
            float(left_arm["gripper"]),
            float(right_arm["gripper"]),
        ],
        dtype=np.float32,
    )


def make_arx_observation(
    state: np.ndarray,
    images: Mapping[str, np.ndarray],
    prompt: str,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Pack ARX state and 3 RGB streams into the OpenPI policy observation schema."""
    return {
        "observation/state": np.asarray(state, dtype=np.float32),
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images["head_image"], height, width)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images["left_wrist_image"], height, width)
        ),
        "observation/right_wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images["right_wrist_image"], height, width)
        ),
        "prompt": prompt,
    }


class DummyCameraRig:
    def __init__(self, *, width: int = 640, height: int = 480):
        self._width = width
        self._height = height

    def connect(self) -> None:
        log.info("[MOCK] Using dummy ARX camera rig")

    def disconnect(self) -> None:
        log.info("[MOCK] Dummy ARX camera rig disconnected")

    def read(self) -> dict[str, np.ndarray]:
        image = np.random.randint(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
        return {key: image.copy() for key in CAMERA_KEYS}


class RealSenseCameraRig:
    """Three-camera local RealSense rig for ARX inference."""

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serials: Mapping[str, str] | None = None,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._serials = self._resolve_serials(serials)
        self._cameras = None

    def _resolve_serials(self, serials: Mapping[str, str] | None) -> dict[str, str]:
        if serials is not None:
            resolved = {key: serials[key] for key in CAMERA_KEYS}
        else:
            resolved = {key: os.environ.get(env_var, "") for key, env_var in CAMERA_ENV_MAP.items()}

        missing = [key for key, value in resolved.items() if not value]
        if missing:
            missing_env = ", ".join(CAMERA_ENV_MAP[key] for key in missing)
            raise RuntimeError(
                "Missing ARX camera serials. Set the corresponding environment variables: "
                f"{missing_env}"
            )
        return resolved

    def _import_camera_stack(self):
        try:
            from lerobot.cameras import make_cameras_from_configs
            from lerobot.cameras.configs import ColorMode
            from lerobot.cameras.configs import Cv2Rotation
            try:
                from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
            except ImportError:
                from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
        except ImportError as exc:  # pragma: no cover - depends on runtime environment.
            raise RuntimeError(
                "LeRobot RealSense camera stack is not available. Install `lerobot` on the inference machine."
            ) from exc

        return RealSenseCameraConfig, ColorMode, Cv2Rotation, make_cameras_from_configs

    def connect(self) -> None:
        RealSenseCameraConfig, ColorMode, Cv2Rotation, make_cameras_from_configs = self._import_camera_stack()
        common = dict(
            fps=self._fps,
            width=self._width,
            height=self._height,
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.NO_ROTATION,
        )
        camera_config = {
            key: RealSenseCameraConfig(serial_number_or_name=self._serials[key], **common) for key in CAMERA_KEYS
        }
        self._cameras = make_cameras_from_configs(camera_config)
        for camera in self._cameras.values():
            camera.connect()

    def disconnect(self) -> None:
        if self._cameras is None:
            return
        for camera in self._cameras.values():
            disconnect = getattr(camera, "disconnect", None)
            if disconnect is not None:
                disconnect()
        self._cameras = None

    def read(self) -> dict[str, np.ndarray]:
        if self._cameras is None:
            raise RuntimeError("Camera rig is not connected")
        return {key: np.asarray(camera.read(), dtype=np.uint8) for key, camera in self._cameras.items()}


class ArxRobotAdapter:
    """Local ARX inference adapter: 3 RGB cameras + repository-internal ZeroRPC client."""

    def __init__(
        self,
        *,
        rpc_client: ArxROS2RPCClient | None = None,
        camera_rig: DummyCameraRig | RealSenseCameraRig | None = None,
        control_mode: str = "normal",
        dry_run: bool = False,
    ):
        self._client = rpc_client
        self._camera_rig = camera_rig
        self._control_mode = control_mode
        self._dry_run = dry_run
        self._dummy_state = np.zeros(14, dtype=np.float32)

    def connect(self) -> None:
        if self._control_mode != "normal":
            raise ValueError("ARX OpenPI inference currently requires robot.control_mode='normal'")

        if self._camera_rig is not None:
            self._camera_rig.connect()

        if self._dry_run:
            return

        if self._client is None:
            raise RuntimeError("RPC client is required when dry_run is false")

        if not self._client.system_connect():
            raise RuntimeError("Failed to connect to the ARX RPC server")

    def disconnect(self) -> None:
        if self._camera_rig is not None:
            self._camera_rig.disconnect()

        if self._client is None:
            return

        if self._dry_run:
            self._client.close()
        else:
            self._client.disconnect()

    def get_full_state(self) -> dict[str, Any]:
        if self._dry_run:
            return {
                "left_arm": {
                    "joint_positions": np.zeros(7, dtype=np.float32),
                    "joint_velocities": np.zeros(7, dtype=np.float32),
                    "joint_currents": np.zeros(7, dtype=np.float32),
                    "end_pose": self._dummy_state[:6].copy(),
                    "gripper": float(self._dummy_state[12]),
                },
                "right_arm": {
                    "joint_positions": np.zeros(7, dtype=np.float32),
                    "joint_velocities": np.zeros(7, dtype=np.float32),
                    "joint_currents": np.zeros(7, dtype=np.float32),
                    "end_pose": self._dummy_state[6:12].copy(),
                    "gripper": float(self._dummy_state[13]),
                },
            }

        if self._client is None:
            raise RuntimeError("RPC client is not configured")

        full_state = self._client.get_full_state()
        if full_state is None:
            raise RuntimeError("Failed to read full state from the ARX RPC server")
        return full_state

    def get_command_mode(self) -> str | None:
        if self._dry_run:
            return "mock"
        if self._client is None:
            raise RuntimeError("RPC client is not configured")
        return self._client.get_command_mode()

    def read_policy_observation(self, *, image_height: int, image_width: int, prompt: str) -> dict[str, Any]:
        full_state = self.get_full_state()
        if self._camera_rig is None:
            raise RuntimeError("Camera rig is not configured")
        images = self._camera_rig.read()
        return make_arx_observation(
            state_from_full_state(full_state),
            images,
            prompt,
            height=image_height,
            width=image_width,
        )

    def apply_action_chunk(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        *,
        action_horizon: int,
    ) -> list[DualArmCommandResult]:
        commands = delta_action_chunk_to_absolute_commands(state, actions, action_horizon=action_horizon)
        results = []

        for command in commands:
            if self._dry_run:
                self._dummy_state[:6] = command.left_pose
                self._dummy_state[6:12] = command.right_pose
                self._dummy_state[12] = command.left_gripper
                self._dummy_state[13] = command.right_gripper
                results.append(
                    DualArmCommandResult(
                        command=command,
                        ack={
                            "accepted": True,
                            "executed": False,
                            "sequence_id": -1,
                            "left_pose": command.left_pose.tolist(),
                            "right_pose": command.right_pose.tolist(),
                            "left_gripper": command.left_gripper,
                            "right_gripper": command.right_gripper,
                            "message": "mock mode",
                        },
                    )
                )
                continue

            if self._client is None:
                raise RuntimeError("RPC client is not configured")

            ack = self._client.set_dual_ee_poses(
                command.left_pose,
                command.right_pose,
                command.left_gripper,
                command.right_gripper,
            )
            if not ack or not ack.get("accepted", False):
                message = "missing ack" if not ack else ack.get("message", "command rejected")
                raise RuntimeError(f"ARX command was not accepted by RPC server: {message}")
            results.append(DualArmCommandResult(command=command, ack=ack))

        return results
