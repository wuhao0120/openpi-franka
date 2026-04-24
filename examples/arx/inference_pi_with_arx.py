from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from openpi.arx.arx_robot_adapter import ArxRobotAdapter
from openpi.arx.arx_robot_adapter import DummyCameraRig
from openpi.arx.arx_robot_adapter import RealSenseCameraRig
from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class DummyArxPolicy:
    """Small fixed-action policy used to validate the dry-run pipeline without a checkpoint."""

    def __init__(self, action_horizon: int):
        action = np.zeros((action_horizon, 14), dtype=np.float32)
        action[:, 0] = 0.005
        action[:, 6] = -0.005
        action[:, 12] = 0.2
        action[:, 13] = 0.2
        self._action_chunk = action

    def infer(self, _obs: dict[str, object]) -> dict[str, np.ndarray]:
        return {"actions": self._action_chunk.copy()}


class ArxInference:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file)

        model_cfg = cfg["model"]
        self.train_config = _config.get_config(model_cfg["name"])
        self.checkpoint_dir = self._resolve_path_or_uri(model_cfg["checkpoint_dir"])
        self.norm_stats_dir = model_cfg.get("norm_stats_dir")

        robot_cfg = cfg["robot"]
        self.robot_ip = robot_cfg["ip"]
        self.robot_port = int(robot_cfg.get("port", 4242))
        self.control_mode = robot_cfg.get("control_mode", "normal")
        self.action_fps = float(robot_cfg.get("action_fps", 10))
        self.action_horizon = int(robot_cfg.get("action_horizon", 1))

        image_cfg = cfg["image"]
        self.image_height = int(image_cfg.get("height", 224))
        self.image_width = int(image_cfg.get("width", 224))

        run_cfg = cfg["run"]
        self.task_description = run_cfg.get("task_description", "")
        self.max_steps = int(run_cfg.get("max_steps", 0))
        self.run_mode = str(run_cfg.get("mode", "mock" if bool(run_cfg.get("dry_run", True)) else "execute"))
        if self.run_mode not in ("mock", "silent", "execute"):
            raise ValueError("run.mode must be one of: mock, silent, execute")

        if self.run_mode == "mock":
            camera_rig = DummyCameraRig(width=max(self.image_width, 640), height=max(self.image_height, 480))
            rpc_client = ArxROS2RPCClient(ip=self.robot_ip, port=self.robot_port, autoconnect=False)
        else:
            camera_rig = RealSenseCameraRig()
            rpc_client = ArxROS2RPCClient(ip=self.robot_ip, port=self.robot_port)

        self.robot = ArxRobotAdapter(
            rpc_client=rpc_client,
            camera_rig=camera_rig,
            control_mode=self.control_mode,
            dry_run=self.run_mode == "mock",
        )

    @staticmethod
    def _resolve_path_or_uri(value: str) -> str:
        if value.startswith("gs://"):
            return value
        return str(Path(value).expanduser())

    def _checkpoint_exists(self) -> bool:
        if self.checkpoint_dir.startswith("gs://"):
            return True
        return Path(self.checkpoint_dir).exists()

    def _load_norm_stats(self):
        if not self.norm_stats_dir:
            return None
        return _normalize.load(Path(self.norm_stats_dir).expanduser())

    def _load_policy(self):
        if self._checkpoint_exists():
            return _policy_config.create_trained_policy(
                self.train_config,
                self.checkpoint_dir,
                default_prompt=self.task_description,
                norm_stats=self._load_norm_stats(),
            )

        if self.run_mode != "mock":
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        log.info("Checkpoint %s not found. Using a dummy dry-run policy.", self.checkpoint_dir)
        return DummyArxPolicy(self.train_config.model.action_horizon)

    def run(self) -> None:
        policy = self._load_policy()
        self.robot.connect()
        try:
            command_mode = self.robot.get_command_mode()
            if command_mode != self.run_mode:
                raise RuntimeError(f"ARX RPC server command mode is {command_mode!r}, expected {self.run_mode!r}")

            warmup_obs = self.robot.read_policy_observation(
                image_height=self.image_height,
                image_width=self.image_width,
                prompt=self.task_description,
            )
            policy.infer(warmup_obs)

            step = 0
            while True:
                started = time.perf_counter()
                obs = self.robot.read_policy_observation(
                    image_height=self.image_height,
                    image_width=self.image_width,
                    prompt=self.task_description,
                )
                result = policy.infer(obs)
                command_results = self.robot.apply_action_chunk(
                    obs["observation/state"],
                    np.asarray(result["actions"], dtype=np.float32),
                    action_horizon=self.action_horizon,
                )

                if self.run_mode == "silent":
                    for command_result in command_results:
                        command = command_result.command
                        ack = command_result.ack
                        log.info(
                            "ack seq=%s executed=%s left_xyz=%s right_xyz=%s gripper=(%.2f, %.2f)",
                            ack.get("sequence_id"),
                            ack.get("executed"),
                            np.round(command.left_pose[:3], 4).tolist(),
                            np.round(command.right_pose[:3], 4).tolist(),
                            command.left_gripper,
                            command.right_gripper,
                        )
                elif step % 10 == 0 and command_results:
                    command = command_results[0].command
                    log.info(
                        "step=%d left_xyz=%s right_xyz=%s gripper=(%.2f, %.2f)",
                        step,
                        np.round(command.left_pose[:3], 4).tolist(),
                        np.round(command.right_pose[:3], 4).tolist(),
                        command.left_gripper,
                        command.right_gripper,
                    )

                step += 1
                if self.max_steps > 0 and step >= self.max_steps:
                    break

                sleep_time = 1.0 / self.action_fps - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local OpenPI inference for ARX R5")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "cfg_arx_pi.yaml",
        help="Path to the local ARX inference YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ArxInference(args.config).run()


if __name__ == "__main__":
    main()
