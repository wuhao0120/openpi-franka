"""ARX LIFT2 ROS 2 ZeroRPC server vendored into openpi-franka."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np

from openpi.arx.arx_lift2_ros2_bridge import ARXLift2Bridge

try:
    import zerorpc
except ImportError:  # pragma: no cover - depends on deployment environment.
    zerorpc = None

log = logging.getLogger(__name__)

NUM_ARM_JOINTS = 7
NUM_POSE_DIMS = 6
COMMAND_MODES = ("silent", "execute")


class _LatestOnlyCommandWorker:
    """Latest-only worker: only the most recent command is executed."""

    def __init__(self, name: str, handler):
        self._name = name
        self._handler = handler
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._latest_args = None
        self._thread = threading.Thread(target=self._run, name=f"latest-only-{name}", daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, args):
        with self._lock:
            self._latest_args = args
        self._event.set()

    def clear(self):
        with self._lock:
            self._latest_args = None
        self._event.clear()

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop_event.is_set():
            self._event.wait(timeout=0.1)
            if not self._event.is_set():
                continue

            while not self._stop_event.is_set():
                with self._lock:
                    args = self._latest_args
                    self._latest_args = None
                    if args is None:
                        self._event.clear()
                        break

                try:
                    self._handler(*args)
                except Exception as exc:
                    log.error("Latest-only worker[%s] error: %s", self._name, exc)


class ArxROS2RPCServer:
    """ARX LIFT2 ROS 2 RPC server implementation."""

    _REPORT_EVERY = 200

    def __init__(
        self,
        arms_only: bool = False,
        control_mode: str = "normal",
        command_mode: str = "silent",
        *,
        start_workers: bool = True,
    ):
        if command_mode not in COMMAND_MODES:
            raise ValueError(f"Unsupported command_mode: {command_mode}")
        self.bridge = None
        self.arms_only = arms_only
        self.control_mode = control_mode
        self.command_mode = command_mode

        self._state_lock = threading.RLock()
        self._perf_lock = threading.Lock()
        self._perf_count = 0
        self._command_sequence = 0
        self._perf_sums = {}
        self._perf_maxs = {}

        self._set_full_command_worker = None
        if start_workers:
            self._set_full_command_worker = _LatestOnlyCommandWorker(
                name="set_full_command",
                handler=self._apply_latest_full_command,
            )
            self._set_full_command_worker.start()

    @staticmethod
    def _to_list(data):
        if isinstance(data, list):
            return data
        if isinstance(data, tuple):
            return list(data)
        if hasattr(data, "tolist"):
            return data.tolist()
        return list(data)

    def _record_perf(self, method: str, elapsed_ms: float):
        with self._perf_lock:
            self._perf_count += 1
            self._perf_sums[method] = self._perf_sums.get(method, 0.0) + elapsed_ms
            self._perf_maxs[method] = max(self._perf_maxs.get(method, 0.0), elapsed_ms)

            if self._perf_count % self._REPORT_EVERY == 0:
                parts = []
                for name in sorted(self._perf_sums.keys()):
                    avg = self._perf_sums[name] / max(self._REPORT_EVERY, 1)
                    parts.append(f"{name}={avg:.2f}/{self._perf_maxs[name]:.2f}ms")
                log.info("[PERF server] %s calls | %s", self._perf_count, " | ".join(parts))
                self._perf_sums.clear()
                self._perf_maxs.clear()

    def _require_bridge(self) -> ARXLift2Bridge:
        with self._state_lock:
            bridge = self.bridge
        if bridge is None:
            raise RuntimeError("Bridge not connected. Call system_connect first.")
        return bridge

    def _next_command_sequence(self) -> int:
        with self._state_lock:
            self._command_sequence += 1
            return self._command_sequence

    @staticmethod
    def _to_float_list(data, expected_dim: int, *, name: str) -> list[float]:
        values = [float(v) for v in ArxROS2RPCServer._to_list(data)]
        if len(values) != expected_dim:
            raise ValueError(f"{name} must have {expected_dim} values, got {len(values)}")
        return values

    def _make_dual_ee_ack(
        self,
        *,
        left_pose: list[float],
        right_pose: list[float],
        left_gripper: float,
        right_gripper: float,
        accepted: bool = True,
        executed: bool = False,
        message: str = "ok",
    ) -> dict:
        return {
            "accepted": bool(accepted),
            "executed": bool(executed),
            "sequence_id": self._next_command_sequence(),
            "left_pose": left_pose,
            "right_pose": right_pose,
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
            "message": message,
        }

    def _apply_latest_full_command(self, left_positions, right_positions, vx, vy, wz, height):
        t0 = time.perf_counter()
        try:
            bridge = self._require_bridge()
            bridge.set_full_command(left_positions, right_positions, vx, vy, wz, height)
        except Exception as exc:
            log.error("Error applying latest set_full_command: %s", exc)
        finally:
            self._record_perf("set_full_command_apply", (time.perf_counter() - t0) * 1000)

    def system_connect(self, timeout: float = 10.0):
        t0 = time.perf_counter()
        try:
            with self._state_lock:
                if self.bridge is not None and self.bridge.is_connected():
                    return True

            import rclpy

            if not rclpy.ok():
                rclpy.init()

            new_bridge = ARXLift2Bridge(enable_lift=not self.arms_only, control_mode=self.control_mode)
            new_bridge.connect(timeout=timeout)

            old_bridge = None
            with self._state_lock:
                old_bridge = self.bridge
                self.bridge = new_bridge

            if old_bridge is not None and old_bridge is not new_bridge:
                try:
                    old_bridge.disconnect()
                except Exception:
                    pass

            return True
        except Exception as exc:
            log.error("Failed to connect to ARX LIFT2 system: %s", exc)
            return False
        finally:
            self._record_perf("system_connect", (time.perf_counter() - t0) * 1000)

    def disconnect(self):
        t0 = time.perf_counter()
        if self._set_full_command_worker is not None:
            self._set_full_command_worker.clear()

        bridge = None
        with self._state_lock:
            bridge = self.bridge
            self.bridge = None

        if bridge is not None:
            try:
                bridge.disconnect()
            except Exception as exc:
                log.error("Error disconnecting from ARX LIFT2 system: %s", exc)

        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            log.error("Error shutting down rclpy: %s", exc)
        finally:
            self._record_perf("disconnect", (time.perf_counter() - t0) * 1000)

    def _shutdown_server(self):
        if self._set_full_command_worker is not None:
            try:
                self._set_full_command_worker.stop()
            except Exception as exc:
                log.warning("Error stopping latest-only worker: %s", exc)

        try:
            self.disconnect()
        except Exception as exc:
            log.warning("Error during disconnect: %s", exc)

    def is_connected(self) -> bool:
        with self._state_lock:
            bridge = self.bridge
        return bridge is not None and bridge.is_connected()

    def get_command_mode(self) -> str:
        return self.command_mode

    def get_full_state(self):
        bridge = self._require_bridge()
        t0 = time.perf_counter()
        try:
            return bridge.get_full_state_serialized()
        finally:
            self._record_perf("get_full_state", (time.perf_counter() - t0) * 1000)

    def get_chassis_height(self) -> float:
        bridge = self._require_bridge()
        try:
            return float(bridge.get_chassis_height())
        except Exception as exc:
            log.error("Error getting chassis height: %s", exc)
            return 0.0

    def set_chassis_height(self, height: float):
        bridge = self._require_bridge()
        try:
            bridge.set_chassis_height(height)
        except Exception as exc:
            log.error("Error setting chassis height: %s", exc)

    def set_chassis_velocity(self, vx: float, vy: float, wz: float):
        bridge = self._require_bridge()
        try:
            bridge.set_chassis_velocity(vx, vy, wz)
        except Exception as exc:
            log.error("Error setting chassis velocity: %s", exc)

    def get_left_joint_positions(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_left_joint_positions().tolist()
        except Exception as exc:
            log.error("Error getting left joint positions: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_left_joint_velocities(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_left_joint_velocities().tolist()
        except Exception as exc:
            log.error("Error getting left joint velocities: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_left_joint_currents(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_left_joint_currents().tolist()
        except Exception as exc:
            log.error("Error getting left joint currents: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_left_end_pose(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_left_end_pose().tolist()
        except Exception as exc:
            log.error("Error getting left end pose: %s", exc)
            return [0.0] * NUM_POSE_DIMS

    def get_left_gripper_position(self) -> float:
        bridge = self._require_bridge()
        try:
            return float(bridge.get_left_gripper_position())
        except Exception as exc:
            log.error("Error getting left gripper position: %s", exc)
            return 0.0

    def set_left_joint_positions(self, positions):
        bridge = self._require_bridge()
        try:
            bridge.set_left_joint_positions(np.asarray(positions, dtype=np.float32))
        except Exception as exc:
            log.error("Error setting left joint positions: %s", exc)

    def set_left_gripper(self, position: float):
        bridge = self._require_bridge()
        try:
            bridge.set_left_gripper(position)
        except Exception as exc:
            log.error("Error setting left gripper: %s", exc)

    def get_right_joint_positions(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_right_joint_positions().tolist()
        except Exception as exc:
            log.error("Error getting right joint positions: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_right_joint_velocities(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_right_joint_velocities().tolist()
        except Exception as exc:
            log.error("Error getting right joint velocities: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_right_joint_currents(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_right_joint_currents().tolist()
        except Exception as exc:
            log.error("Error getting right joint currents: %s", exc)
            return [0.0] * NUM_ARM_JOINTS

    def get_right_end_pose(self):
        bridge = self._require_bridge()
        try:
            return bridge.get_right_end_pose().tolist()
        except Exception as exc:
            log.error("Error getting right end pose: %s", exc)
            return [0.0] * NUM_POSE_DIMS

    def get_right_gripper_position(self) -> float:
        bridge = self._require_bridge()
        try:
            return float(bridge.get_right_gripper_position())
        except Exception as exc:
            log.error("Error getting right gripper position: %s", exc)
            return 0.0

    def set_right_joint_positions(self, positions):
        bridge = self._require_bridge()
        try:
            bridge.set_right_joint_positions(np.asarray(positions, dtype=np.float32))
        except Exception as exc:
            log.error("Error setting right joint positions: %s", exc)

    def set_right_gripper(self, position: float):
        bridge = self._require_bridge()
        try:
            bridge.set_right_gripper(position)
        except Exception as exc:
            log.error("Error setting right gripper: %s", exc)

    def set_dual_joint_positions(self, left_positions, right_positions):
        bridge = self._require_bridge()
        try:
            bridge.set_dual_joint_positions(
                np.asarray(left_positions, dtype=np.float32),
                np.asarray(right_positions, dtype=np.float32),
            )
        except Exception as exc:
            log.error("Error setting dual joint positions: %s", exc)

    def set_full_command(self, left_positions, right_positions, vx, vy, wz, height):
        self._require_bridge()
        t0 = time.perf_counter()

        left_list = self._to_list(left_positions)
        right_list = self._to_list(right_positions)
        if self._set_full_command_worker is None:
            self._apply_latest_full_command(left_list, right_list, float(vx), float(vy), float(wz), float(height))
        else:
            self._set_full_command_worker.submit(
                (left_list, right_list, float(vx), float(vy), float(wz), float(height))
            )

        self._record_perf("set_full_command_enqueue", (time.perf_counter() - t0) * 1000)

    def set_left_ee_pose(self, pose, gripper: float = 0.0):
        bridge = self._require_bridge()
        try:
            bridge.set_left_ee_pose(np.asarray(pose, dtype=np.float32), gripper)
        except Exception as exc:
            log.error("Error setting left ee pose: %s", exc)

    def set_right_ee_pose(self, pose, gripper: float = 0.0):
        bridge = self._require_bridge()
        try:
            bridge.set_right_ee_pose(np.asarray(pose, dtype=np.float32), gripper)
        except Exception as exc:
            log.error("Error setting right ee pose: %s", exc)

    def set_dual_ee_poses(self, left_pose, right_pose, left_gripper: float = 0.0, right_gripper: float = 0.0):
        bridge = self._require_bridge()
        left_pose = self._to_float_list(left_pose, NUM_POSE_DIMS, name="left_pose")
        right_pose = self._to_float_list(right_pose, NUM_POSE_DIMS, name="right_pose")
        left_gripper = float(left_gripper)
        right_gripper = float(right_gripper)

        if self.command_mode == "silent":
            ack = self._make_dual_ee_ack(
                left_pose=left_pose,
                right_pose=right_pose,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                executed=False,
                message="silent mode: command received but not executed",
            )
            log.info(
                "[SILENT] seq=%s left_xyz=%s right_xyz=%s gripper=(%.3f, %.3f)",
                ack["sequence_id"],
                np.round(left_pose[:3], 4).tolist(),
                np.round(right_pose[:3], 4).tolist(),
                left_gripper,
                right_gripper,
            )
            return ack

        try:
            bridge.set_dual_ee_poses_fast(left_pose, right_pose, left_gripper, right_gripper)
            return self._make_dual_ee_ack(
                left_pose=left_pose,
                right_pose=right_pose,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                executed=True,
                message="executed",
            )
        except Exception as exc:
            log.error("Error setting dual ee poses: %s", exc)
            return self._make_dual_ee_ack(
                left_pose=left_pose,
                right_pose=right_pose,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                accepted=False,
                executed=False,
                message=str(exc),
            )

    def emergency_stop(self):
        bridge = self._require_bridge()
        try:
            bridge.emergency_stop()
        except Exception as exc:
            log.error("Error during emergency stop: %s", exc)


def _serve(
    bind_address: str = "tcp://0.0.0.0:4242",
    arms_only: bool = False,
    control_mode: str = "normal",
    command_mode: str = "silent",
):
    if zerorpc is None:  # pragma: no cover - depends on deployment environment.
        raise RuntimeError("zerorpc is not installed. Install it on the robot-control machine before serving RPC.")

    server_impl = ArxROS2RPCServer(arms_only=arms_only, control_mode=control_mode, command_mode=command_mode)
    rpc_server = zerorpc.Server(server_impl, heartbeat=60)
    rpc_server.bind(bind_address)

    def _signal_handler(signum, _frame):
        log.info("Received signal %s, shutting down...", signum)
        try:
            rpc_server.close()
        except Exception:
            pass
        try:
            server_impl._shutdown_server()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rpc_server.run()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received, shutting down...")
    finally:
        try:
            rpc_server.close()
        except Exception:
            pass
        server_impl._shutdown_server()


def main():
    parser = argparse.ArgumentParser(description="ARX ROS2 RPC Server")
    parser.add_argument("--arms-only", action="store_true", default=True, help="Only enable dual-arm control")
    parser.add_argument("--with-lift", action="store_true", help="Also enable the LIFT chassis")
    parser.add_argument(
        "--control-mode",
        default="normal",
        choices=["normal", "joint_control_v1"],
        help="Dual-arm controller mode",
    )
    parser.add_argument("--bind", default="tcp://0.0.0.0:4242", help="ZeroRPC bind address")
    parser.add_argument(
        "--command-mode",
        default="silent",
        choices=COMMAND_MODES,
        help="Whether command RPCs only acknowledge receipt or publish to the robot",
    )
    args = parser.parse_args()

    arms_only = not args.with_lift
    _serve(bind_address=args.bind, arms_only=arms_only, control_mode=args.control_mode, command_mode=args.command_mode)


if __name__ == "__main__":
    main()
