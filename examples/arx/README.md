# ARX Local Inference

This directory runs the full ARX deployment in the repository-local shape:

- The robot-control computer runs the built-in ZeroRPC server from `openpi-franka`.
- The inference computer runs the built-in local OpenPI inference loop from `openpi-franka`.
- No cross-repo imports, `sys.path` injection, or `ARX_new` source paths are required.

## 1. Start the robot-control RPC server

On the robot-control computer, activate the ROS2 environment that provides `rclpy`, `arm_control`, and `arx5_arm_msg`,
then run:

```bash
python scripts/serve_arx_rpc.py --control-mode normal --command-mode silent
```

`normal` mode is required because OpenPI actions are executed as dual-arm absolute EE poses.
Use `--command-mode execute` only after validating a fine-tuned checkpoint.

## 2. Configure the inference computer

Edit [cfg_arx_pi.yaml](/home/wuhao/workspace/openpi-franka/examples/arx/config/cfg_arx_pi.yaml) and set:

- `model.name`
- `model.checkpoint_dir`
- `model.norm_stats_dir` (optional, useful when evaluating pi05_base with ARX norm stats)
- `robot.ip`
- `robot.port`
- `robot.control_mode`
- `robot.action_fps`
- `robot.action_horizon`
- `image.height`
- `image.width`
- `run.task_description`
- `run.max_steps`
- `run.mode`

For `silent` or `execute`, export the three local camera serial numbers before starting inference:

```bash
export OPENPI_ARX_HEAD_CAMERA=<head_camera_serial>
export OPENPI_ARX_LEFT_WRIST_CAMERA=<left_wrist_camera_serial>
export OPENPI_ARX_RIGHT_WRIST_CAMERA=<right_wrist_camera_serial>
```

These environment variables keep the YAML surface minimal while preserving a fixed 3-camera schema.

## 3. Run local inference

```bash
python examples/arx/inference_pi_with_arx.py
```

The runtime loop is fixed to:

1. Load the local OpenPI policy.
2. Read three local RGB cameras.
3. Read `get_full_state()` through the vendored `ArxROS2RPCClient`.
4. Build `observation/state`, three images, and `prompt`.
5. Call `policy.infer(obs)`.
6. Convert each 14D `delta_ee` action to absolute dual-arm EE poses.
7. Call `set_dual_ee_poses(...)` through ZeroRPC and require an ack from the robot-control computer.

## Modes

- `mock`: uses dummy cameras and can fall back to a tiny dummy policy if the checkpoint path is missing.
- `silent`: uses the real policy, cameras, and ZeroRPC server; the server acknowledges commands without publishing them.
- `execute`: uses the same ZeroRPC path, but the server publishes commands to the robot.

`silent` is the expected first pi05_base hardware validation mode.
