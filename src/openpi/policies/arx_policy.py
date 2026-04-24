import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ARX_STATE_DIM = 14
ARX_ACTION_DIM = 14
# The ARX LeRobot recorder stores the full robot state as:
# left joints 6D, right joints 6D, left EE pose 6D, right EE pose 6D,
# left gripper state/cmd, right gripper state/cmd. The policy consumes only
# EE poses and gripper states.
ARX_FULL_STATE_TO_POLICY_INDICES = (*range(12, 24), 24, 26)


def make_arx_example() -> dict:
    """Creates a random input example for the ARX R5 dual-arm policy."""
    return {
        "observation/state": np.random.rand(ARX_STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_state(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == ARX_STATE_DIM:
        return state
    if state.shape[-1] <= max(ARX_FULL_STATE_TO_POLICY_INDICES):
        raise ValueError(
            f"ARX state must be either {ARX_STATE_DIM}D policy state or full LeRobot state with at least "
            f"{max(ARX_FULL_STATE_TO_POLICY_INDICES) + 1} dims, got shape {state.shape}"
        )
    return np.take(state, ARX_FULL_STATE_TO_POLICY_INDICES, axis=-1)


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Convert ARX observations into the common model input format."""

    model_type: _model.ModelType
    state_key: str = "observation/state"
    base_image_key: str = "observation/image"
    left_wrist_image_key: str = "observation/wrist_image"
    right_wrist_image_key: str = "observation/right_wrist_image"
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data[self.base_image_key])
        left_wrist_image = _parse_image(data[self.left_wrist_image_key])
        right_wrist_image = _parse_image(data[self.right_wrist_image_key])

        inputs = {
            "state": _parse_state(data[self.state_key]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if self.prompt_key in data:
            prompt = data[self.prompt_key]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the 14D ARX action format."""

    action_dim: int = ARX_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}
