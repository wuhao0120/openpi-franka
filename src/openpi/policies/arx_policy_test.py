import numpy as np

from openpi.models import model as _model
from openpi.policies import arx_policy


def test_arx_inputs_maps_fixed_schema():
    transform = arx_policy.ArxInputs(model_type=_model.ModelType.PI05)
    data = {
        "observation/state": np.arange(14, dtype=np.float32),
        "observation/image": np.zeros((3, 8, 6), dtype=np.float32),
        "observation/wrist_image": np.ones((3, 8, 6), dtype=np.float32),
        "observation/right_wrist_image": np.full((3, 8, 6), 0.5, dtype=np.float32),
        "actions": np.ones((4, 14), dtype=np.float32),
        "prompt": "pick up the block",
    }

    result = transform(data)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (4, 14)
    assert result["prompt"] == "pick up the block"
    assert result["image"]["base_0_rgb"].shape == (8, 6, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (8, 6, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (8, 6, 3)
    assert set(result["image_mask"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image"]["base_0_rgb"].dtype == np.uint8


def test_arx_inputs_extracts_policy_state_from_full_lerobot_state():
    transform = arx_policy.ArxInputs(model_type=_model.ModelType.PI05)
    full_state = np.arange(28, dtype=np.float32)
    data = {
        "observation/state": full_state,
        "observation/image": np.zeros((8, 6, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((8, 6, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.full((8, 6, 3), 2, dtype=np.uint8),
        "actions": np.ones((4, 14), dtype=np.float32),
        "prompt": "pick up the block",
    }

    result = transform(data)

    np.testing.assert_array_equal(
        result["state"],
        np.asarray([*range(12, 24), 24, 26], dtype=np.float32),
    )


def test_arx_outputs_slice_to_14d():
    transform = arx_policy.ArxOutputs(action_dim=14)
    outputs = transform({"actions": np.ones((6, 32), dtype=np.float32)})

    assert outputs["actions"].shape == (6, 14)
    np.testing.assert_array_equal(outputs["actions"], np.ones((6, 14), dtype=np.float32))
