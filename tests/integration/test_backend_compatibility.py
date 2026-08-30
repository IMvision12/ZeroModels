import os

import pytest
from keras import ops

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    get_cached_model,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# TF CPU segfaults in tf.matmul for large SAM models (known TF bug).
SKIP_TF_CPU = {"SAMModel", "SAMPromptableSegment", "SAM2PromptableSegment"}


def _skip_if_incompatible(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_TF_CPU:
        try:
            import tensorflow as tf

            if not tf.config.list_physical_devices("GPU"):
                pytest.skip(f"{model_name}: TF CPU segfaults in matmul")
        except ImportError:
            pass


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_forward_pass(model_name):
    _skip_if_incompatible(model_name)
    config = MODEL_TEST_CONFIGS[model_name]
    model = get_cached_model(config)
    input_data = create_test_input(config, model=model)
    output = model(input_data)

    expected = config["expected_output_shape"]
    if expected is None:
        # Models with dynamic output shapes (e.g. SAM): just check it runs
        return

    if isinstance(expected, dict):
        assert isinstance(output, dict), (
            f"{model_name}: expected dict output, got {type(output)}"
        )
        for key, shape in expected.items():
            if shape is not None:
                assert key in output, f"{model_name}: missing output key '{key}'"
                assert output[key].shape == shape, (
                    f"{model_name}[{key}]: expected {shape}, got {output[key].shape}"
                )
    elif isinstance(expected, list):
        assert isinstance(output, (list, tuple)), (
            f"{model_name}: expected list output, got {type(output)}"
        )
        assert len(output) == len(expected), (
            f"{model_name}: expected {len(expected)} outputs, got {len(output)}"
        )
        for i, shape in enumerate(expected):
            if shape is not None:
                assert output[i].shape == shape, (
                    f"{model_name}[{i}]: expected {shape}, got {output[i].shape}"
                )
    else:
        assert output.shape == expected, (
            f"{model_name}: expected {expected}, got {output.shape}"
        )


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_no_nans(model_name):
    _skip_if_incompatible(model_name)
    config = MODEL_TEST_CONFIGS[model_name]
    model = get_cached_model(config)
    input_data = create_test_input(config, model=model)
    output = model(input_data)

    if isinstance(output, dict):
        for key, value in output.items():
            has_nans = bool(ops.any(ops.isnan(value)))
            assert not has_nans, f"{model_name}[{key}] contains NaN values"
    elif isinstance(output, (list, tuple)):
        for i, value in enumerate(output):
            has_nans = bool(ops.any(ops.isnan(value)))
            assert not has_nans, f"{model_name}[{i}] contains NaN values"
    else:
        has_nans = bool(ops.any(ops.isnan(output)))
        assert not has_nans, f"{model_name} output contains NaN values"
