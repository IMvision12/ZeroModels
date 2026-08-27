import numpy as np
import pytest
from keras import ops

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    instantiate_model,
)

MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())
SKIP_SAVING = {
    "DeepseekVLHybridConditionalGenerate",
}


def _to_numpy(tensor):
    return ops.convert_to_numpy(tensor)


def _assert_outputs_close(original, loaded, model_name, rtol=1e-5, atol=1e-5):
    if isinstance(original, dict):
        for key in original:
            np.testing.assert_allclose(
                _to_numpy(original[key]),
                _to_numpy(loaded[key]),
                rtol=rtol,
                atol=atol,
                err_msg=f"{model_name}[{key}] output mismatch after save/load",
            )
    elif isinstance(original, (list, tuple)):
        for i in range(len(original)):
            np.testing.assert_allclose(
                _to_numpy(original[i]),
                _to_numpy(loaded[i]),
                rtol=rtol,
                atol=atol,
                err_msg=f"{model_name}[{i}] output mismatch after save/load",
            )
    else:
        np.testing.assert_allclose(
            _to_numpy(original),
            _to_numpy(loaded),
            rtol=rtol,
            atol=atol,
            err_msg=f"{model_name} output mismatch after save/load",
        )


@pytest.mark.saving
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_save_weights_h5_roundtrip(model_name, tmp_path):
    if model_name in SKIP_SAVING:
        pytest.skip(f"{model_name} is subclassed; load_weights needs a built model")
    config = MODEL_TEST_CONFIGS[model_name]
    model = instantiate_model(config)
    input_data = create_test_input(config, model=model)

    original_output = model(input_data)

    weights_path = str(tmp_path / f"{model_name}.weights.h5")
    model.save_weights(weights_path)

    fresh_model = instantiate_model(config)
    if not fresh_model.built:
        # Subclassed models materialize weights on first call; build the
        # fresh instance the same way before loading into it.
        fresh_model(input_data)
    fresh_model.load_weights(weights_path)

    loaded_output = fresh_model(input_data)
    _assert_outputs_close(original_output, loaded_output, model_name)
