import json
import os

import keras
import pytest

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    instantiate_model,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# Models that cause backend-specific issues during serialization
SKIP_SERIALIZATION_TF = {"SAM2PromptableSegment"}


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_config_roundtrip(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_SERIALIZATION_TF:
        pytest.skip(f"{model_name} causes TF backend segfault during serialization")

    config = MODEL_TEST_CONFIGS[model_name]
    model = instantiate_model(config)

    cfg = model.get_config()
    revived = model.__class__.from_config(cfg)

    assert isinstance(revived, model.__class__), (
        f"{model_name}: from_config produced wrong type: {type(revived).__name__}"
    )

    input_data = create_test_input(config, model=model)
    output = revived(input_data)
    assert output is not None, f"{model_name}: revived model produced None output"


@pytest.mark.serialization
def test_quantized_layer_serialization():
    """Config + keras (de)serialization round-trip for the quantized layers.

    int8 / int4 only so it is backend-portable (fp8 is torch/jax-only).
    """
    from zeromodels.quantization import (
        QuantizedDense,
        QuantizedEinsumDense,
        QuantizedEmbedding,
        QuantizedExperts,
    )

    quant_layers = [
        QuantizedDense(8, mode="int8"),
        QuantizedDense(8, mode="int4", group_size=64),
        QuantizedEmbedding(16, 8),
        QuantizedExperts(4, 8, 16, mode="int4"),
        QuantizedEinsumDense(
            "abc,cde->abde",
            (None, 4, 8),
            (16, 4, 8),
            mode="int8",
            bias_axes="de",
            bias_shape=(4, 8),
        ),
    ]
    for layer in quant_layers:
        name = type(layer).__name__
        revived = layer.__class__.from_config(layer.get_config())
        assert isinstance(revived, layer.__class__), name
        assert revived.get_config() == layer.get_config(), f"{name}: config mismatch"

        blob = json.dumps(keras.saving.serialize_keras_object(layer), default=str)
        deserialized = keras.saving.deserialize_keras_object(json.loads(blob))
        assert isinstance(deserialized, layer.__class__), f"{name}: keras deserialize"


def _toy_quantizable_model(name="toy"):
    import numpy as np
    from keras import layers

    class Blk(layers.Layer):
        def __init__(self, dim, **kw):
            super().__init__(**kw)
            self.q = layers.Dense(dim, name="q")
            self.o = layers.Dense(dim, name="o")

        def call(self, x):
            return self.o(self.q(x))

    class Toy(keras.Model):
        def __init__(self, n=64, dim=32, depth=2, **kw):
            super().__init__(**kw)
            self.emb = layers.Embedding(n, dim, name="token_embedding")
            self.blocks = [Blk(dim, name=f"block_{i}") for i in range(depth)]
            self.lm_head = layers.Dense(n, use_bias=False, name="lm_head")

        def call(self, x):
            h = self.emb(x)
            for b in self.blocks:
                h = b(h)
            return self.lm_head(h)

    m = Toy(name=name)
    m(np.array([[1, 2, 3, 4]]))
    return m


def test_quantize_in_place_paths_have_no_collisions():
    """In-place swap keeps full layer paths (no `block_*/q` -> bare `q` collapse),
    so the sharded `.weights.json` format round-trips."""
    from zeromodels.quantization import quantize_model

    model = _toy_quantizable_model()
    quantize_model(model, "int4")
    paths = [w.path for w in model.weights]
    assert len(paths) == len(set(paths)), f"path collision: {paths}"


def test_no_float_load_matches_load_then_quantize():
    """quantize_and_load streams a float checkpoint into int storage and lands
    byte-identical to building float then quantizing (int8 / int4, all backends)."""
    import numpy as np
    from keras import ops

    from zeromodels.conversion.weight_transfer_util import transfer_weights
    from zeromodels.quantization import quantize_and_load, quantize_model

    def transfer(model, sd):
        if not model.built or not model.weights:
            model(np.array([[0, 1, 2, 3]]))
        name_map = {"token_embedding.embeddings": "emb", "kernel": "weight"}
        for w in model.weights:
            key = w.path.split("/", 1)[1].replace("/", ".")
            for old, new in name_map.items():
                key = key.replace(old, new)
            transfer_weights(w.path, w, sd[key])

    rng = np.random.default_rng(0)

    def make_sd(model):
        sd = {}
        for w in model.weights:
            key = w.path.split("/", 1)[1].replace("/", ".")
            key = key.replace("token_embedding.embeddings", "emb")
            key = key.replace("kernel", "weight")
            shape = tuple(w.shape)
            if (
                key.endswith(".weight")
                and "block" in key
                or key.endswith("head.weight")
            ):
                shape = (shape[1], shape[0])  # HF stores Dense weight transposed
            sd[key] = rng.standard_normal(shape).astype("float32")
        return sd

    x = np.array([[3, 9, 40, 60]])
    for mode in ("int8", "int4"):
        ref = _toy_quantizable_model()
        sd = make_sd(ref)
        transfer(ref, sd)
        quantize_model(ref, mode)
        y_ref = ops.convert_to_numpy(ref(x))

        # a fresh UNBUILT instance of the same class for the no-float path
        model = type(ref)(name="toy")
        quantize_and_load(model, mode, transfer, sd)
        y = ops.convert_to_numpy(model(x))
        assert float(np.max(np.abs(y - y_ref))) == 0.0, mode
        assert model._quantization_config.mode == mode


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_keras_serialization_roundtrip(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_SERIALIZATION_TF:
        pytest.skip(f"{model_name} causes TF backend segfault during serialization")

    config = MODEL_TEST_CONFIGS[model_name]
    model = instantiate_model(config)

    serialized = keras.saving.serialize_keras_object(model)
    json_str = json.dumps(serialized, indent=4, default=str)
    revived = keras.saving.deserialize_keras_object(json.loads(json_str))

    assert isinstance(revived, model.__class__), (
        f"{model_name}: keras deserialization produced wrong type: "
        f"{type(revived).__name__}"
    )

    input_data = create_test_input(config, model=model)
    output = revived(input_data)
    assert output is not None, f"{model_name}: revived model produced None output"
