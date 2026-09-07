import json
import os

import keras
import pytest

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    get_cached_model,
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
    model = get_cached_model(config)

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


def test_activation_survives_quantize_roundtrip():
    """A Dense/EinsumDense activation is carried through quantize (and the swap
    back on dequantize), so a fused activation is not silently dropped."""
    import numpy as np
    from keras import layers, ops

    from zeromodels.quantization import dequantize_model, quantize_model

    inp = layers.Input((16,))
    h = layers.Dense(8, activation="relu", name="relu_dense")(inp)
    out = layers.Dense(4, activation="tanh", name="tanh_dense")(h)
    model = keras.Model(inp, out)

    x = np.random.default_rng(0).standard_normal((4, 16)).astype("float32")
    float_out = ops.convert_to_numpy(model(x))

    model = quantize_model(model, "int8")
    relu_dense = model.get_layer("relu_dense")
    assert type(relu_dense).__name__ == "QuantizedDense"
    assert keras.activations.serialize(relu_dense.activation) == "relu"

    quant_out = ops.convert_to_numpy(model(x))
    # tanh head bounds the output; a dropped activation would run unbounded.
    assert quant_out.min() >= -1.001 and quant_out.max() <= 1.001
    cos = float(
        float_out.ravel()
        @ quant_out.ravel()
        / (np.linalg.norm(float_out) * np.linalg.norm(quant_out) + 1e-9)
    )
    assert cos > 0.99, cos

    model = dequantize_model(model)
    revived = model.get_layer("relu_dense")
    assert isinstance(revived, layers.Dense) and not hasattr(revived, "quant_kernel")
    assert keras.activations.serialize(revived.activation) == "relu"
    deq_out = ops.convert_to_numpy(model(x))
    assert deq_out.min() >= -1.001 and deq_out.max() <= 1.001


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


def test_resolve_config_group_size_and_no_shared_singleton():
    """A bare mode honors an explicit group_size, and neither a bare mode nor a
    named scheme hands back the shared module-level singleton (QNT-3)."""
    from zeromodels.quantization.quant_config import SCHEMES, resolve_config

    assert resolve_config("int4", group_size=128).group_size == 128
    assert resolve_config("int4") is not resolve_config("int4")
    named = resolve_config("int4-g128")
    assert named is not SCHEMES["int4-g128"] and named.group_size == 128


def test_weight_only_zm_quantizer_carries_full_recipe():
    """The weight-only ZmQuantizer applies the whole repo recipe (group_size,
    skip_modules, quantize_embeddings), not just the method name, and its return
    (a new functional clone) is what gets quantized (QNT-3 + QNT-2)."""
    import numpy as np

    from zeromodels.quantization import get_zm_quantizer
    from zeromodels.quantization.quantized_layers import QuantizedDense

    def toy():
        inp = keras.layers.Input((16,))
        h = keras.layers.Dense(16, name="d1")(inp)
        out = keras.layers.Dense(16, name="d2")(h)
        m = keras.Model(inp, out)
        m(np.zeros((1, 16), "float32"))
        return m

    qc = {
        "quant_method": "int4",
        "group_size": 64,
        "skip_modules": ["d2"],
        "quantize_embeddings": False,
    }
    model = get_zm_quantizer(qc).preprocess_model(toy())
    cfg = model._quantization_config
    assert cfg.group_size == 64 and cfg.skip_modules == ("d2",)
    assert cfg.quantize_embeddings is False
    assert isinstance(model.get_layer("d1"), QuantizedDense)  # d1 quantized
    assert type(model.get_layer("d2")).__name__ == "Dense"  # d2 skipped


def test_fused_experts_predicate_is_layout_strict():
    """Only the (E, 2I, H) / (E, H, I) contiguous-SwiGLU banks QuantizedExperts
    actually replicates are matched; GPT-OSS and Llama4 layouts are left float
    instead of silently mis-quantized (QNT-4)."""
    from zeromodels.models.gpt_oss.gpt_oss_layers import GptOssExperts
    from zeromodels.models.llama4.llama4_layers import Llama4Experts
    from zeromodels.models.qwen3_moe.qwen3_moe_layers import Qwen3MoeExperts
    from zeromodels.quantization.quantize import _is_experts_skeleton, _is_fused_experts

    e, i, h = 4, 32, 16
    qwen = Qwen3MoeExperts(e, h, i)
    qwen.build(None)
    gpt = GptOssExperts(e, h, i)
    gpt.build(None)
    llama = Llama4Experts(e, h, i)
    llama.build(None)
    assert _is_fused_experts(qwen) and not _is_fused_experts(gpt)
    assert not _is_fused_experts(llama)
    # unbuilt (skeleton) predicate: same exclusions
    assert _is_experts_skeleton(Qwen3MoeExperts(e, h, i))
    assert not _is_experts_skeleton(GptOssExperts(e, h, i))
    assert not _is_experts_skeleton(Llama4Experts(e, h, i))


def test_dequantize_rebuilds_experts_with_correct_activation():
    """dequantize round-trips a fused-expert bank back to its original class, and
    the quantized Gemma bank uses tanh-approx GeGLU so it matches (QNT-4 + QNT-6)."""
    import numpy as np
    from keras import ops

    from zeromodels.models.gemma4.gemma4_layers import Gemma4Experts
    from zeromodels.quantization.quantized_layers import QuantizedExperts

    e, i, h = 4, 32, 16
    bank = Gemma4Experts(e, h, i)
    bank.build(None)
    for w in bank.weights:
        w.assign(np.random.default_rng(0).standard_normal(w.shape).astype("float32") * 0.1)
    q = QuantizedExperts.from_experts(bank, "int8", 32, "gelu")
    revived = q.to_experts()
    assert type(revived).__name__ == "Gemma4Experts"
    # same int weights on both sides -> a mismatched (exact-erf) gelu would diverge
    x = np.random.default_rng(1).standard_normal((5, h)).astype("float32")
    rw = np.abs(np.random.default_rng(2).standard_normal((5, e)).astype("float32"))
    diff = float(np.max(np.abs(ops.convert_to_numpy(q(x, rw)) - ops.convert_to_numpy(revived(x, rw)))))
    assert diff < 1e-4, diff


def test_dequantize_functional_survives_python_lambda():
    """dequantize_model must not crash on a model that quantized fine but holds an
    un-cloneable python-lambda Lambda (QNT-6)."""
    import numpy as np

    from zeromodels.quantization.quantize import dequantize_model, quantize_model

    inp = keras.layers.Input((16,))
    h = keras.layers.Dense(16, name="dd")(inp)
    h = keras.layers.Lambda(lambda t: t * 2.0)(h)
    out = keras.layers.Dense(16, name="oo")(h)
    model = keras.Model(inp, out)
    model(np.zeros((1, 16), "float32"))
    model = quantize_model(model, "int8")
    model = dequantize_model(model)  # must not raise
    assert any(type(layer).__name__ == "Dense" for layer in model.layers)


def test_int4_storage_spec_rejects_odd_dim_and_warns_on_collapse():
    """int4 storage sizing fails fast on an odd contracting dim (rather than mid
    stream) and warns when group_size cannot divide the axis (adjacent to QNT-6)."""
    from zeromodels.quantization.int4_quantize import Int4Quantizer

    with pytest.raises(ValueError, match="even contracting dim"):
        Int4Quantizer(32).storage_spec((7, 8), axis=0)
    with pytest.warns(UserWarning, match="does not divide"):
        Int4Quantizer(32).storage_spec((8198, 4), axis=0)  # 8198 = 2 * 4099


@pytest.mark.serialization
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_keras_serialization_roundtrip(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_SERIALIZATION_TF:
        pytest.skip(f"{model_name} causes TF backend segfault during serialization")

    config = MODEL_TEST_CONFIGS[model_name]
    model = get_cached_model(config)

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
