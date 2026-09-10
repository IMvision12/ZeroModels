import json

import keras
import numpy as np
import pytest
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_mixin import WeightLoadingMixin
from zeromodels.conversion import converted_cache
from zeromodels.conversion.hf_download_utils import LazyStateDict


@keras.saving.register_keras_serializable(package="zeromodels_tests")
class _CacheToy(BaseModel):
    """A minimal functional model for exercising the converted-weight cache."""

    def __init__(self, dim=32, depth=2, vocab=64, name=None, **kwargs):
        ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        embedding = layers.Embedding(vocab, dim, name="embedding")
        blocks = [layers.Dense(dim, name=f"block_{i}") for i in range(depth)]
        head = layers.Dense(vocab, name="head")
        x = embedding(ids)
        for block in blocks:
            x = block(x)
        outputs = head(x)
        super().__init__(
            inputs={"input_ids": ids},
            outputs=outputs,
            name=name or type(self).__name__,
            **kwargs,
        )
        self.embedding = embedding
        self.blocks = blocks
        self.head = head
        self.dim = dim
        self.depth = depth
        self.vocab = vocab

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "vocab": self.vocab,
                "name": self.name,
            }
        )
        return config


def test_converted_cache_streams_bounded_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(converted_cache, "SHARD_LIMIT_BYTES", 1024)
    rng = np.random.default_rng(5)
    inputs = {"input_ids": np.array([[2, 4, 8, 16]], dtype="int32")}
    model = _CacheToy()
    for weight in model.weights:
        weight.assign(rng.standard_normal(tuple(weight.shape)).astype("float32"))
    expected = ops.convert_to_numpy(model(inputs))

    converted_cache.save_converted(model, str(tmp_path), None)
    with open(tmp_path / "meta.json") as f:
        meta = json.load(f)
    assert meta["cache_format"] == converted_cache.CACHE_FORMAT_VERSION
    assert len(meta["shards"]) > 1

    restored = converted_cache.load_converted(str(tmp_path), None, None)
    actual = ops.convert_to_numpy(restored(inputs))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="load_dtype"):
        converted_cache.load_converted(str(tmp_path), None, "float16")


def test_cache_key_includes_source_revision(monkeypatch):
    class _Model:
        __module__ = "test_models"
        __qualname__ = "Model"

    monkeypatch.setattr(
        converted_cache,
        "_source_identity",
        lambda _: {"kind": "hf", "repo": "org/model", "revision": "a"},
    )
    first = converted_cache.cache_dir(_Model, "hf:org/model", "int4", None, {})
    monkeypatch.setattr(
        converted_cache,
        "_source_identity",
        lambda _: {"kind": "hf", "repo": "org/model", "revision": "b"},
    )
    second = converted_cache.cache_dir(_Model, "hf:org/model", "int4", None, {})
    assert first != second


def test_lazy_state_dict_reads_one_tensor_at_a_time(tmp_path):
    from safetensors.numpy import save_file

    s1 = tmp_path / "one.safetensors"
    s2 = tmp_path / "two.safetensors"
    save_file({"a": np.array([1]), "b": np.array([2])}, str(s1))
    save_file({"c": np.array([3])}, str(s2))

    # The sharded construction path: name -> the shard file that holds it.
    state = LazyStateDict({"a": str(s1), "b": str(s1), "c": str(s2)})
    assert len(state) == 3
    assert set(state) == {"a", "b", "c"}
    # Membership must not read a tensor (the reason __contains__ is overridden):
    # it is answered from the key index, and "missing" is simply absent.
    assert "a" in state and "missing" not in state
    np.testing.assert_array_equal(state["b"], np.array([2]))
    np.testing.assert_array_equal(state["c"], np.array([3]))
    with pytest.raises(KeyError):
        state["missing"]
    state.close()  # drops the mmap handles; safe to call after conversion.


def test_mxfp4_quantize_is_exact_inverse_of_dequant():
    """``quantize_to_mxfp4`` inverts ``dequantize_mxfp4`` value-for-value on the
    MXFP4 lattice, and rounds arbitrary floats to the nearest FP4 grid point.

    The dequant is a byte-exact port of HF ``convert_moe_packed_tensors``, so a
    zero round-trip pins the quantizer to the same values HF's downcast produces
    on every representable point (no GPU triton kernel needed to check).
    """
    from zeromodels.quantization.mxfp4_quantize import (
        FP4_VALUES,
        dequantize_mxfp4,
        quantize_to_mxfp4,
    )

    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (4, 64, 4, 16), dtype=np.uint8)
    scales = rng.integers(110, 140, (4, 64, 4), dtype=np.uint8)
    lattice = ops.convert_to_numpy(dequantize_mxfp4(blocks, scales))
    b2, s2 = quantize_to_mxfp4(ops.convert_to_tensor(lattice))
    reconstructed = ops.convert_to_numpy(dequantize_mxfp4(b2, s2))
    np.testing.assert_array_equal(lattice, reconstructed)

    # Arbitrary floats: dequant(quant(w)) is the closest lattice point at the
    # block's chosen scale (round-to-nearest, matching HF).
    w = (rng.standard_normal((8, 256)) * 3).astype("float32")
    bq, sq = quantize_to_mxfp4(ops.convert_to_tensor(w))
    wq = ops.convert_to_numpy(dequantize_mxfp4(bq, sq))
    fp4 = np.asarray(FP4_VALUES, "float32")
    scale = 2.0 ** (ops.convert_to_numpy(sq).astype(np.int32) - 127)
    candidates = fp4[None, None, None, :] * scale[..., None, None]
    blocks_of_w = w.reshape(8, 256 // 32, 32)
    nearest = np.take_along_axis(
        candidates,
        np.argmin(np.abs(blocks_of_w[..., None] - candidates), axis=-1)[..., None],
        axis=-1,
    )[..., 0].reshape(8, 256)
    np.testing.assert_allclose(wq, nearest)


def _timm_families():
    """Family dir names whose ``*_model.py`` defines a ``transfer_from_timm``."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "zeromodels" / "models"
    return sorted(
        p.parent.name
        for p in root.glob("*/*_model.py")
        if "def transfer_from_timm" in p.read_text(encoding="utf-8", errors="ignore")
    )


@pytest.mark.parametrize("family", _timm_families())
def test_timm_conversion_path_is_wired(family):
    """Each timm-ported family exposes its converter's arch table and builds through
    the on-the-fly ``hf:timm/...`` path (guards ZOO-1: the timm ``from_hf`` branch
    resolved the variant from ``BASE_MODEL_CONFIG``, which these classes stopped
    setting when arch moved into the converters, so both documented calls raised).

    Network-free: ``load_weights=False`` exercises variant inference + build without
    downloading a checkpoint.
    """
    import importlib
    import inspect

    module = importlib.import_module(f"zeromodels.models.{family}")
    cls = next(
        (
            obj
            for name in getattr(module, "__all__", dir(module))
            if inspect.isclass(obj := getattr(module, name, None))
            and getattr(obj, "HF_MODEL_TYPE", "x") is None
            and hasattr(obj, "timm_model_configs")
            and obj.timm_model_configs()
        ),
        None,
    )
    assert cls is not None, f"{family}: no timm-loadable class with a resolvable arch"

    configs = cls.timm_model_configs()
    assert configs, f"{cls.__name__}.timm_model_configs() is empty"
    variant = sorted(configs)[0]
    # `from_weights("hf:<repo>", variant=...)`: explicit variant.
    assert isinstance(
        cls.from_weights("hf:test/repo", variant=variant, load_weights=False), cls
    )
    # `from_weights("hf:timm/<id>")`: variant inferred from the repo tail.
    assert isinstance(
        cls.from_weights(f"hf:timm/{variant}.pretrained", load_weights=False), cls
    )


class _QuantBlk(layers.Layer):
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.q = layers.Dense(dim, name="q")
        self.o = layers.Dense(dim, name="o")

    def call(self, x):
        return self.o(self.q(x))


class _SubclassedQuantToy(WeightLoadingMixin, keras.Model):
    """Unbuilt subclassed model whose converter builds by iterating model.weights."""

    def __init__(self, n=64, dim=32, depth=2, **kw):
        super().__init__(**kw)
        self.emb = layers.Embedding(n, dim, name="token_embedding")
        self.blocks = [_QuantBlk(dim, name=f"block_{i}") for i in range(depth)]
        self.lm_head = layers.Dense(n, use_bias=False, name="lm_head")

    def call(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.lm_head(h)

    @classmethod
    def transfer_from_hf(cls, model, sd):
        from zeromodels.conversion.weight_transfer_util import transfer_weights

        if not model.built or not model.weights:
            model(np.array([[0, 1, 2, 3]]))
        for w in model.weights:
            key = w.path.split("/", 1)[1].replace("/", ".")
            key = key.replace("token_embedding.embeddings", "emb").replace(
                "kernel", "weight"
            )
            transfer_weights(w.path, w, sd[key])


class _LazyTowerQuantToy(_SubclassedQuantToy):
    """A subclassed model that materializes lazily via build_for_transfer."""

    def build_for_transfer(self):
        self(np.array([[0, 1, 2, 3]]))


def _quant_toy_state_dict():
    rng = np.random.default_rng(0)
    ref = _SubclassedQuantToy(name="toy")
    ref(np.array([[3, 9, 40, 60]]))
    sd = {}
    for w in ref.weights:
        key = w.path.split("/", 1)[1].replace("/", ".")
        key = key.replace("token_embedding.embeddings", "emb").replace(
            "kernel", "weight"
        )
        shape = tuple(w.shape)
        if (key.endswith(".weight") and "block" in key) or key.endswith("head.weight"):
            shape = (shape[1], shape[0])  # HF stores Dense weight transposed
        sd[key] = rng.standard_normal(shape).astype("float32")
    return sd


def test_quantized_transfer_uses_no_float_path():
    """``_quantized_transfer`` streams an unbuilt subclassed model straight into int
    storage instead of materializing the full float model then quantizing (guards
    LOAD-1: the no-float path was dead, so ``from_weights(..., quantization=...)``
    built the whole float checkpoint before quantizing and OOM'd on the machines
    the flag exists for).

    Network-free: exercises the classmethod directly with an in-memory state dict.
    """
    from zeromodels.quantization import quantize_model
    from zeromodels.quantization.quantized_layers import (
        QuantizedDense,
        QuantizedEmbedding,
    )

    x = np.array([[3, 9, 40, 60]])
    sd = _quant_toy_state_dict()

    # Reference: build float, then quantize in place.
    ref = _SubclassedQuantToy(name="toy")
    _SubclassedQuantToy.transfer_from_hf(ref, sd)
    quantize_model(ref, "int8")
    y_ref = ops.convert_to_numpy(ref(x))

    # No-float path: an unbuilt instance streamed straight into int storage.
    model = _SubclassedQuantToy(name="toy")
    assert not model.built
    took_no_float = _SubclassedQuantToy._quantized_transfer(model, sd, "int8", False)
    assert took_no_float is True
    # Recorded config -> from_weights skips the post-hoc quantize_model.
    assert getattr(model, "_quantization_config", None) is not None
    assert model._quantization_config.mode == "int8"
    assert isinstance(model.get_layer("token_embedding"), QuantizedEmbedding)
    assert isinstance(model.blocks[0].q, QuantizedDense)
    # Byte-identical to load-then-quantize.
    y = ops.convert_to_numpy(model(x))
    assert float(np.max(np.abs(y - y_ref))) == 0.0

    # quantization=None keeps the plain float transfer (no config, float layers).
    plain = _SubclassedQuantToy(name="toy")
    assert _SubclassedQuantToy._quantized_transfer(plain, sd, None, False) is False
    assert getattr(plain, "_quantization_config", None) is None
    assert isinstance(plain.blocks[0].q, layers.Dense)


def test_quantized_transfer_skips_no_float_for_lazy_towers():
    """A ``build_for_transfer`` tower (lazy VLM/ASR sublayers) is not eligible for the
    no-float skeleton, so ``_quantized_transfer`` runs a float transfer and leaves it
    unquantized for ``from_weights`` to quantize afterwards (LOAD-1 guardrail)."""
    sd = _quant_toy_state_dict()
    tower = _LazyTowerQuantToy(name="toy")
    took_no_float = _LazyTowerQuantToy._quantized_transfer(tower, sd, "int8", False)
    assert took_no_float is False
    assert getattr(tower, "_quantization_config", None) is None
    assert isinstance(tower.blocks[0].q, layers.Dense)
