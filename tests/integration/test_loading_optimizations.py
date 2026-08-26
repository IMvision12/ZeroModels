import json

import keras
import numpy as np
import pytest
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.conversion import converted_cache
from kerasformers.conversion.hf_download_utils import LazyStateDict


@keras.saving.register_keras_serializable(package="kerasformers_tests")
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
    from kerasformers.quantization.mxfp4_quantize import (
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
