"""Cross-backend output parity: the same tiny model must produce the same forward
output on torch / jax / tensorflow.

The library's promise is "pure Keras 3, identical across backends", but every other
suite runs under a single ``KERAS_BACKEND`` and never compares across them. This one
does it the snapshot way: build each registry model with **deterministic
numpy-seeded weights** (keyed by weight path, so every backend reconstructs the same
weights from code alone -- no shared weight file) and a **fixed-seed random input**,
run the forward, and compare a fingerprint of the output against a committed golden.
The golden is generated once on a reference backend and shared, so a backend whose
output drifts (a JAX tracing bug, a TF layout bug, a scrambled reshape) fails here on
whichever backend diverged.

Random (not ones) input on purpose: a uniform input gives normalization layers ~0
variance, and ``1/sqrt(var+eps)`` then amplifies float noise into false failures
(see the OneFormer note in project_channels_first_dataformat).

Regenerate on the reference backend after an intentional change, then review the diff:

    ZM_UPDATE_PARITY=1 KERAS_BACKEND=torch pytest tests/integration/test_cross_backend_parity.py
"""

import hashlib
import json
import os
import pathlib

import keras
import numpy as np
import pytest
from keras import ops

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())
GOLDEN_PATH = (
    pathlib.Path(__file__).parent.parent / "fixtures" / "cross_backend_parity.json"
)
# Cross-backend float noise is ~1e-5..1e-3; a real divergence (scramble / wrong op /
# layout bug) is O(1). 2% cleanly separates the two, matching the channels_first test.
# ATOL is an absolute floor (numpy-allclose style) so a near-zero output value does not
# inflate the relative metric into a false failure.
RTOL = float(os.environ.get("ZM_PARITY_RTOL", "2e-2"))
ATOL = float(os.environ.get("ZM_PARITY_ATOL", "1e-4"))
SAMPLE = 64  # stride-sampled output values stored per tensor

# Models that cannot be compared cross-backend (a backend can't build/run them at all).
SKIP = {
    # TF CPU segfaults in tf.matmul for the large SAM models (known TF bug; also
    # skipped by test_backend_compatibility).
    "SAMModel",
    "SAMPromptableSegment",
    "SAM2PromptableSegment",
}

# Models that build and run everywhere but whose output legitimately differs across
# backends for an understood, non-bug reason (not a real-use portability defect). Keyed
# by the reason.
KNOWN_DIVERGENT = {
    # RT-DETR selects decoder queries with ops.top_k over encoder scores. With the tiny
    # synthetic parity weights those scores are near-tied, and top_k breaks ties by index
    # differently on torch vs jax, so a different (equally-scored) set of tokens is
    # picked -> pred_boxes differ while the near-uniform class logits still match. Real
    # checkpoints have well-separated scores, so top_k is deterministic in practice.
    "RTDETRDetect": "top_k query-selection tie-sensitivity under synthetic weights",
    "RTDETRV2Detect": "top_k query-selection tie-sensitivity under synthetic weights",
    # D-FINE uses the same ops.top_k(max_sc, k=num_queries) query selection as RT-DETR.
    # (GroundingDino / RF-DETR also use top_k but their scores are not tied here, so they
    # stay in the checked set.)
    "DFineDetect": "top_k query-selection tie-sensitivity under synthetic weights",
}


def _updating():
    return os.environ.get("ZM_UPDATE_PARITY") == "1"


def _skip_tf_incompatible(name):
    if BACKEND == "tensorflow" and name in SKIP:
        try:
            import tensorflow as tf

            if not tf.config.list_physical_devices("GPU"):
                pytest.skip(f"{name}: TF CPU cannot run this model")
        except ImportError:
            pass


def _assign_deterministic_weights(model):
    """Overwrite every float weight with numpy-seeded values keyed by its path.

    NumPy's RNG is identical on every backend, so each backend rebuilds the *same*
    weights from code alone -- no shared checkpoint. Non-float buffers (int position
    ids, arange tables) are already deterministic and left as-is.
    """
    for w in model.weights:
        dtype = keras.backend.standardize_dtype(w.dtype)
        if "float" not in dtype:
            continue
        seed = int(hashlib.md5(w.path.encode()).hexdigest()[:8], 16)
        value = np.random.RandomState(seed).standard_normal(tuple(w.shape)) * 0.02
        # BatchNorm's moving_variance feeds 1/sqrt(var+eps); a negative random value
        # would produce NaN, so keep any variance buffer comfortably positive.
        if "variance" in w.path.lower():
            value = np.abs(value) + 0.5
        w.assign(value.astype("float32"))


def _vocab_size(config):
    """Best-effort token vocab size from the config (top-level or a nested sub-config)."""
    kwargs = config["init_kwargs"]
    vocab = kwargs.get("vocab_size")
    if vocab is None:
        for value in kwargs.values():
            if isinstance(value, dict) and value.get("vocab_size"):
                vocab = value["vocab_size"]
                break
    return vocab or 30


def _parity_input(config, model):
    """Deterministic input: create_test_input's structure with float tensors replaced by
    fixed-seed random values, and ``input_ids`` replaced by *varied* (non-degenerate)
    token ids. Everything is numpy-derived, so it is byte-identical across backends.

    Varied ids matter: create_test_input returns all-ones ids, which make every token
    position identical, so per-token heads (QnA, token-classify) emit a near-constant
    output dominated by cross-backend float noise -- a false divergence. Varied ids give
    the head real signal (same OneFormer normalization-amplification lesson).
    """
    x = create_test_input(config, model=model)
    rng = np.random.RandomState(1234)
    safe_vocab = max(2, min(_vocab_size(config) - 1, 50))
    # VLM input_ids carry image-placeholder tokens whose count must match the pixels, so
    # only vary ids for plain text models.
    vary_ids = not config.get("multimodal_vlm")

    def convert(key, v):
        arr = np.asarray(ops.convert_to_numpy(v))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (rng.standard_normal(arr.shape) * 0.5).astype("float32")
        elif key == "input_ids" and vary_ids:
            arr = (np.arange(arr.size).reshape(arr.shape) % safe_vocab + 1).astype(
                arr.dtype
            )
        return ops.convert_to_tensor(arr)

    if isinstance(x, dict):
        return {k: convert(k, v) for k, v in x.items()}
    return convert(None, x)


def _flatten(output):
    # Sort dict outputs by key: keras dict-output ordering is backend-dependent (torch may
    # yield {end_logits, start_logits}, jax {start_logits, end_logits}), so a positional
    # compare of list(values()) would pit start against end and false-fail. Sorting keys
    # makes position [i] the same named tensor on every backend.
    if isinstance(output, dict):
        return [output[k] for k in sorted(output)]
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def _fingerprint(tensor):
    arr = np.asarray(ops.convert_to_numpy(tensor))
    flat = arr.astype("float64").ravel()
    n = flat.size
    idx = np.linspace(0, n - 1, min(n, SAMPLE)).astype(int) if n else np.array([], int)
    return {"shape": list(arr.shape), "sample": [round(float(flat[i]), 6) for i in idx]}


def _fingerprints(name):
    config = MODEL_TEST_CONFIGS[name]
    keras.utils.clear_session()
    # Reset all RNGs so any layer that seeds itself from global state at build time
    # (and any inference-time draw) is reproducible across processes and backends.
    keras.utils.set_random_seed(0)
    model = import_model_class(config)(**config["init_kwargs"])
    _assign_deterministic_weights(model)
    output = model(_parity_input(config, model), training=False)
    return [_fingerprint(t) for t in _flatten(output)]


def _load_golden():
    if GOLDEN_PATH.exists():
        return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return {}


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_cross_backend_output_parity(model_name):
    if model_name in SKIP:
        _skip_tf_incompatible(model_name)
    if model_name in KNOWN_DIVERGENT and not _updating():
        pytest.skip(
            f"{model_name}: known cross-backend diff ({KNOWN_DIVERGENT[model_name]})"
        )
    original = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_last")
        current = _fingerprints(model_name)
    finally:
        keras.config.set_image_data_format(original)

    golden = _load_golden()
    if _updating():
        data = golden
        data[model_name] = current
        GOLDEN_PATH.write_text(
            json.dumps(dict(sorted(data.items())), indent=1) + "\n", encoding="utf-8"
        )
        pytest.skip(f"parity golden updated: {model_name}")

    if model_name not in golden:
        pytest.fail(
            f"no cross-backend parity golden for {model_name}. Generate on the "
            f"reference backend: ZM_UPDATE_PARITY=1 KERAS_BACKEND=torch pytest "
            f"tests/integration/test_cross_backend_parity.py"
        )

    expected = golden[model_name]
    assert len(expected) == len(current), (
        f"{model_name}: output count {len(current)} != golden {len(expected)}"
    )
    for i, (exp, cur) in enumerate(zip(expected, current)):
        assert exp["shape"] == cur["shape"], (
            f"{model_name}[{i}]: shape {cur['shape']} != golden {exp['shape']} "
            f"on {BACKEND}"
        )
        g = np.asarray(exp["sample"], dtype="float64")
        c = np.asarray(cur["sample"], dtype="float64")
        if g.size == 0:
            continue
        abs_diff = float(np.abs(g - c).max())
        # numpy-allclose style: pass if within an absolute floor OR a relative bound.
        assert abs_diff <= ATOL + RTOL * float(np.abs(g).max()), (
            f"{model_name}[{i}]: {BACKEND} output diverges from the golden "
            f"(max|diff|={abs_diff:.3e}, tol={ATOL + RTOL * float(np.abs(g).max()):.3e}) "
            f"-- a backend-specific op/layout/trace bug"
        )
