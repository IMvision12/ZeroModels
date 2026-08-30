import gc
import hashlib
import os

import pytest


def _node_model_name(node):
    """The ``model_name`` parametrization of a test node, or None if it has none."""
    callspec = getattr(node, "callspec", None)
    if callspec is None:
        return None
    return callspec.params.get("model_name")


def _parse_shard(spec):
    """Parse a ``k/n`` shard spec (1-based k) into a 0-based (index, count)."""
    k_str, n_str = spec.split("/")
    k, n = int(k_str), int(n_str)
    if not 1 <= k <= n:
        raise pytest.UsageError(f"--shard {spec!r}: need 1 <= k <= n")
    return k - 1, n


def pytest_collection_modifyitems(config, items):
    """Group every test of a model together, then optionally keep one shard.

    A stable sort by ``model_name`` puts all of one model's tests (across the
    backend-compat / serialization / saving / data-format files) back to back,
    so :func:`get_cached_model` can hand out one built model to that model's
    read-only tests and it can be released in a single teardown when the model
    changes. Tests with no ``model_name`` keep their original order as one
    leading group.

    With ``--shard k/n`` the models are round-robin assigned to ``n`` shards and
    only shard ``k`` is kept. Sharding is BY MODEL (not by test) so a model's
    whole test group stays on one shard: the per-model build-once reuse holds,
    and no model is built on more than one CI runner. Non-model tests are
    distributed by a stable hash of their node id so each runs on exactly one
    shard. Splitting a backend's models across parallel jobs is what brings the
    slow JAX / TF legs under the per-job time cap (stacks with the reuse above).
    """
    original = {id(item): i for i, item in enumerate(items)}
    items.sort(key=lambda item: (_node_model_name(item) or "", original[id(item)]))

    spec = config.getoption("shard")
    if not spec:
        return
    shard_index, shard_count = _parse_shard(spec)
    if shard_count == 1:
        return
    model_names = sorted({n for n in map(_node_model_name, items) if n is not None})
    model_shard = {name: i % shard_count for i, name in enumerate(model_names)}

    def item_shard(item):
        name = _node_model_name(item)
        if name is not None:
            return model_shard[name]
        digest = hashlib.md5(item.nodeid.encode()).hexdigest()
        return int(digest, 16) % shard_count

    selected, deselected = [], []
    for item in items:
        (selected if item_shard(item) == shard_index else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


_CURRENT_MODEL = ["\x00unset"]


def _flush_backend_state():
    try:
        from tests.base.model_test_registry import clear_model_cache

        clear_model_cache()
    except Exception:
        pass
    try:
        import keras

        keras.utils.clear_session()
        if keras.config.backend() == "jax":
            import jax

            jax.clear_caches()
    except Exception:
        pass
    gc.collect()


@pytest.fixture(autouse=True)
def _release_backend_state(request):
    """Release the previous model's build + XLA compilation when the model changes.

    Each parametrized model triggers fresh XLA / TF tracing; the JIT cache,
    compiled HLO, and dead layers otherwise accumulate across the 300+ tests and
    the JAX matrix entry hits the ubuntu-latest 7 GB RAM / 60 min cap (SIGTERM,
    exit 143). The old fix cleared after *every* test, which also threw away the
    build + compile so each of a model's ~10 tests paid them again (hours on JAX).

    Because tests are now grouped per model, clearing only when the model changes
    keeps peak memory at ~one model *and* lets that model's build + compile be
    reused across its tests. Non-model tests (``model_name is None``) clear every
    time, preserving the original bounded-memory behavior for them.
    """
    # ZM_LEGACY_CLEAR=1 restores the old clear-after-every-test behavior, for
    # A/B timing against the per-model reuse (pair with ZM_NO_MODEL_CACHE=1).
    if os.environ.get("ZM_LEGACY_CLEAR") == "1":
        yield
        _flush_backend_state()
        return
    name = _node_model_name(request.node)
    if name is None or name != _CURRENT_MODEL[0]:
        _flush_backend_state()
        _CURRENT_MODEL[0] = name
    yield


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Keras backend to use: torch, tensorflow, jax, numpy",
    )
    parser.addoption(
        "--data-format",
        action="store",
        default=None,
        help="Image data format: channels_first, channels_last",
    )
    parser.addoption(
        "--shard",
        action="store",
        dest="shard",
        default=None,
        help=(
            "Run only shard k of n (format 'k/n', 1-based), sharded BY MODEL so "
            "a model's tests stay together. Splits a backend's models across "
            "parallel CI jobs."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "serialization: serialization roundtrip tests")
    config.addinivalue_line("markers", "saving: model save/load tests")
    config.addinivalue_line("markers", "data_format: channels first/last tests")
    config.addinivalue_line(
        "markers",
        "link_validation: weight URL + download tests (requires network)",
    )
    config.addinivalue_line("markers", "slow: slow tests excluded from default runs")
    config.addinivalue_line("markers", "gpu: tests that require GPU (skipped on CI)")


def is_gpu_available():
    import keras

    backend = keras.config.backend()
    if backend == "tensorflow":
        try:
            import tensorflow as tf

            return len(tf.config.list_physical_devices("GPU")) > 0
        except ImportError:
            return False
    if backend == "torch":
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
    return False


def skip_if_no_gpu(reason="This test requires GPU"):
    return pytest.mark.skipif(not is_gpu_available(), reason=reason)


def skip_tf_channels_first():
    import keras

    if keras.config.backend() == "tensorflow" and not is_gpu_available():
        pytest.skip("TF channels_first conv2d requires GPU (cuDNN)")


def skip_numpy_backend():
    import keras

    if keras.config.backend() == "numpy":
        pytest.skip("numpy backend doesn't support this operation")


@pytest.fixture
def backend():
    """Return the current Keras backend name."""
    return os.environ.get("KERAS_BACKEND", "torch")
