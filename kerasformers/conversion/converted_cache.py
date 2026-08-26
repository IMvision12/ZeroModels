import hashlib
import json
import os
import re
import warnings

import keras
import numpy as np

from kerasformers.conversion.hf_download_utils import LazyStateDict

# Cache files are intentionally much smaller than the maximum safetensors shard
# size.  ``save_file`` needs one complete shard in memory, so a 5 GB shard would
# defeat the cache's low-memory load path.
SHARD_LIMIT_BYTES = 512 * 1024**2
CACHE_FORMAT_VERSION = 2
CACHE_LAYOUT_VERSION = 2
CONVERTER_VERSION = 1
QUANTIZATION_FORMAT_VERSION = 1


def cache_root():
    """Root directory for cached converted models.

    ``$KERASFORMERS_HOME/converted`` (else ``~/.cache/kerasformers/converted``),
    self-managed like the HF cache. On an ephemeral box (Colab), point
    ``KERASFORMERS_HOME`` at a persistent mount (Drive) to keep the benefit
    across sessions.
    """
    home = os.environ.get(
        "KERASFORMERS_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "kerasformers"),
    )
    return os.path.join(home, "converted")


def quant_id(quantization):
    """A stable, JSON-safe identity for a quantization spec (or float)."""
    if quantization is None:
        return "float"
    to_dict = getattr(quantization, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(quantization)


def _source_identity(identifier):
    """Return the immutable source identity used in a converted-cache key.

    A moving Hub branch such as ``main`` is resolved to its commit SHA.  A
    transient Hub failure does not make loading impossible: the unresolved
    identity simply cannot collide with a cache created for a known revision.
    """
    if not identifier.startswith("hf:"):
        return {"kind": "release", "identifier": identifier}

    repo_id = identifier[len("hf:") :]
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, timeout=5, token=os.environ.get("HF_TOKEN"))
        return {"kind": "hf", "repo": repo_id, "revision": info.sha}
    except Exception:
        # An offline user can still safely reuse the immutable snapshot already
        # selected by Hugging Face's local cache.
        try:
            from huggingface_hub import try_to_load_from_cache

            config_path = try_to_load_from_cache(repo_id, "config.json")
            if isinstance(config_path, str):
                parts = os.path.normpath(config_path).split(os.sep)
                snapshot_index = parts.index("snapshots")
                return {
                    "kind": "hf",
                    "repo": repo_id,
                    "revision": parts[snapshot_index + 1],
                }
        except (ImportError, OSError, ValueError):
            pass
        # Do not reuse a known-revision cache when we could not determine the
        # source. This keeps an unavailable source correct rather than stale.
        return {"kind": "hf", "repo": repo_id, "revision": "unresolved"}


def _json_hash(value):
    blob = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_dir(cls, identifier, quantization, load_dtype, kwargs):
    """Deterministic per-config cache directory for one converted model.

    Keyed on the model class, immutable source revision, converter/cache layout,
    quantization recipe, backend/dtype and architecture arguments. Excludes
    ``attn_implementation``, which is reapplied at build and does not change the
    stored weights.
    """
    payload = {
        "cache_format": CACHE_FORMAT_VERSION,
        "cache_layout": CACHE_LAYOUT_VERSION,
        "converter": CONVERTER_VERSION,
        "quantizer": QUANTIZATION_FORMAT_VERSION,
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source": _source_identity(identifier),
        "quantization": quant_id(quantization),
        "load_dtype": load_dtype,
        "backend": keras.backend.backend(),
        "kwargs": {k: kwargs[k] for k in sorted(kwargs)},
    }
    digest = _json_hash(payload)[:16]
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)
    return os.path.join(cache_root(), f"{safe_id}.{digest}")


def is_cached(directory):
    """True if ``directory`` holds a complete cache (``meta.json`` is last-written)."""
    return os.path.exists(os.path.join(directory, "meta.json"))


def cache_supported(cls, quantization):
    """Whether ``cls`` can be cache-reloaded from a serialized skeleton.

    Functional models round-trip only as float: their built graph can't be
    re-quantized from a serialized skeleton, so functional + quantization is
    bypassed.
    """
    from kerasformers.base import BaseModel

    if issubclass(cls, BaseModel):
        return quantization is None
    return False


def save_converted(model, directory, quantization, load_dtype=None):
    """Cache a converted model's final weights + a rebuild recipe.

    Stores ``model.weights`` as index-keyed sharded safetensors (≤512 MB shards)
    plus ``meta.json`` (serialized config, quant id, per-weight keys + shapes).
    Weights are keyed by position: a functional model's auto-numbered layer names
    can differ across a reload, so index order is the stable key. Written
    meta.json last so a partial write is never seen as a cache hit.
    """
    from safetensors.numpy import save_file

    os.makedirs(directory, exist_ok=True)
    weights = list(model.weights)

    keys = [f"{i:06d}" for i in range(len(weights))]

    shards = []
    current, current_bytes, shard_idx = {}, 0, 0
    for i, weight in enumerate(weights):
        # Materialize and retain only the active shard.  The old implementation
        # constructed a NumPy copy of *every* model weight before writing the
        # first byte, which could double host memory for a large quantized model.
        arr = np.ascontiguousarray(keras.ops.convert_to_numpy(weight))
        if current and current_bytes + arr.nbytes > SHARD_LIMIT_BYTES:
            name = f"weights-{shard_idx:05d}.safetensors"
            save_file(current, os.path.join(directory, name))
            shards.append(name)
            current, current_bytes = {}, 0
            shard_idx += 1
        current[f"{i:06d}"] = arr
        current_bytes += arr.nbytes
    name = f"weights-{shard_idx:05d}.safetensors"
    save_file(current, os.path.join(directory, name))
    shards.append(name)

    config = keras.saving.serialize_keras_object(model)

    meta = {
        "cache_format": CACHE_FORMAT_VERSION,
        "cache_layout": CACHE_LAYOUT_VERSION,
        "converter": CONVERTER_VERSION,
        "quantizer": QUANTIZATION_FORMAT_VERSION,
        "backend": keras.backend.backend(),
        "load_dtype": load_dtype,
        "config": config,
        "architecture_hash": _json_hash(config),
        "quantization": quant_id(quantization),
        "keying": "index",
        "keys": keys,
        "shapes": [list(w.shape) for w in weights],
        "count": len(weights),
        "shards": shards,
    }
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_converted(directory, quantization, load_dtype):
    """Rebuild a model from a cache directory and stream its weights back.

    Deserializes the config to the model, then streams each cached tensor onto
    its weight by position. Raises on any count / shape / keying mismatch so the
    caller can fall back to the source.
    """
    from kerasformers.base.base_mixin import build_dtype_scope

    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)

    expected = {
        "cache_format": CACHE_FORMAT_VERSION,
        "cache_layout": CACHE_LAYOUT_VERSION,
        "converter": CONVERTER_VERSION,
        "quantizer": QUANTIZATION_FORMAT_VERSION,
        "backend": keras.backend.backend(),
        "quantization": quant_id(quantization),
        "load_dtype": load_dtype,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"Converted cache {key}={meta.get(key)!r} does not match {value!r}."
            )
    if meta.get("architecture_hash") != _json_hash(meta.get("config")):
        raise ValueError("Converted cache architecture config is corrupt.")

    with build_dtype_scope(load_dtype):
        model = keras.saving.deserialize_keras_object(meta["config"])
        if quantization is not None:
            from kerasformers.quantization import quantize_skeleton

            quantize_skeleton(model, quantization)
        if not model.built:
            model.build_for_transfer()

    weights = list(model.weights)
    if len(weights) != meta["count"]:
        raise ValueError(
            f"Cached model has {meta['count']} weights but the rebuilt skeleton "
            f"has {len(weights)}; cache is stale."
        )

    shapes = meta["shapes"]
    if meta.get("keying") != "index":
        raise ValueError(
            f"Converted cache keying {meta.get('keying')!r} is unsupported "
            f"(only 'index'); rebuilding from source."
        )
    order = list(range(len(weights)))

    paths = [os.path.join(directory, s) for s in meta["shards"]]
    state = LazyStateDict.from_files(paths)
    try:
        for w, i in zip(weights, order):
            if list(w.shape) != list(shapes[i]):
                raise ValueError(
                    f"Cached weight shape {shapes[i]} != rebuilt {list(w.shape)}; "
                    f"cache is stale."
                )
            w.assign(state[f"{i:06d}"])
    finally:
        state.close()
    return model


def try_load_converted(directory, quantization, load_dtype):
    """Best-effort cache load: return the model, or ``None`` on any failure."""
    try:
        return load_converted(directory, quantization, load_dtype)
    except Exception as e:
        warnings.warn(
            f"Converted-cache reload failed ({e}); rebuilding from source.",
            stacklevel=2,
        )
        return None


def try_save_converted(model, directory, quantization, load_dtype=None):
    """Best-effort cache save: warn (never raise) if it can't be written."""
    try:
        save_converted(model, directory, quantization, load_dtype)
    except Exception as e:
        warnings.warn(
            f"Could not cache converted model to {directory} ({e}).", stacklevel=2
        )
