import collections.abc
import json
import os

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


class LazyStateDict(collections.abc.Mapping):
    """Memory-mapped, lazily-read ``{name: numpy_array}`` view over safetensors.

    A drop-in for the eager ``{name: np.ndarray}`` dict every converter consumes,
    but each tensor is materialized only when looked up (``safe_open`` mmap +
    ``get_tensor``) and freed as soon as the converter assigns it, so peak host
    RAM is ~one tensor instead of the whole checkpoint. Backed by ``{name: shard
    path}``; ``safe_open`` handles open on first use and are cached per shard.
    Returns exactly what ``safetensors.numpy.load_file`` would (same values and
    dtypes, incl. bf16 as ``ml_dtypes`` arrays): purely a memory optimization.

    ``__contains__`` is overridden to test membership against the key index only:
    the default :class:`collections.abc.Mapping` implementation calls
    ``__getitem__``, which for this class would read the whole tensor, so
    ``name in sd`` (which converters do right before ``sd[name]``) would otherwise
    read every present tensor twice.
    """

    def __init__(self, tensor_to_path):
        self._paths = dict(tensor_to_path)
        self._handles = {}

    @classmethod
    def from_files(cls, paths):
        from safetensors import safe_open

        tensor_to_path = {}
        for path in paths:
            with safe_open(path, framework="numpy") as f:
                for name in f.keys():
                    tensor_to_path[name] = path
        return cls(tensor_to_path)

    def handle(self, path):
        f = self._handles.get(path)
        if f is None:
            from safetensors import safe_open

            f = safe_open(path, framework="numpy")
            self._handles[path] = f
        return f

    def __getitem__(self, key):
        try:
            path = self._paths[key]
        except KeyError:
            raise KeyError(key) from None
        return self.handle(path).get_tensor(key)

    def __contains__(self, key):
        # O(1) key check: must NOT read the tensor (see class docstring).
        return key in self._paths

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def close(self, completed=True):
        # Release the mmap handles after conversion. ``completed`` is accepted for
        # the caller's finally-block contract but no longer affects anything.
        self._handles.clear()


def load_bin_state_dict(paths):
    """Eagerly load legacy ``.bin`` pickle shards to a ``{name: numpy}`` dict.

    Pickle checkpoints have no mmap / lazy path, so they are read in full on CPU
    (``weights_only=True``: no arbitrary code execution). torch is conversion-only
    here and never touches CUDA. bf16 / fp8 tensors are upcast to float32 first,
    since ``torch.bfloat16`` / ``float8`` have no numpy dtype and ``.numpy()`` on
    them raises ``TypeError: Got unsupported ScalarType``.
    """
    import torch

    def to_numpy(v):
        if not hasattr(v, "cpu"):
            return v
        if v.dtype == torch.bfloat16 or "float8" in str(v.dtype):
            v = v.float()
        return v.cpu().numpy()

    state = {}
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        state.update({k: to_numpy(v) for k, v in shard.items()})
    return state


def download_hf_state_dict(hf_id, token=None):
    """Download model weights and return a ``{name: numpy_array}`` mapping.

    ``token`` is forwarded to ``hf_hub_download`` for gated / private repos.
    Safetensors are returned as a :class:`LazyStateDict`, memory-mapped and read
    one tensor at a time, so the full checkpoint never sits in host RAM. torch is
    used only as an eager CPU fallback for legacy ``.bin`` checkpoints (never
    touches CUDA); those return a plain dict.

    Tries (in order):

    1. ``model.safetensors`` (single-file safetensors)
    2. ``model.safetensors.index.json`` (sharded safetensors)
    3. ``pytorch_model.bin`` (single-file pickle)
    4. ``pytorch_model.bin.index.json`` (sharded pickle)
    """
    try:
        path = hf_hub_download(hf_id, "model.safetensors", token=token)
    except EntryNotFoundError:
        path = None
    if path is not None:
        return LazyStateDict.from_files([path])

    try:
        index_path = hf_hub_download(hf_id, "model.safetensors.index.json", token=token)
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        local = {
            shard: hf_hub_download(hf_id, shard, token=token)
            for shard in sorted(set(weight_map.values()))
        }
        return LazyStateDict({name: local[shard] for name, shard in weight_map.items()})

    try:
        path = hf_hub_download(hf_id, "pytorch_model.bin", token=token)
    except EntryNotFoundError:
        path = None
    if path is not None:
        return load_bin_state_dict([path])

    try:
        index_path = hf_hub_download(hf_id, "pytorch_model.bin.index.json", token=token)
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shards = [
            hf_hub_download(hf_id, shard, token=token)
            for shard in sorted(set(weight_map.values()))
        ]
        return load_bin_state_dict(shards)

    raise FileNotFoundError(
        f"No supported weights file found in HF repo '{hf_id}'. "
        f"Expected one of: model.safetensors, model.safetensors.index.json, "
        f"pytorch_model.bin, pytorch_model.bin.index.json."
    )


def load_and_convert_from_hf(
    model,
    model_name,
    hf_model_id,
    transfer_fn,
    is_gated=False,
):
    """Download, convert, and cache source weights for a Keras model.

    Generic helper used for models whose Keras weights cannot be
    redistributed directly: either due to **license gating** (SAM3,
    DINOv3) or because the converted weights **exceed distribution host
    limits** (MetaCLIP 2 Huge/Giant variants at 3-4 GB single tensors,
    larger than GitHub's 2 GB release asset cap).

    Weights are cached at ``~/.cache/zeromodels/<model_name>/``. Sharded at
    5 GB per shard for local cache.

    Args:
        model: The Keras model instance to load weights into.
        model_name: String used as the cache subdirectory name.
        hf_model_id: Model-hub identifier.
        transfer_fn: Callable ``(keras_model, hf_state_dict) -> None``. Receives
            the raw on-disk checkpoint tensors (the same key layout the
            ``hf:`` / safetensors release path produces).
        is_gated: When True, emits the license-acceptance error message
            on 401/403. When False (default), lets the download error propagate.
    """
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "zeromodels", model_name
    )
    cached_weights = os.path.join(cache_dir, f"{model_name}.weights.h5")

    if os.path.exists(cached_weights):
        print(f"Loading cached {model_name} weights from {cached_weights}")
        model.load_weights(cached_weights)
        return

    gated_note = " (requires accepted license + HF token)" if is_gated else ""
    print(f"Downloading {model_name} from HuggingFace{gated_note}...")

    hf_token = os.environ.get("HF_TOKEN")
    try:
        hf_state_dict = download_hf_state_dict(hf_model_id, token=hf_token)
    except Exception as e:
        error_msg = str(e)
        if is_gated and (
            "gated" in error_msg.lower() or "401" in error_msg or "403" in error_msg
        ):
            raise OSError(
                f"\n{'=' * 60}\n"
                f"Access denied for '{hf_model_id}'.\n\n"
                f"This model is gated and requires license acceptance.\n"
                f"Please follow these steps:\n\n"
                f"  1. Go to https://huggingface.co/{hf_model_id}\n"
                f"     and accept the license agreement.\n\n"
                f"  2. Authenticate using one of:\n"
                f"     - Run: huggingface-cli login\n"
                f"     - Or set: export HF_TOKEN=<your_token>\n"
                f"{'=' * 60}"
            ) from e
        raise

    print(f"Converting {model_name} weights to Keras...")
    completed = False
    try:
        transfer_fn(model, hf_state_dict)
        completed = True
    finally:
        close = getattr(hf_state_dict, "close", None)
        if callable(close):
            close(completed=completed)

    os.makedirs(cache_dir, exist_ok=True)

    total_bytes = sum(w.numpy().nbytes for w in model.weights)
    size_gb = total_bytes / (1024**3)
    save_kwargs = {}
    if size_gb > 5:
        save_kwargs["max_shard_size"] = 5.0

    model.save_weights(cached_weights, **save_kwargs)
    print(f"Cached {model_name} weights to {cached_weights} ({size_gb:.1f} GB)")
