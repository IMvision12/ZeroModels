"""Offline Diffusers ``Qwen/Qwen-Image`` → ZeroModels Keras weight conversion.

Converts the transformer, VAE, and Qwen2.5-VL text encoder into a hosted
``zeromodels/qwen-image`` layout (``zm_config.json`` + sharded weights).

Usage::

    python -m zeromodels.models.qwen_image.convert_qwen_image_diffusers_to_keras

Env:
    ZM_OUT_DIR   output directory (default ``./qwen_image_weights``)
    HF_TOKEN     optional Hub token
    ZM_DTYPE     ``float16`` / ``bfloat16`` / ``float32`` (default ``float16``)
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
    zeros_init,
)
from zeromodels.models.qwen2_5_vl.convert_qwen2_5_vl_hf_to_keras import (
    transfer_qwen2_5_vl_weights,
)
from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    WEIGHT_NAME_MAPPING,
)

QWEN_IMAGE_SOURCES = {
    "qwen-image": "Qwen/Qwen-Image",
}

# VAE Diffusers RMSNorm keeps the leaf name ``gamma`` (not ``weight``).
VAE_WEIGHT_NAME_MAPPING = {
    k: v for k, v in WEIGHT_NAME_MAPPING.items() if "/gamma" not in k
}
VAE_WEIGHT_NAME_MAPPING["/"] = "."
VAE_WEIGHT_NAME_MAPPING["__"] = "."


def config_from_diffusers(repo, token=None):
    """Build a :class:`QwenImageConfig` from a Diffusers Qwen-Image repo."""
    from huggingface_hub import hf_hub_download
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers import QwenImageTransformer2DModel as DiffusersTransformer
    from transformers import AutoConfig

    from zeromodels.models.qwen_image.qwen_image_config import QwenImageConfig
    from zeromodels.models.qwen_image.qwen_image_model import (
        QwenImageTransformer2DModel,
    )
    from zeromodels.models.qwen_image.qwen_image_vae import QwenImageVAEConfig

    transformer = dict(
        DiffusersTransformer.load_config(repo, subfolder="transformer", token=token)
    )
    vae = json.load(
        open(
            hf_hub_download(repo, "config.json", subfolder="vae", token=token),
            encoding="utf-8",
        )
    )
    text = AutoConfig.from_pretrained(
        repo, subfolder="text_encoder", token=token
    ).to_dict()
    text_inner = text.get("text_config") or text
    scheduler = {
        k: v
        for k, v in FlowMatchEulerDiscreteScheduler.load_config(
            repo, subfolder="scheduler", token=token
        ).items()
        if k == "_class_name" or not k.startswith("_")
    }
    temperal = tuple(vae.get("temperal_downsample", (False, True, True)))
    return QwenImageConfig(
        transformer_config={
            **QwenImageTransformer2DModel.kwargs_from_diffusers_config(transformer),
            "sample_size": 128,
            "text_seq_len": 512,
        },
        vae_config=QwenImageVAEConfig(
            base_dim=vae.get("base_dim", 96),
            z_dim=vae.get("z_dim", 16),
            dim_mult=tuple(vae.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=vae.get("num_res_blocks", 2),
            attn_scales=tuple(vae.get("attn_scales") or ()),
            temperal_downsample=temperal,
            dropout=vae.get("dropout", 0.0),
            latents_mean=tuple(vae["latents_mean"]),
            latents_std=tuple(vae["latents_std"]),
            sample_size=1024,
        ),
        text_config={
            "vocab_size": text_inner.get("vocab_size", 152064),
            "embed_dim": text_inner.get("hidden_size", 3584),
            "mlp_dim": text_inner.get("intermediate_size", 18944),
            "num_layers": text_inner.get("num_hidden_layers", 28),
            "num_heads": text_inner.get("num_attention_heads", 28),
            "num_kv_heads": text_inner.get("num_key_value_heads", 4),
            "norm_eps": text_inner.get("rms_norm_eps", 1e-6),
            "rope_theta": text_inner.get("rope_theta", 1000000.0),
            "mrope_section": tuple(
                (text_inner.get("rope_scaling") or {}).get(
                    "mrope_section", (16, 24, 24)
                )
            ),
            "tie_embeddings": text.get("tie_word_embeddings", False),
            "max_seq_len": 1024,
        },
        scheduler_config=scheduler,
        bos_token_id=text.get("bos_token_id", 151643),
        eos_token_id=text.get("eos_token_id", 151645),
        pad_token_id=text.get("bos_token_id", 151643),
    )


def _keras_to_torch_key(keras_weight, mapping):
    key = "/".join(keras_weight.path.split("/")[-2:])
    for old, new in mapping.items():
        key = key.replace(old, new)
    return key


def _prepare_torch_array(keras_weight, torch_weight):
    """Transpose Conv3D / squeeze VAE RMSNorm gamma to match Keras shapes."""
    arr = np.asarray(torch_weight)
    kshape = tuple(keras_weight.shape)
    if len(kshape) == 5 and arr.ndim == 5:
        # Diffusers Conv3d NCDHW ``(O, I, T, H, W)`` → Keras NDHWC ``(T, H, W, I, O)``.
        arr = np.transpose(arr, (2, 3, 4, 1, 0))
    elif arr.ndim > 1 and len(kshape) == 1 and int(np.prod(arr.shape)) == kshape[0]:
        arr = arr.reshape(kshape)
    return arr


def _load_safetensors_state(repo, subfolder, index_name, token=None):
    """Lazy shard map: ``{key: (shard_path, key)}`` resolved on demand via safe_open."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(
        repo, index_name, subfolder=subfolder, token=token
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_paths = {
        shard: hf_hub_download(repo, shard, subfolder=subfolder, token=token)
        for shard in sorted(set(weight_map.values()))
    }

    class _ShardDict(dict):
        def __contains__(self, key):
            return key in weight_map

        def __getitem__(self, key):
            if key not in weight_map:
                raise KeyError(key)
            path = shard_paths[weight_map[key]]
            with safe_open(path, framework="np") as shard:
                return shard.get_tensor(key)

        def keys(self):
            return weight_map.keys()

        def __iter__(self):
            return iter(weight_map)

        def __len__(self):
            return len(weight_map)

    return _ShardDict()


def _load_single_safetensors(repo, subfolder, filename, token=None):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo, filename, subfolder=subfolder, token=token)
    state = {}
    with safe_open(path, framework="np") as shard:
        for key in shard.keys():
            state[key] = shard.get_tensor(key)
    return state


def transfer_component(component, state, mapping, desc, ignore=()):
    """Copy ``state`` tensors into a Keras sub-model using ``mapping``."""
    consumed = set(ignore)
    trainable, non_trainable = split_model_weights(component)
    for keras_weight, _ in tqdm(trainable + non_trainable, desc=desc):
        if any(part in keras_weight.path for part in ("pos_freqs", "neg_freqs")):
            continue
        key = _keras_to_torch_key(keras_weight, mapping)
        if key in ignore:
            continue
        if key not in state:
            raise WeightMappingError(keras_weight.path, key)
        consumed.add(key)
        raw = state[key]
        torch_weight = _prepare_torch_array(keras_weight, raw)
        if tuple(keras_weight.shape) != tuple(np.asarray(torch_weight).shape):
            # Dense / Conv2D still need the shared transpose inside transfer_weights;
            # only enforce exact match after _prepare (Conv3D / squeezed gamma).
            if len(keras_weight.shape) == 5 or (
                len(keras_weight.shape) == 1
                and np.asarray(raw).ndim > 1
            ):
                raise WeightShapeMismatchError(
                    keras_weight.path,
                    keras_weight.shape,
                    key,
                    torch_weight.shape,
                )
            if not compare_keras_torch_names(
                keras_weight.path, keras_weight, key, raw
            ):
                raise WeightShapeMismatchError(
                    keras_weight.path,
                    keras_weight.shape,
                    key,
                    np.asarray(raw).shape,
                )
        transfer_weights(keras_weight.path, keras_weight, torch_weight)
    unused = sorted(set(state) - consumed)
    # RoPE tables / unused buffers on the torch side are fine to ignore when empty.
    if unused:
        raise ValueError(
            f"{type(component).__name__}: {len(unused)} checkpoint tensors "
            f"unused, e.g. {unused[:5]}."
        )


def transfer_qwen_image(repo, token=None, dtype="float16", build_sample_size=16):
    """Convert Diffusers Qwen-Image weights into a :class:`QwenImageModel`.

    Builds the Keras container at a small ``transformer_sample_size`` (weights are
    resolution-independent for the DiT) to keep the functional graph smaller,
    transfers transformer / VAE / text-encoder weights, and returns ``(model, config)``.
    """
    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.qwen_image.qwen_image_model import QwenImageModel

    config = config_from_diffusers(repo, token=token)
    flat = config.constructor_kwargs()
    # Smaller spatial graph for conversion; Dense DiT weights do not depend on it.
    flat["transformer_sample_size"] = build_sample_size
    flat["vae_sample_size"] = max(build_sample_size * 8, 64)

    print(f"[1/4] Building Keras QwenImageModel (dtype={dtype})…", flush=True)
    import sys
    import time

    t0 = time.time()
    with build_dtype_scope(dtype), zeros_init():
        print("  building components…", flush=True)
        model = QwenImageModel(**flat)
    print(
        f"  build done in {time.time() - t0:.1f}s  "
        f"({len(model.weights)} weights)",
        flush=True,
    )
    sys.stdout.flush()

    print("[2/4] Transferring transformer…", flush=True)
    transformer_state = _load_safetensors_state(
        repo,
        "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        token=token,
    )
    transfer_component(
        model.transformer,
        transformer_state,
        WEIGHT_NAME_MAPPING,
        desc="transformer",
    )
    del transformer_state
    gc.collect()

    print("[3/4] Transferring VAE…")
    vae_state = _load_single_safetensors(
        repo, "vae", "diffusion_pytorch_model.safetensors", token=token
    )
    transfer_component(
        model.vae,
        vae_state,
        VAE_WEIGHT_NAME_MAPPING,
        desc="vae",
    )
    del vae_state
    gc.collect()

    print("[4/4] Transferring text encoder (Qwen2.5-VL text tower)…")
    text_state = _load_safetensors_state(
        repo, "text_encoder", "model.safetensors.index.json", token=token
    )
    # Drop vision + LM head: our text-only tower does not carry them.
    text_only = {
        k: text_state[k]
        for k in text_state.keys()
        if k.startswith("model.") and not k.startswith("model.visual.")
    }
    # Materialize into a plain dict for the shared VL transfer helper.
    text_np = {k: np.asarray(text_only[k]) for k in tqdm(text_only, desc="load text")}
    del text_state, text_only
    gc.collect()
    transfer_qwen2_5_vl_weights(model.text_encoder, text_np)
    del text_np
    gc.collect()

    return model, config


def save_converted(model, config, out_dir, variant="qwen-image", max_shard_gb=5.0):
    """Write ``zm_config.json``, sharded weights, and copy the tokenizer files."""
    import shutil

    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, variant.replace("-", "_"))

    config_path = os.path.join(out_dir, "zm_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  wrote {config_path}")

    n_bytes = sum(int(np.prod(w.shape)) * 2 for w in model.weights)  # ~fp16
    if n_bytes > max_shard_gb * 1024**3:
        out = f"{stem}.weights.json"
        model.save_weights(out, max_shard_size=max_shard_gb)
    else:
        out = f"{stem}.weights.h5"
        model.save_weights(out)
    print(f"  saved weights -> {out}  (~{n_bytes / 1024**3:.2f} GB fp16-est)")

    source = QWEN_IMAGE_SOURCES[variant]
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ):
        try:
            src = hf_hub_download(source, name, subfolder="tokenizer")
            shutil.copy2(src, os.path.join(out_dir, name if name != "tokenizer.json" else "tokenizer.json"))
        except Exception as exc:  # noqa: BLE001 — best-effort tokenizer copy
            print(f"  skip tokenizer file {name}: {exc}")


if __name__ == "__main__":
    import keras

    OUT_DIR = os.environ.get(
        "ZM_OUT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "qwen_image_weights"),
    )
    OUT_DIR = os.path.abspath(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    dtype = os.environ.get("ZM_DTYPE", "float16")
    selected = [v for v in os.environ.get("ZM_VARIANTS", "qwen-image").split(",") if v]

    for variant in selected:
        source = QWEN_IMAGE_SOURCES[variant]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = transfer_qwen_image(source, token=token, dtype=dtype)
        save_converted(model, config, OUT_DIR, variant=variant)
        del model
        keras.backend.clear_session()
        gc.collect()
        print(f"Done: {variant} -> {OUT_DIR}")
