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
from zeromodels.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    WEIGHT_NAME_MAPPING as QWEN2_VL_WEIGHT_NAME_MAPPING,
)
from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    WEIGHT_NAME_MAPPING,
)

QWEN_IMAGE_SOURCES = {
    "qwen-image": "Qwen/Qwen-Image",
}
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
    """Transpose conv kernels / squeeze VAE RMSNorm gamma to match Keras shapes."""
    arr = np.asarray(torch_weight)
    kshape = tuple(keras_weight.shape)
    if len(kshape) == 5 and arr.ndim == 5:
        arr = np.transpose(arr, (2, 3, 4, 1, 0))
    elif len(kshape) == 4 and arr.ndim == 4:
        arr = np.transpose(arr, (2, 3, 1, 0))
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
        if len(keras_weight.shape) in (4, 5):
            if tuple(keras_weight.shape) != torch_weight.shape:
                raise WeightShapeMismatchError(
                    keras_weight.path, keras_weight.shape, key, np.shape(raw)
                )
            keras_weight.assign(torch_weight)
            continue
        if tuple(keras_weight.shape) != tuple(np.asarray(torch_weight).shape):
            if len(keras_weight.shape) == 1 and np.asarray(raw).ndim > 1:
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
    if unused:
        raise ValueError(
            f"{type(component).__name__}: {len(unused)} checkpoint tensors "
            f"unused, e.g. {unused[:5]}."
        )


def transfer_qwen_image(
    repo, token=None, dtype="float16", build_sample_size=16, config=None
):
    """Convert the Diffusers pipeline into a :class:`QwenImageModel`.

    Builds at a small ``transformer_sample_size`` (the DiT weights are
    resolution-independent) to keep the functional graph small.
    Returns ``(model, config)``.
    """
    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.qwen_image.qwen_image_model import QwenImageModel

    config = config or config_from_diffusers(repo, token=token)
    flat = config.constructor_kwargs()
    flat["transformer_sample_size"] = build_sample_size
    flat["vae_sample_size"] = max(build_sample_size * 8, 64)

    print(f"[1/4] Building QwenImageModel (dtype={dtype})…", flush=True)
    with build_dtype_scope(dtype), zeros_init():
        model = QwenImageModel(**flat)

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

    print("[3/4] Transferring VAE…", flush=True)
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

    print("[4/4] Transferring text encoder…", flush=True)
    transfer_text_encoder(model.text_encoder, repo, token=token)
    gc.collect()
    return model, config


def transfer_text_encoder(text_encoder, repo, token=None):
    """Copy the Qwen2.5-VL text tower into ``text_encoder``.

    Streams tensors from the safetensors shards one at a time (never a full
    in-memory state dict); the vision tower and LM head are skipped.
    """
    state = _load_safetensors_state(
        repo, "text_encoder", "model.safetensors.index.json", token=token
    )
    hf_keys = {}
    for key in state.keys():
        if key.startswith("model.language_model."):
            hf_keys["model." + key[len("model.language_model.") :]] = key
        elif key.startswith("model.") and not key.startswith("model.visual."):
            hf_keys[key] = key

    consumed = set()
    for weight in tqdm(text_encoder.weights, desc="text_encoder"):
        path = weight.path.removeprefix(f"{text_encoder.name}/")
        name = path.replace("/", ".")
        for old, new in QWEN2_VL_WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in hf_keys:
            raise WeightMappingError(weight.path, name)
        consumed.add(name)
        transfer_weights(weight.path, weight, state[hf_keys[name]])
    unused = sorted(set(hf_keys) - consumed)
    if unused:
        raise ValueError(
            f"text_encoder: {len(unused)} checkpoint tensors unused, e.g. {unused[:5]}."
        )
    del state


if __name__ == "__main__":
    import keras

    OUT_DIR = os.environ.get("ZM_OUT_DIR", "C:/Users/gites/Desktop/code/qwen_image_weights")
    os.makedirs(OUT_DIR, exist_ok=True)
    MAX_SHARD_GB = 5.0
    token = os.environ.get("HF_TOKEN")
    dtype = os.environ.get("ZM_DTYPE", "bfloat16")
    device = os.environ.get("ZM_DEVICE", "cpu")
    selected = [v for v in os.environ.get("ZM_VARIANTS", "").split(",") if v]
    sources = {
        variant: source
        for variant, source in QWEN_IMAGE_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        with keras.device(device):
            model, config = transfer_qwen_image(source, token=token, dtype=dtype)

        itemsize = 2 if "16" in dtype else 4
        n_bytes = sum(int(np.prod(w.shape)) * itemsize for w in model.weights)
        stem = os.path.join(OUT_DIR, variant.replace("-", "_"))
        if n_bytes > MAX_SHARD_GB * 1024**3:
            out = f"{stem}.weights.json"
            model.save_weights(out, max_shard_size=MAX_SHARD_GB)
        else:
            out = f"{stem}.weights.h5"
            model.save_weights(out)
        print(f"  saved -> {out}  ({n_bytes / 1024**3:.2f} GB {dtype})")

        del model
        keras.backend.clear_session()
        gc.collect()

