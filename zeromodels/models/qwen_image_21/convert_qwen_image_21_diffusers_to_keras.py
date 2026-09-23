from __future__ import annotations

from typing import Dict

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

QWEN_IMAGE_21_SOURCES = {
    "qwen-image-2.1": "Qwen/Qwen-Image-2.1",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "__": ".",
    "/kernel": ".weight",
    "/gamma": ".weight",
    "/beta": ".bias",
    "/scale": ".weight",
    "/": ".",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

TEXT_WEIGHT_NAME_MAPPING: Dict[str, str] = {
    **WEIGHT_NAME_MAPPING,
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.norm.weight",
    "language_model.": "model.",
    "decoder_layer_": "layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
}


def config_from_diffusers(repo, token=None):
    import json

    from huggingface_hub import hf_hub_download
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoConfig

    from zeromodels.models.qwen_image_21.qwen_image_21_config import (
        QwenImage21Config,
        QwenImage21VAEConfig,
    )
    from zeromodels.models.qwen_image_21.qwen_image_21_model import (
        QwenImage21Transformer2DModel,
    )

    transformer = json.load(
        open(
            hf_hub_download(repo, "config.json", subfolder="transformer", token=token),
            encoding="utf-8",
        )
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
    temperal = tuple(vae.get("temperal_downsample", (False, True, True, True)))
    rope = text_inner.get("rope_parameters") or text_inner.get("rope_scaling") or {}
    return QwenImage21Config(
        transformer_config={
            **QwenImage21Transformer2DModel.kwargs_from_diffusers_config(transformer),
            "sample_size": 64,
            "text_seq_len": 512,
        },
        vae_config=QwenImage21VAEConfig(
            base_dim=vae.get("base_dim", 96),
            decoder_base_dim=vae.get("decoder_base_dim", 144),
            z_dim=vae.get("z_dim", 64),
            dim_mult=tuple(vae.get("dim_mult", (1, 2, 4, 8, 8))),
            num_res_blocks=vae.get("num_res_blocks", 2),
            attn_scales=tuple(vae.get("attn_scales") or ()),
            temperal_downsample=temperal,
            dropout=vae.get("dropout", 0.0),
            input_channels=vae.get("in_channels", 4),
            out_channels=vae.get("out_channels", 4),
            is_residual=bool(vae.get("is_residual", True)),
            scale_factor_spatial=vae.get("scale_factor_spatial", 16),
            latents_mean=tuple(vae["latents_mean"]),
            latents_std=tuple(vae["latents_std"]),
            sample_size=1024,
        ),
        text_config={
            "vocab_size": text_inner.get("vocab_size", 151936),
            "embed_dim": text_inner.get("hidden_size", 4096),
            "mlp_dim": text_inner.get("intermediate_size", 12288),
            "num_layers": text_inner.get("num_hidden_layers", 36),
            "num_heads": text_inner.get("num_attention_heads", 32),
            "num_kv_heads": text_inner.get("num_key_value_heads", 8),
            "head_dim": text_inner.get("head_dim", 128),
            "norm_eps": text_inner.get("rms_norm_eps", 1e-6),
            "rope_theta": float(
                rope.get("rope_theta", text_inner.get("rope_theta", 5000000.0))
            ),
            "mrope_section": tuple(rope.get("mrope_section", (24, 20, 20))),
            "tie_embeddings": text.get("tie_word_embeddings", False),
            "max_seq_len": 1024,
        },
        scheduler_config=scheduler,
        bos_token_id=text_inner.get("bos_token_id", 151643),
        eos_token_id=text_inner.get("eos_token_id", 151645),
        pad_token_id=text_inner.get("bos_token_id", 151643),
        image_token_id=text.get("image_token_id", 151655),
    )


def transfer_qwen_image_21(
    repo, token=None, dtype="bfloat16", build_sample_size=8, config=None
):
    import gc
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.qwen_image_21.qwen_image_21_model import QwenImage21Model

    config = config or config_from_diffusers(repo, token=token)
    flat = config.constructor_kwargs()
    flat["transformer_sample_size"] = build_sample_size
    flat["vae_sample_size"] = max(build_sample_size * 16, 64)
    flat["max_sequence_length"] = min(int(flat.get("max_sequence_length", 512)), 64)

    print(f"[1/4] Building QwenImage21Model (dtype={dtype})…", flush=True)
    with build_dtype_scope(dtype), zeros_init():
        model = QwenImage21Model(**flat)

    # VAE RMSNorm checkpoints keep the name ``gamma`` (not ``weight``).
    vae_mapping = {
        k: v
        for k, v in WEIGHT_NAME_MAPPING.items()
        if k not in ("/gamma", "gamma")
    }
    for step, (component, subfolder, mapping, index_name, filename) in enumerate(
        (
            (
                model.transformer,
                "transformer",
                WEIGHT_NAME_MAPPING,
                "diffusion_pytorch_model.safetensors.index.json",
                None,
            ),
            (
                model.vae,
                "vae",
                vae_mapping,
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model.safetensors",
            ),
        ),
        start=2,
    ):
        print(f"[{step}/4] Transferring {subfolder}…", flush=True)
        state = {}
        try:
            if index_name is not None:
                index_path = hf_hub_download(
                    repo, index_name, subfolder=subfolder, token=token
                )
                with open(index_path, encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                shard_paths = {
                    shard: hf_hub_download(
                        repo, shard, subfolder=subfolder, token=token
                    )
                    for shard in sorted(set(weight_map.values()))
                }

                class _State:
                    def __contains__(self, key):
                        return key in weight_map

                    def __getitem__(self, key):
                        with safe_open(
                            shard_paths[weight_map[key]], framework="np"
                        ) as shard:
                            return shard.get_tensor(key)

                    def keys(self):
                        return weight_map.keys()

                    def __iter__(self):
                        return iter(weight_map)

                state = _State()
            else:
                raise FileNotFoundError
        except Exception:
            path = hf_hub_download(
                repo, filename or "diffusion_pytorch_model.safetensors",
                subfolder=subfolder,
                token=token,
            )
            state = {}
            with safe_open(path, framework="np") as shard:
                for key in shard.keys():
                    state[key] = shard.get_tensor(key)

        consumed = set()
        trainable, non_trainable = split_model_weights(component)
        # VAE / DiT nest safe_name prefixes; leaf-pair mapping matches Qwen-Image 1.x.
        prefer_leaf = subfolder in ("vae", "transformer")
        for keras_weight, _ in tqdm(
            trainable + non_trainable,
            desc=f"Transferring {subfolder} weights to Keras",
        ):
            leaf = "/".join(keras_weight.path.split("/")[-2:])
            if prefer_leaf:
                candidates = [leaf]
            else:
                parts = keras_weight.path.split("/")
                if parts and parts[0] in (
                    component.name,
                    "transformer",
                    "vae",
                    "QwenImage21Transformer2DModel",
                    "AutoencoderKLQwenImage21",
                ):
                    rel = "/".join(parts[1:])
                else:
                    rel = "/".join(parts)
                candidates = [rel, leaf]
            key = None
            for cand in candidates:
                mapped = cand
                for old, new in mapping.items():
                    mapped = mapped.replace(old, new)
                if mapped in state:
                    key = mapped
                    break
            if key is None:
                raise WeightMappingError(
                    keras_weight.path,
                    candidates[0]
                    if prefer_leaf
                    else "/".join(keras_weight.path.split("/")[-2:]),
                )
            consumed.add(key)
            raw = state[key]
            arr = np.asarray(raw)
            kshape = tuple(keras_weight.shape)
        if len(kshape) == 5 and arr.ndim == 5:
            arr = np.transpose(arr, (2, 3, 4, 1, 0))
        elif len(kshape) == 4 and arr.ndim == 4:
            arr = np.transpose(arr, (2, 3, 1, 0))
        elif (
            len(kshape) == 2
            and arr.ndim == 4
            and tuple(arr.shape[-2:]) == (1, 1)
        ):
            # Diffusers mid-block attention uses 1×1 Conv2d; Keras uses Dense.
            # Keep torch (out, in) layout — transfer_weights will transpose.
            arr = arr[:, :, 0, 0]
        elif (
            arr.ndim > 1
            and len(kshape) == 1
            and int(np.prod(arr.shape)) == kshape[0]
        ):
            arr = arr.reshape(kshape)
            if len(keras_weight.shape) in (4, 5):
                if tuple(keras_weight.shape) != arr.shape:
                    raise WeightShapeMismatchError(
                        keras_weight.path, keras_weight.shape, key, np.shape(raw)
                    )
                keras_weight.assign(arr)
                continue
            if tuple(keras_weight.shape) != tuple(arr.shape):
                if not compare_keras_torch_names(
                    keras_weight.path, keras_weight, key, raw
                ):
                    raise WeightShapeMismatchError(
                        keras_weight.path,
                        keras_weight.shape,
                        key,
                        np.asarray(raw).shape,
                    )
            transfer_weights(keras_weight.path, keras_weight, arr)
        del state
        gc.collect()

    print("[4/4] Transferring text encoder…", flush=True)
    index_path = hf_hub_download(
        repo, "model.safetensors.index.json", subfolder="text_encoder", token=token
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_paths = {
        shard: hf_hub_download(repo, shard, subfolder="text_encoder", token=token)
        for shard in sorted(set(weight_map.values()))
    }
    hf_keys = {}
    for key in weight_map:
        # Text-only tower: keep language_model / embed_tokens; drop vision + lm_head
        # + final norm (Diffusers reads pre-norm hidden states).
        if key.startswith("model.visual.") or key.startswith("lm_head."):
            continue
        if key in ("model.norm.weight",) or key.endswith(".norm.weight") and key.count(".") <= 2:
            if key == "model.norm.weight":
                continue
        if key.startswith("model.language_model."):
            hf_keys["model." + key[len("model.language_model.") :]] = key
        elif key.startswith("model.") and not key.startswith("model.visual."):
            hf_keys[key] = key
    # Drop final norm from the transferable set.
    hf_keys.pop("model.norm.weight", None)
    consumed = set()
    text_encoder = model.text_encoder
    for weight in tqdm(
        text_encoder.weights, desc="Transferring text_encoder weights to Keras"
    ):
        name = weight.path.removeprefix(f"{text_encoder.name}/")
        for old, new in TEXT_WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in hf_keys:
            raise WeightMappingError(weight.path, name)
        consumed.add(name)
        shard_key = hf_keys[name]
        with safe_open(shard_paths[weight_map[shard_key]], framework="np") as shard:
            transfer_weights(weight.path, weight, shard.get_tensor(shard_key))
    del weight_map, shard_paths, hf_keys
    gc.collect()
    return model, config


if __name__ == "__main__":
    import gc
    import os

    import keras

    OUT_DIR = os.environ.get(
        "ZM_OUT_DIR", "C:/Users/gites/Desktop/code/qwen_image_21_weights"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    MAX_SHARD_GB = 5.0
    token = os.environ.get("HF_TOKEN")
    dtype = os.environ.get("ZM_DTYPE", "bfloat16")
    device = os.environ.get("ZM_DEVICE", "cpu")
    selected = [v for v in os.environ.get("ZM_VARIANTS", "").split(",") if v]
    sources = {
        variant: source
        for variant, source in QWEN_IMAGE_21_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        with keras.device(device):
            model, config = transfer_qwen_image_21(source, token=token, dtype=dtype)

        itemsize = 2 if "16" in dtype else 4
        n_bytes = sum(int(np.prod(w.shape)) * itemsize for w in model.weights)
        stem = os.path.join(OUT_DIR, variant.replace("-", "_").replace(".", "_"))
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
