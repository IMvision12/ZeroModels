import re

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
)
from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    WEIGHT_NAME_MAPPING,
)
from zeromodels.models.stable_diffusion_xl.convert_stable_diffusion_xl_diffusers_to_keras import (
    LazyNumpyStateDict,
    text_config_kwargs,
    transfer_text_encoder,
)

STABLE_DIFFUSION_3_SOURCES = {
    "stable-diffusion-3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}
T5_XXL_ENCODER_VARIANT = "t5-v1_1-xxl-encoder"
T5_ENCODER_FIXED_LEAVES = {
    "shared": "shared.weight",
    "encoder_rel_bias": (
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ),
    "encoder_final_layer_norm": "encoder.final_layer_norm.weight",
}


def config_from_diffusers(repo, token=None, config_cls=None):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers import SD3Transformer2DModel as DiffusersSD3Transformer2DModel
    from transformers import CLIPTextConfig, CLIPTokenizerFast, T5TokenizerFast

    from zeromodels.models.stable_diffusion_3.stable_diffusion_3_config import (
        StableDiffusion3Config,
    )
    from zeromodels.models.stable_diffusion_3.stable_diffusion_3_model import (
        SD3Transformer2DModel,
    )

    config_cls = config_cls or StableDiffusion3Config
    transformer = dict(
        DiffusersSD3Transformer2DModel.load_config(
            repo, subfolder="transformer", token=token
        )
    )
    vae = dict(AutoencoderKL.load_config(repo, subfolder="vae", token=token))
    text = CLIPTextConfig.from_pretrained(
        repo, subfolder="text_encoder", token=token
    ).to_dict()
    text_2 = CLIPTextConfig.from_pretrained(
        repo, subfolder="text_encoder_2", token=token
    ).to_dict()
    scheduler = {
        k: v
        for k, v in FlowMatchEulerDiscreteScheduler.load_config(
            repo, subfolder="scheduler", token=token
        ).items()
        if k == "_class_name" or not k.startswith("_")
    }
    tokenizer = CLIPTokenizerFast.from_pretrained(
        repo, subfolder="tokenizer", token=token
    )
    tokenizer_2 = CLIPTokenizerFast.from_pretrained(
        repo, subfolder="tokenizer_2", token=token
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        repo, subfolder="tokenizer_3", token=token
    )
    return config_cls(
        transformer_config=SD3Transformer2DModel.kwargs_from_diffusers_config(
            transformer
        ),
        vae_config={
            "in_channels": vae.get("in_channels", 3),
            "out_channels": vae.get("out_channels", 3),
            "latent_channels": vae.get("latent_channels", 16),
            "block_out_channels": tuple(vae["block_out_channels"]),
            "layers_per_block": vae.get("layers_per_block", 2),
            "norm_num_groups": vae.get("norm_num_groups", 32),
            "sample_size": transformer.get("sample_size", 128)
            * 2 ** (len(vae["block_out_channels"]) - 1),
            "scaling_factor": vae.get("scaling_factor") or 1.5305,
            "shift_factor": vae.get("shift_factor") or 0.0,
            "force_upcast": bool(vae.get("force_upcast", True)),
            "use_quant_conv": bool(vae.get("use_quant_conv", True)),
            "use_post_quant_conv": bool(vae.get("use_post_quant_conv", True)),
        },
        text_config={
            **text_config_kwargs(text),
            "projection_dim": text.get("projection_dim", 768),
            "hidden_act": text.get("hidden_act", "quick_gelu"),
        },
        text_config_2={
            **text_config_kwargs(text_2),
            "projection_dim": text_2.get("projection_dim", 1280),
            "hidden_act": text_2.get("hidden_act", "gelu"),
        },
        hidden_act=text.get("hidden_act", "quick_gelu"),
        layer_norm_eps=text.get("layer_norm_eps", 1e-5),
        scheduler_config=scheduler,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        pad_token_id_2=tokenizer_2.pad_token_id,
        eos_token_id_3=tokenizer_3.eos_token_id,
        pad_token_id_3=tokenizer_3.pad_token_id,
    )


def from_pretrained_fp16(cls, repo, subfolder, token=None):
    import torch

    try:
        return cls.from_pretrained(
            repo,
            subfolder=subfolder,
            variant="fp16",
            torch_dtype=torch.float16,
            token=token,
        )
    except OSError:
        return cls.from_pretrained(
            repo, subfolder=subfolder, torch_dtype=torch.float16, token=token
        )


def transfer_stable_diffusion_3(
    repo, token=None, dtype="float16", model_cls=None, config_cls=None
):
    import gc

    import torch
    from diffusers import AutoencoderKL as DiffusersAutoencoderKL
    from diffusers import SD3Transformer2DModel as DiffusersSD3Transformer2DModel
    from transformers import CLIPTextModelWithProjection

    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.stable_diffusion_3.stable_diffusion_3_model import (
        StableDiffusion3Model,
    )

    model_cls = model_cls or StableDiffusion3Model
    config = config_from_diffusers(repo, token=token, config_cls=config_cls)
    with build_dtype_scope(dtype):
        model = model_cls(config)

    legacy = {
        ".query.": ".to_q.",
        ".key.": ".to_k.",
        ".value.": ".to_v.",
        ".proj_attn.": ".to_out.0.",
    }
    for component, subfolder in (
        (model.transformer, "transformer"),
        (model.vae, "vae"),
    ):
        ignore = ()
        if subfolder == "transformer":
            module = from_pretrained_fp16(
                DiffusersSD3Transformer2DModel, repo, subfolder, token=token
            )
            state = LazyNumpyStateDict(module)
            table = component.get_layer("pos_embed").pos_embed
            table.assign(np.asarray(state["pos_embed.pos_embed"]).reshape(table.shape))
            ignore = ("pos_embed.pos_embed",)
        else:
            module = DiffusersAutoencoderKL.from_pretrained(
                repo, subfolder=subfolder, torch_dtype=torch.float32, token=token
            )
            state = {}
            for key, value in module.state_dict().items():
                if ".attentions." in key:
                    for old, new in legacy.items():
                        key = key.replace(old, new)
                state[key] = value.detach().cpu().numpy()
        del module
        consumed = set(ignore)
        trainable, non_trainable = split_model_weights(component)
        for keras_weight, _ in tqdm(
            trainable + non_trainable, desc=f"Transferring {subfolder} weights to Keras"
        ):
            key = "/".join(keras_weight.path.split("/")[-2:])
            for old, new in WEIGHT_NAME_MAPPING.items():
                key = key.replace(old, new)
            if key in ignore:
                continue
            consumed.add(key)
            if key not in state:
                raise WeightMappingError(keras_weight.path, key)
            torch_weight = state[key]
            if not compare_keras_torch_names(
                keras_weight.path, keras_weight, key, torch_weight
            ):
                raise WeightShapeMismatchError(
                    keras_weight.path, keras_weight.shape, key, torch_weight.shape
                )
            transfer_weights(key, keras_weight, torch_weight)
        unused = sorted(set(state) - consumed)
        if unused:
            raise ValueError(
                f"{type(component).__name__}: {len(unused)} checkpoint tensors "
                f"unused, e.g. {unused[:3]}."
            )
        del state
        gc.collect()

    for attr, subfolder in (
        ("text_encoder", "text_encoder"),
        ("text_encoder_2", "text_encoder_2"),
    ):
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            repo, subfolder=subfolder, torch_dtype=torch.float32, token=token
        )
        transfer_text_encoder(getattr(model, attr), text_encoder)
        del text_encoder
        gc.collect()
    return model, config


def transfer_t5_encoder(repo, token=None, dtype="float16"):
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import T5Config

    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.stable_diffusion_3.stable_diffusion_3_model import (
        SD3T5EncoderModel,
    )

    hf_config = T5Config.from_pretrained(repo, subfolder="text_encoder_3", token=token)
    cfg = SD3T5EncoderModel.kwargs_from_hf_config(hf_config.to_dict())
    with build_dtype_scope(dtype):
        model = SD3T5EncoderModel(**cfg)

    from huggingface_hub.errors import EntryNotFoundError

    try:
        index = hf_hub_download(
            repo,
            "model.safetensors.index.fp16.json",
            subfolder="text_encoder_3",
            token=token,
        )
    except EntryNotFoundError:
        index = hf_hub_download(
            repo,
            "model.safetensors.index.json",
            subfolder="text_encoder_3",
            token=token,
        )
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_paths = {
        shard: hf_hub_download(repo, shard, subfolder="text_encoder_3", token=token)
        for shard in sorted(set(weight_map.values()))
    }
    for weight in tqdm(model.weights, desc="Transferring weights to Keras"):
        leaf = weight.path.split("/")[-2]
        if leaf in T5_ENCODER_FIXED_LEAVES:
            key = T5_ENCODER_FIXED_LEAVES[leaf]
        else:
            m = re.match(r"enc_(\d+)_(attn|ff)_(q|k|v|o|ln|wi_0|wi_1|wo)$", leaf)
            if not m:
                raise WeightMappingError(weight.path, leaf)
            idx, role, part = m.groups()
            sub, name = (
                ("0", "SelfAttention") if role == "attn" else ("1", "DenseReluDense")
            )
            if part == "ln":
                key = f"encoder.block.{idx}.layer.{sub}.layer_norm.weight"
            else:
                key = f"encoder.block.{idx}.layer.{sub}.{name}.{part}.weight"
        with safe_open(shard_paths[weight_map[key]], framework="np") as shard:
            transfer_weights(weight.path, weight, shard.get_tensor(key))
    return model, cfg


if __name__ == "__main__":
    import gc
    import os

    import keras

    OUT_DIR = os.environ.get("ZM_OUT_DIR", "C:/Users/gites/Desktop/code/v1_weights")
    os.makedirs(OUT_DIR, exist_ok=True)
    MAX_SHARD_GB = 5.0
    token = os.environ.get("HF_TOKEN")
    selected = [v for v in os.environ.get("ZM_VARIANTS", "").split(",") if v]
    sources = {
        variant: source
        for variant, source in STABLE_DIFFUSION_3_SOURCES.items()
        if not selected or variant in selected
    }

    def save(model, variant):
        n_bytes = sum(
            int(np.prod(w.shape)) * np.dtype(w.dtype).itemsize for w in model.weights
        )
        stem = os.path.join(OUT_DIR, variant.replace("-", "_").replace(".", "_"))
        if n_bytes > MAX_SHARD_GB * 1024**3:
            out = f"{stem}.weights.json"
            model.save_weights(out, max_shard_size=MAX_SHARD_GB)
        else:
            out = f"{stem}.weights.h5"
            model.save_weights(out)
        print(f"  saved -> {out}  ({n_bytes / 1024**3:.2f} GB)")

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = transfer_stable_diffusion_3(source, token=token)
        save(model, variant)
        del model
        keras.backend.clear_session()
        gc.collect()

    if not selected or T5_XXL_ENCODER_VARIANT in selected:
        source = next(iter(STABLE_DIFFUSION_3_SOURCES.values()))
        print(
            f"\n{'=' * 60}\nConverting: {T5_XXL_ENCODER_VARIANT}  <-  {source}\n{'=' * 60}"
        )
        model, cfg = transfer_t5_encoder(source, token=token)
        save(model, T5_XXL_ENCODER_VARIANT)
        del model
        keras.backend.clear_session()
        gc.collect()
