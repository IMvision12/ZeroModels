import collections.abc

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

STABLE_DIFFUSION_SOURCES = {
    "stable-diffusion-v1-1": "CompVis/stable-diffusion-v1-1",
    "stable-diffusion-v1-2": "CompVis/stable-diffusion-v1-2",
    "stable-diffusion-v1-3": "CompVis/stable-diffusion-v1-3",
    "stable-diffusion-v1-4": "CompVis/stable-diffusion-v1-4",
    "stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}

LEGACY_ATTENTION_KEYS = {
    ".query.": ".to_q.",
    ".key.": ".to_k.",
    ".value.": ".to_v.",
    ".proj_attn.": ".to_out.0.",
}


class RenamedStateDict(collections.abc.Mapping):
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.to_source = {}
        for key in state_dict:
            new = key
            if ".attentions." in key:
                for old, current in LEGACY_ATTENTION_KEYS.items():
                    new = new.replace(old, current)
            self.to_source[new] = key

    def __getitem__(self, key):
        return self.state_dict[self.to_source[key]]

    def __contains__(self, key):
        return key in self.to_source

    def __iter__(self):
        return iter(self.to_source)

    def __len__(self):
        return len(self.to_source)


WEIGHT_SUFFIX = {"kernel": "weight", "bias": "bias", "gamma": "weight", "beta": "bias"}

SUBMODEL_PREFIX = {"vae_encoder": "encoder.", "vae_decoder": "decoder."}

ATTN_NAME_REPLACE = {
    "..": ".",
    "down.blocks": "down_blocks",
    "up.blocks": "up_blocks",
    "mid.block": "mid_block",
    "transformer.blocks": "transformer_blocks",
    "proj.in": "proj_in",
    "proj.out": "proj_out",
    "to.q": "to_q",
    "to.k": "to_k",
    "to.v": "to_v",
    "to.out": "to_out",
    "group.norm": "group_norm",
}


def torch_key(keras_weight_name, keras_weight):
    top_layer = keras_weight_name.rsplit("_", 1)[0]
    leaf, variable = keras_weight.path.split("/")[-2:]
    prefix = SUBMODEL_PREFIX.get(top_layer, "")
    return f"{prefix}{leaf.replace('__', '.')}.{WEIGHT_SUFFIX[variable]}"


def transfer_component(keras_model, state):
    scoped = {
        prefix: {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        for prefix in set(SUBMODEL_PREFIX.values())
    }
    consumed = set()
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        key = torch_key(keras_weight_name, keras_weight)
        prefix = SUBMODEL_PREFIX.get(keras_weight_name.rsplit("_", 1)[0], "")
        consumed.add(key)

        if "attention" in key:
            transfer_attention_weights(
                keras_weight.path,
                keras_weight,
                scoped[prefix] if prefix else state,
                ATTN_NAME_REPLACE,
            )
            continue

        if key not in state:
            raise WeightMappingError(keras_weight.path, key)
        torch_weight = state[key]
        if not compare_keras_torch_names(
            keras_weight.path, keras_weight, key, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight.path, keras_weight.shape, key, torch_weight.shape
            )
        if key.startswith("time_embedding.") and keras_weight.ndim == 2:
            # transfer_weights treats any "embedding" 2D weight as a lookup table;
            # the timestep MLP's linears are plain dense kernels
            keras_weight.assign(np.transpose(torch_weight))
            continue
        transfer_weights(key, keras_weight, torch_weight)

    unused = sorted(set(state) - consumed)
    if unused:
        raise ValueError(
            f"{type(keras_model).__name__}: {len(unused)} checkpoint tensors unused, "
            f"e.g. {unused[:3]}."
        )


def numpy_state_dict(module):
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def diffusers_configs(repo, token=None):
    from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig

    return {
        "unet": dict(
            UNet2DConditionModel.load_config(repo, subfolder="unet", token=token)
        ),
        "vae": dict(AutoencoderKL.load_config(repo, subfolder="vae", token=token)),
        "text": CLIPTextConfig.from_pretrained(
            repo, subfolder="text_encoder", token=token
        ).to_dict(),
        # load_config only reads the json, whichever scheduler the repo declares
        "scheduler": dict(
            PNDMScheduler.load_config(repo, subfolder="scheduler", token=token)
        ),
    }


def config_from_diffusers(repo, token=None):
    from zeromodels.models.stable_diffusion.stable_diffusion_config import (
        StableDiffusionConfig,
    )

    src = diffusers_configs(repo, token=token)
    unet, vae, text = src["unet"], src["vae"], src["text"]
    scheduler = {
        k: v
        for k, v in src["scheduler"].items()
        if k == "_class_name" or not k.startswith("_")
    }

    heads = unet.get("num_attention_heads") or unet.get("attention_head_dim", 8)
    if isinstance(heads, (list, tuple)):
        heads = heads[0]
    hidden = text["hidden_size"]
    return StableDiffusionConfig(
        unet_config={
            "sample_size": unet.get("sample_size", 64),
            "in_channels": unet.get("in_channels", 4),
            "out_channels": unet.get("out_channels", 4),
            "down_block_types": tuple(unet["down_block_types"]),
            "up_block_types": tuple(unet["up_block_types"]),
            "block_out_channels": tuple(unet["block_out_channels"]),
            "layers_per_block": unet.get("layers_per_block", 2),
            "cross_attention_dim": unet.get("cross_attention_dim", 768),
            "num_attention_heads": heads,
            "norm_num_groups": unet.get("norm_num_groups", 32),
        },
        vae_config={
            "in_channels": vae.get("in_channels", 3),
            "out_channels": vae.get("out_channels", 3),
            "latent_channels": vae.get("latent_channels", 4),
            "block_out_channels": tuple(vae["block_out_channels"]),
            "layers_per_block": vae.get("layers_per_block", 2),
            "norm_num_groups": vae.get("norm_num_groups", 32),
            "sample_size": vae.get("sample_size", 512),
            "scaling_factor": vae.get("scaling_factor", 0.18215),
        },
        text_config={
            "hidden_dim": hidden,
            "num_heads": text.get("num_attention_heads", 12),
            "num_layers": text.get("num_hidden_layers", 12),
            "mlp_ratio": text.get("intermediate_size", hidden * 4) / hidden,
            "vocab_size": text.get("vocab_size", 49408),
            "max_seq_len": text.get("max_position_embeddings", 77),
        },
        hidden_act=text.get("hidden_act", "quick_gelu"),
        layer_norm_eps=text.get("layer_norm_eps", 1e-5),
        scheduler_config=scheduler,
    )


def build_from_diffusers(repo, token=None):
    import gc

    import torch
    from diffusers import AutoencoderKL as DiffusersAutoencoderKL
    from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel
    from transformers import CLIPTextModel as HFCLIPTextModel

    from zeromodels.models.clip import CLIPTextModel
    from zeromodels.models.stable_diffusion.stable_diffusion_model import (
        StableDiffusionModel,
        UNet2DConditionModel,
    )

    config = config_from_diffusers(repo, token=token)
    model = StableDiffusionModel(config)
    load = {"torch_dtype": torch.float32, "token": token}

    unet = DiffusersUNet2DConditionModel.from_pretrained(repo, subfolder="unet", **load)
    UNet2DConditionModel.transfer_from_hf(model.unet, numpy_state_dict(unet))
    del unet
    gc.collect()

    vae = DiffusersAutoencoderKL.from_pretrained(repo, subfolder="vae", **load)
    model.vae.transfer_from_hf(numpy_state_dict(vae))
    del vae
    gc.collect()

    text_encoder = HFCLIPTextModel.from_pretrained(
        repo, subfolder="text_encoder", **load
    )
    text_state = {
        k if k.startswith("text_model.") else f"text_model.{k}": v
        for k, v in numpy_state_dict(text_encoder).items()
    }
    CLIPTextModel.transfer_from_hf(model.text_encoder, text_state)
    del text_encoder
    gc.collect()
    return model, config


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
        for variant, source in STABLE_DIFFUSION_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = build_from_diffusers(source, token=token)

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        stem = os.path.join(OUT_DIR, variant.replace("-", "_"))
        if n_bytes > MAX_SHARD_GB * 1024**3:
            out = f"{stem}.weights.json"
            model.save_weights(out, max_shard_size=MAX_SHARD_GB)
        else:
            out = f"{stem}.weights.h5"
            model.save_weights(out)
        print(f"  saved -> {out}  ({n_bytes / 1024**3:.2f} GB fp32)")

        del model
        keras.backend.clear_session()
        gc.collect()
