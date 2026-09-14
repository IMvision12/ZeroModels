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
)

STABLE_DIFFUSION_SOURCES = {
    "stable-diffusion-v1-1": "CompVis/stable-diffusion-v1-1",
    "stable-diffusion-v1-2": "CompVis/stable-diffusion-v1-2",
    "stable-diffusion-v1-3": "CompVis/stable-diffusion-v1-3",
    "stable-diffusion-v1-4": "CompVis/stable-diffusion-v1-4",
    "stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "__": ".",
    "/kernel": ".weight",
    "/gamma": ".weight",
    "/beta": ".bias",
    "/scale": ".weight",  # RMSNormalization (the SD 3.5 qk norms)
    "/": ".",
}


def config_from_diffusers(repo, token=None, config_cls=None):
    from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTokenizerFast

    from zeromodels.models.stable_diffusion.stable_diffusion_config import (
        StableDiffusionConfig,
    )
    from zeromodels.models.stable_diffusion.stable_diffusion_model import (
        UNet2DConditionModel as KerasUNet2DConditionModel,
    )

    config_cls = config_cls or StableDiffusionConfig
    unet = dict(UNet2DConditionModel.load_config(repo, subfolder="unet", token=token))
    vae = dict(AutoencoderKL.load_config(repo, subfolder="vae", token=token))
    text = CLIPTextConfig.from_pretrained(
        repo, subfolder="text_encoder", token=token
    ).to_dict()
    # load_config only reads the json, whichever scheduler the repo declares
    scheduler = {
        k: v
        for k, v in PNDMScheduler.load_config(
            repo, subfolder="scheduler", token=token
        ).items()
        if k == "_class_name" or not k.startswith("_")
    }
    # the pad token differs between checkpoints (<|endoftext|> 49407, "!" 0)
    tokenizer = CLIPTokenizerFast.from_pretrained(
        repo, subfolder="tokenizer", token=token
    )
    hidden = text["hidden_size"]
    return config_cls(
        unet_config=KerasUNet2DConditionModel.kwargs_from_diffusers_config(unet),
        vae_config={
            "in_channels": vae.get("in_channels", 3),
            "out_channels": vae.get("out_channels", 3),
            "latent_channels": vae.get("latent_channels", 4),
            "block_out_channels": tuple(vae["block_out_channels"]),
            "layers_per_block": vae.get("layers_per_block", 2),
            "norm_num_groups": vae.get("norm_num_groups", 32),
            # the pipeline resolution is the UNet's latent size x the VAE's
            # compression; the VAE config's own sample_size is not what the
            # checkpoint generates at (768 on the 512px SD 2.1-base repo)
            "sample_size": unet.get("sample_size", 64)
            * 2 ** (len(vae["block_out_channels"]) - 1),
            "scaling_factor": vae.get("scaling_factor")
            or 0.18215,  # null in some repos
            "force_upcast": bool(vae.get("force_upcast", False)),
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
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def transfer_stable_diffusion(repo, token=None, model_cls=None, config_cls=None):
    import gc

    import torch
    from diffusers import AutoencoderKL as DiffusersAutoencoderKL
    from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel
    from transformers import CLIPTextModel as HFCLIPTextModel

    from zeromodels.models.clip import CLIPTextModel
    from zeromodels.models.stable_diffusion.stable_diffusion_model import (
        StableDiffusionModel,
    )

    model_cls = model_cls or StableDiffusionModel
    config = config_from_diffusers(repo, token=token, config_cls=config_cls)
    model = model_cls(config)
    load = {"torch_dtype": torch.float32, "token": token}

    # the SD 1.x era VAE files spell the mid-block attention query / key / value /
    # proj_attn (diffusers renames them on load, a raw checkpoint read keeps them):
    # map them to the to_q / to_k / to_v / to_out.0 names the layers use
    legacy = {
        ".query.": ".to_q.",
        ".key.": ".to_k.",
        ".value.": ".to_v.",
        ".proj_attn.": ".to_out.0.",
    }
    for component, module_cls, subfolder in (
        (model.unet, DiffusersUNet2DConditionModel, "unet"),
        (model.vae, DiffusersAutoencoderKL, "vae"),
    ):
        module = module_cls.from_pretrained(repo, subfolder=subfolder, **load)
        state = {}
        for key, value in module.state_dict().items():
            if subfolder == "vae" and ".attentions." in key:
                for old, new in legacy.items():
                    key = key.replace(old, new)
            state[key] = value.detach().cpu().numpy()
        del module
        consumed = set()
        trainable, non_trainable = split_model_weights(component)
        for keras_weight, _ in tqdm(
            trainable + non_trainable, desc=f"Transferring {subfolder} weights to Keras"
        ):
            key = "/".join(keras_weight.path.split("/")[-2:])
            for old, new in WEIGHT_NAME_MAPPING.items():
                key = key.replace(old, new)
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
            if key.startswith("time_embedding.") and keras_weight.ndim == 2:
                # transfer_weights treats any "embedding" 2D weight as a lookup
                # table; the timestep MLP's linears are dense kernels
                keras_weight.assign(np.transpose(torch_weight))
                continue
            transfer_weights(key, keras_weight, torch_weight)
        unused = sorted(set(state) - consumed)
        if unused:
            raise ValueError(
                f"{type(component).__name__}: {len(unused)} checkpoint tensors "
                f"unused, e.g. {unused[:3]}."
            )
        del state
        gc.collect()

    text_encoder = HFCLIPTextModel.from_pretrained(
        repo, subfolder="text_encoder", **load
    )
    text_state = {
        k if k.startswith("text_model.") else f"text_model.{k}": v
        for k, v in {
            k: v.detach().cpu().numpy() for k, v in text_encoder.state_dict().items()
        }.items()
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
        model, config = transfer_stable_diffusion(source, token=token)

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
