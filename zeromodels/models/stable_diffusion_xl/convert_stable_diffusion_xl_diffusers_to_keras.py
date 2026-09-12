import collections.abc

import numpy as np

from zeromodels.models.clip.convert_clip_hf_to_keras import transfer_clip_weights
from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    numpy_state_dict,
)

# SDXL ships in float16 (the release checkpoints' precision; the diffusers fp32
# files are upcasts of the fp16 variant), so the container is built and saved in
# float16 (the VAE excepted: force_upcast keeps it float32). Loading a repo
# rebuilds at that precision by default; load_dtype="float32" upcasts. The 0.9
# repos are gated (research license): set HF_TOKEN to convert them. The refiner
# repos (an image-to-image model: one text tower, a four-level UNet) convert to
# StableDiffusionXLRefinerModel, everything else to StableDiffusionXLModel.
STABLE_DIFFUSION_XL_SOURCES = {
    "stable-diffusion-xl-base-0.9": "stabilityai/stable-diffusion-xl-base-0.9",
    "stable-diffusion-xl-refiner-0.9": "stabilityai/stable-diffusion-xl-refiner-0.9",
    "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-xl-refiner-1.0": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}


class LazyNumpyStateDict(collections.abc.Mapping):
    def __init__(self, module):
        self.state = module.state_dict()

    def __getitem__(self, key):
        return self.state[key].detach().cpu().numpy()

    def __contains__(self, key):
        return key in self.state

    def __iter__(self):
        return iter(self.state)

    def __len__(self):
        return len(self.state)


def diffusers_configs(repo, token=None):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from transformers import CLIPTextConfig, CLIPTokenizerFast

    pipeline = dict(DiffusionPipeline.load_config(repo, token=token))
    has_text = pipeline.get("text_encoder", [None])[0] is not None  # refiner: no
    tokenizer_2 = CLIPTokenizerFast.from_pretrained(
        repo, subfolder="tokenizer_2", token=token
    )
    tokenizer = (
        CLIPTokenizerFast.from_pretrained(repo, subfolder="tokenizer", token=token)
        if has_text
        else tokenizer_2
    )
    return {
        "unet": dict(
            UNet2DConditionModel.load_config(repo, subfolder="unet", token=token)
        ),
        "vae": dict(AutoencoderKL.load_config(repo, subfolder="vae", token=token)),
        "text": CLIPTextConfig.from_pretrained(
            repo, subfolder="text_encoder", token=token
        ).to_dict()
        if has_text
        else None,
        "text_2": CLIPTextConfig.from_pretrained(
            repo, subfolder="text_encoder_2", token=token
        ).to_dict(),
        "scheduler": dict(
            EulerDiscreteScheduler.load_config(repo, subfolder="scheduler", token=token)
        ),
        "force_zeros_for_empty_prompt": bool(
            pipeline.get("force_zeros_for_empty_prompt", True)
        ),
        "requires_aesthetics_score": bool(
            pipeline.get("requires_aesthetics_score", False)
        ),
        "tokens": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "pad_token_id_2": tokenizer_2.pad_token_id,
        },
    }


def text_config_kwargs(text):
    hidden = text["hidden_size"]
    return {
        "hidden_dim": hidden,
        "num_heads": text.get("num_attention_heads", 12),
        "num_layers": text.get("num_hidden_layers", 12),
        "mlp_ratio": text.get("intermediate_size", hidden * 4) / hidden,
        "vocab_size": text.get("vocab_size", 49408),
        "max_seq_len": text.get("max_position_embeddings", 77),
    }


def config_from_diffusers(repo, token=None):
    from zeromodels.models.stable_diffusion.stable_diffusion_model import (
        UNet2DConditionModel,
    )
    from zeromodels.models.stable_diffusion_xl.stable_diffusion_xl_config import (
        StableDiffusionXLConfig,
        StableDiffusionXLRefinerConfig,
    )

    src = diffusers_configs(repo, token=token)
    unet, vae, text, text_2 = src["unet"], src["vae"], src["text"], src["text_2"]
    scheduler = {
        k: v
        for k, v in src["scheduler"].items()
        if k == "_class_name" or not k.startswith("_")
    }
    unet_kwargs = UNet2DConditionModel.kwargs_from_diffusers_config(unet)
    if src["requires_aesthetics_score"]:
        # size (2) + crop (2) + aesthetic score (1) next to the pooled embedding
        unet_kwargs["num_time_ids"] = 5
    config_cls = (
        StableDiffusionXLRefinerConfig if text is None else StableDiffusionXLConfig
    )
    return config_cls(
        unet_config=unet_kwargs,
        vae_config={
            "in_channels": vae.get("in_channels", 3),
            "out_channels": vae.get("out_channels", 3),
            "latent_channels": vae.get("latent_channels", 4),
            "block_out_channels": tuple(vae["block_out_channels"]),
            "layers_per_block": vae.get("layers_per_block", 2),
            "norm_num_groups": vae.get("norm_num_groups", 32),
            "sample_size": unet.get("sample_size", 128)
            * 2 ** (len(vae["block_out_channels"]) - 1),
            "scaling_factor": vae.get("scaling_factor") or 0.13025,
            "force_upcast": bool(vae.get("force_upcast", True)),
        },
        text_config=None if text is None else text_config_kwargs(text),
        text_config_2={
            **text_config_kwargs(text_2),
            "projection_dim": text_2.get("projection_dim", 1280),
            "hidden_act": text_2.get("hidden_act", "gelu"),
        },
        hidden_act=(text or {}).get("hidden_act", "quick_gelu"),
        layer_norm_eps=(text or text_2).get("layer_norm_eps", 1e-5),
        scheduler_config=scheduler,
        force_zeros_for_empty_prompt=src["force_zeros_for_empty_prompt"],
        requires_aesthetics_score=src["requires_aesthetics_score"],
        **src["tokens"],
    )


def transfer_text_encoder(keras_model, hf_module):
    state = {
        k if k.startswith(("text_model.", "text_projection.")) else f"text_model.{k}": v
        for k, v in numpy_state_dict(hf_module).items()
    }
    transfer_clip_weights(keras_model, state)


def build_from_diffusers(repo, token=None, dtype="float16"):
    import gc

    import torch
    from diffusers import AutoencoderKL as DiffusersAutoencoderKL
    from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    from zeromodels.base.base_mixin import build_dtype_scope
    from zeromodels.models.stable_diffusion.stable_diffusion_model import (
        UNet2DConditionModel,
    )
    from zeromodels.models.stable_diffusion_xl.stable_diffusion_xl_model import (
        StableDiffusionXLModel,
        StableDiffusionXLRefinerModel,
    )

    config = config_from_diffusers(repo, token=token)
    model_cls = (
        StableDiffusionXLRefinerModel
        if config.text_config is None
        else StableDiffusionXLModel
    )
    with build_dtype_scope(dtype):
        model = model_cls(config)
    fp16 = {"torch_dtype": torch.float16, "variant": "fp16", "token": token}
    fp32 = {"torch_dtype": torch.float32, "token": token}

    unet = DiffusersUNet2DConditionModel.from_pretrained(repo, subfolder="unet", **fp16)
    UNet2DConditionModel.transfer_from_hf(model.unet, LazyNumpyStateDict(unet))
    del unet
    gc.collect()

    vae = DiffusersAutoencoderKL.from_pretrained(repo, subfolder="vae", **fp32)
    model.vae.transfer_from_hf(numpy_state_dict(vae))
    del vae
    gc.collect()

    if config.text_config is not None:
        text_encoder = CLIPTextModel.from_pretrained(
            repo, subfolder="text_encoder", **fp32
        )
        transfer_text_encoder(model.text_encoder, text_encoder)
        del text_encoder
        gc.collect()

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        repo, subfolder="text_encoder_2", **fp32
    )
    transfer_text_encoder(model.text_encoder_2, text_encoder_2)
    del text_encoder_2
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
        for variant, source in STABLE_DIFFUSION_XL_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = build_from_diffusers(source, token=token)

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

        del model
        keras.backend.clear_session()
        gc.collect()
