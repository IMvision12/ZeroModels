import numpy as np

from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    config_from_diffusers as stable_diffusion_config_from_diffusers,
)
from zeromodels.models.stable_diffusion.convert_stable_diffusion_diffusers_to_keras import (
    transfer_stable_diffusion,
)

# Stable Diffusion 2.x: one architecture, five checkpoints (the 768px ones are
# v-prediction models built at sample_size 96; SD-Turbo is the 512px 2.1 distilled
# to a few steps). The stabilityai SD 2 repos are no longer on the Hub;
# sd2-community mirrors them in the same diffusers layout. This script only
# converts weights, the SD 1.x way (see that converter); the hosting tooling
# writes zm_config.json / tokenizer.json.
STABLE_DIFFUSION_2_SOURCES = {
    "stable-diffusion-2-base": "sd2-community/stable-diffusion-2-base",
    "stable-diffusion-2": "sd2-community/stable-diffusion-2",
    "stable-diffusion-2-1-base": "sd2-community/stable-diffusion-2-1-base",
    "stable-diffusion-2-1": "sd2-community/stable-diffusion-2-1",
    # SD 2.1 distilled for 1 to 4 steps without guidance (Euler, trailing spacing)
    "sd-turbo": "stabilityai/sd-turbo",
}


def config_from_diffusers(repo, token=None):
    from zeromodels.models.stable_diffusion_2.stable_diffusion_2_config import (
        StableDiffusion2Config,
    )

    return stable_diffusion_config_from_diffusers(
        repo, token=token, config_cls=StableDiffusion2Config
    )


def transfer_stable_diffusion_2(repo, token=None):
    from zeromodels.models.stable_diffusion_2.stable_diffusion_2_config import (
        StableDiffusion2Config,
    )
    from zeromodels.models.stable_diffusion_2.stable_diffusion_2_model import (
        StableDiffusion2Model,
    )

    return transfer_stable_diffusion(
        repo,
        token=token,
        model_cls=StableDiffusion2Model,
        config_cls=StableDiffusion2Config,
    )


if __name__ == "__main__":
    import gc
    import os

    import keras

    OUT_DIR = os.environ.get("ZM_OUT_DIR", "C:/Users/gites/Desktop/code/v1_weights")
    os.makedirs(OUT_DIR, exist_ok=True)
    MAX_SHARD_GB = 5.0
    token = os.environ.get("HF_TOKEN")
    # ZM_VARIANTS="stable-diffusion-2-1,stable-diffusion-2-1-base" converts a subset
    # (each variant is a ~5 GB download and a ~5 GB output); default: all four.
    selected = [v for v in os.environ.get("ZM_VARIANTS", "").split(",") if v]
    sources = {
        variant: source
        for variant, source in STABLE_DIFFUSION_2_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = transfer_stable_diffusion_2(source, token=token)

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
