import numpy as np

from zeromodels.models.stable_diffusion_3.convert_stable_diffusion_3_diffusers_to_keras import (
    config_from_diffusers as stable_diffusion_3_config_from_diffusers,
)
from zeromodels.models.stable_diffusion_3.convert_stable_diffusion_3_diffusers_to_keras import (
    transfer_stable_diffusion_3,
)

STABLE_DIFFUSION_3_5_SOURCES = {
    "stable-diffusion-3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "stable-diffusion-3.5-large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "stable-diffusion-3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
}


def config_from_diffusers(repo, token=None):
    from zeromodels.models.stable_diffusion_3_5.stable_diffusion_3_5_config import (
        StableDiffusion3_5Config,
    )

    return stable_diffusion_3_config_from_diffusers(
        repo, token=token, config_cls=StableDiffusion3_5Config
    )


def transfer_stable_diffusion_3_5(repo, token=None, dtype="float16"):
    from zeromodels.models.stable_diffusion_3_5.stable_diffusion_3_5_config import (
        StableDiffusion3_5Config,
    )
    from zeromodels.models.stable_diffusion_3_5.stable_diffusion_3_5_model import (
        StableDiffusion3_5Model,
    )

    return transfer_stable_diffusion_3(
        repo,
        token=token,
        dtype=dtype,
        model_cls=StableDiffusion3_5Model,
        config_cls=StableDiffusion3_5Config,
    )


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
        for variant, source in STABLE_DIFFUSION_3_5_SOURCES.items()
        if not selected or variant in selected
    }

    for variant, source in sources.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {source}\n{'=' * 60}")
        model, config = transfer_stable_diffusion_3_5(source, token=token)

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
