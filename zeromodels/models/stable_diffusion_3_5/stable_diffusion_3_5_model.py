import keras

from zeromodels.models.stable_diffusion_3.stable_diffusion_3_model import (
    StableDiffusion3Model,
    StableDiffusion3TextToImage,
)

from .stable_diffusion_3_5_config import StableDiffusion3_5Config

# The container and the task build the identical graph, so both load the one
# hosted repo (whose zm_config.json names the container).
STABLE_DIFFUSION_3_5_HUB_SIBLINGS = frozenset(
    {"StableDiffusion3_5Model", "StableDiffusion3_5TextToImage"}
)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3_5Model(StableDiffusion3Model):
    """Stable Diffusion 3.5 weights container: the MMDiT (large) or MMDiT-X (medium),
    the 16-channel VAE and the two CLIP text encoders as one functional
    ``keras.Model``.

    The SD 3 container (:class:`StableDiffusion3Model`) with the SD 3.5
    configuration (:class:`StableDiffusion3_5Config`): RMS-normalized queries and
    keys in every block and, for the medium checkpoint, the dual-attention blocks.
    Same components (``.transformer`` / ``.vae`` / ``.text_encoder`` /
    ``.text_encoder_2``), same separately hosted T5-XXL. Hosted under
    ``zeromodels/stable-diffusion-3.5-*`` in float16.

    Args:
        **kwargs: The flat :class:`StableDiffusion3_5Config` fields, or a config
            positionally.
    """

    config_class = StableDiffusion3_5Config
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_3_5_HUB_SIBLINGS

    def __init__(self, name="StableDiffusion3_5Model", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3_5TextToImage(StableDiffusion3TextToImage):
    """Text-to-image Stable Diffusion 3.5, pure Keras 3 and cross-backend.

    :class:`StableDiffusion3TextToImage` with the SD 3.5 configuration: the same
    ``generate`` over the SD 3.5 graph, the repo's flow-match sampler and
    defaults (28 steps at guidance 3.5 for large, 40 at 4.5 for medium, 4 without
    guidance for large-turbo)::

        sd = StableDiffusion3_5TextToImage.from_weights(
            "zeromodels/stable-diffusion-3.5-large", text_encoder_3="zeromodels/t5-v1_1-xxl-encoder"
        )
        tokenizer = StableDiffusion3_5Tokenizer.from_weights("zeromodels/stable-diffusion-3.5-large")
        image = sd.generate(**tokenizer("a photo of a cat"))  # (1, 1024, 1024, 3) uint8

    Args:
        scheduler: A :class:`BaseScheduler`; defaults to the config's
            ``scheduler_config``.
        **kwargs: The flat :class:`StableDiffusion3_5Config` fields.
    """

    config_class = StableDiffusion3_5Config
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_3_5_HUB_SIBLINGS
    # diffusers' defaults for SD 3.5 large; each repo's generate_args override
    generate_args = {"num_inference_steps": 28, "guidance_scale": 3.5}

    def __init__(self, scheduler=None, name="StableDiffusion3_5TextToImage", **kwargs):
        super().__init__(scheduler=scheduler, name=name, **kwargs)
