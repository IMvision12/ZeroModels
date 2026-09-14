import keras

from zeromodels.models.stable_diffusion.stable_diffusion_model import (
    StableDiffusionModel,
    StableDiffusionTextToImage,
)

from .stable_diffusion_2_config import StableDiffusion2Config

# The container and the task build the identical graph, so both load the one
# hosted repo (whose zm_config.json names the container).
STABLE_DIFFUSION_2_HUB_SIBLINGS = frozenset(
    {"StableDiffusion2Model", "StableDiffusion2TextToImage"}
)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion2Model(StableDiffusionModel):
    """Stable Diffusion 2.x weights container: the UNet, the VAE and the OpenCLIP
    ViT-H/14 text encoder as one functional ``keras.Model``.

    The SD 1.x container with the SD 2.x configuration
    (:class:`StableDiffusion2Config`): the same three disconnected sub-graphs and
    the same components (``.unet`` / ``.vae`` / ``.text_encoder``), with a 1024-d
    text context, one attention head count per UNet level, a linear token
    projection in the Transformer2D blocks and the 23-layer ``gelu`` text tower.
    Hosted as one ``zm_config.json`` + ``model.weights.h5`` per checkpoint under
    ``zeromodels/stable-diffusion-2*``; on-the-fly ``hf:`` conversion is not
    supported for diffusion models.

    Args:
        **kwargs: The flat :class:`StableDiffusion2Config` fields (``unet_`` /
            ``vae_`` / ``text_`` prefixed), or a config positionally. Pass
            ``unet_sample_size=96, vae_sample_size=768`` for the 768px checkpoints
            (their repos already carry it).
    """

    config_class = StableDiffusion2Config
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_2_HUB_SIBLINGS

    def __init__(self, name="StableDiffusion2Model", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion2TextToImage(StableDiffusionTextToImage):
    """Text-to-image Stable Diffusion 2.x, pure Keras 3 and cross-backend.

    :class:`StableDiffusionTextToImage` with the SD 2.x configuration: the same
    ``generate`` (``BaseDiffusion``) over the SD 2 graph, so the checkpoint's
    ``!``-padded empty prompt drives classifier-free guidance and its scheduler
    (PNDM for ``stable-diffusion-2-1-base``, DDIM for the others, v-prediction for
    the 768px checkpoints) comes from the repo's ``scheduler_config``::

        sd = StableDiffusion2TextToImage.from_weights("zeromodels/stable-diffusion-2-1")
        tokenizer = StableDiffusion2Tokenizer.from_weights("zeromodels/stable-diffusion-2-1")
        image = sd.generate(**tokenizer("a photo of a cat"))  # (1, 768, 768, 3) uint8

    Args:
        scheduler: A :class:`BaseScheduler`; defaults to the config's
            ``scheduler_config``.
        **kwargs: The flat :class:`StableDiffusion2Config` fields.
    """

    config_class = StableDiffusion2Config
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_2_HUB_SIBLINGS

    def __init__(self, scheduler=None, name="StableDiffusion2TextToImage", **kwargs):
        super().__init__(scheduler=scheduler, name=name, **kwargs)
