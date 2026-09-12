from zeromodels.models.clip.clip_config import CLIPTextConfig
from zeromodels.models.stable_diffusion.stable_diffusion_config import (
    StableDiffusionConfig,
    UNet2DConditionConfig,
)


class StableDiffusion2UNetConfig(UNet2DConditionConfig):
    r"""The Stable Diffusion 2.x denoiser: a [`UNet2DConditionConfig`] with the SD 2
    widths.

    Same fields as [`UNet2DConditionConfig`]; the defaults are the SD 2.x UNet (865M
    parameters): a 1024-d text context (OpenCLIP ViT-H/14), one head count per level
    ((5, 10, 20, 20), a 64-wide head everywhere) and a linear token projection in the
    Transformer2D blocks. ``sample_size`` is 64 for the 512px checkpoints and 96 for
    the 768px ones."""

    cross_attention_dim: int = 1024
    num_attention_heads: int | tuple = (5, 10, 20, 20)
    use_linear_projection: bool = True


class StableDiffusion2TextConfig(CLIPTextConfig):
    r"""The Stable Diffusion 2.x text tower: OpenCLIP ViT-H/14's text encoder.

    Same fields as [`CLIPTextConfig`], defaulted to the ViT-H/14 sizes (1024 wide,
    16 heads) with 23 layers: the checkpoints ship the encoder truncated to its
    penultimate layer, which SD 2 conditions on. The activation is ``gelu`` (set
    through the top-level ``hidden_act``)."""

    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 23


class StableDiffusion2Config(StableDiffusionConfig):
    r"""Configuration for [`StableDiffusion2Model`], the hosted SD 2.x weights container.

    Same shape as [`StableDiffusionConfig`] (one nested ``unet_config`` /
    ``vae_config`` / ``text_config`` plus the scheduler and token ids; flat
    constructor), with the SD 2.x defaults: the [`StableDiffusion2UNetConfig`]
    denoiser, the [`StableDiffusion2TextConfig`] OpenCLIP-H text tower with ``gelu``,
    and ``!`` (id 0) as the pad token. The four SD 2.x checkpoints (2-base, 2,
    2-1-base, 2-1) share this architecture up to ``sample_size``; the 768px ones ship
    a v-prediction scheduler.

    Args:
        unet_config ([`StableDiffusion2UNetConfig`], *optional*): The denoiser.
        vae_config ([`AutoencoderKLConfig`], *optional*): The VAE.
        text_config ([`StableDiffusion2TextConfig`], *optional*): The OpenCLIP
            ViT-H/14 text encoder.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Text encoder activation.
        pad_token_id (`int`, *optional*, defaults to 0):
            OpenCLIP pads with ``!`` (id 0) rather than ``<|endoftext|>``.

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_2 import StableDiffusion2Config, StableDiffusion2Model

    >>> config = StableDiffusion2Config()
    >>> model = StableDiffusion2Model(config)  # random weights, SD 2.x shapes
    >>> model.config.unet_config.num_attention_heads
    (5, 10, 20, 20)
    ```"""

    model_type = "stable_diffusion_2"

    sub_configs = {
        "unet_config": StableDiffusion2UNetConfig,
        "vae_config": StableDiffusionConfig.sub_configs["vae_config"],
        "text_config": StableDiffusion2TextConfig,
    }

    unet_config: StableDiffusion2UNetConfig | dict | None = None
    text_config: StableDiffusion2TextConfig | dict | None = None
    hidden_act: str = "gelu"
    pad_token_id: int = 0
