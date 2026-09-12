from zeromodels.models.clip.clip_config import CLIPTextConfig
from zeromodels.models.stable_diffusion.stable_diffusion_config import (
    AutoencoderKLConfig,
    StableDiffusionConfig,
    StableDiffusionTextConfig,
    UNet2DConditionConfig,
)


class StableDiffusionXLUNetConfig(UNet2DConditionConfig):
    r"""The Stable Diffusion XL denoiser: a [`UNet2DConditionConfig`] with the SDXL
    layout.

    Same fields as [`UNet2DConditionConfig`]; the defaults are the SDXL base UNet
    (2.57B parameters): three levels ((320, 640, 1280) wide, no attention at the
    first), a 2048-d text context (the two text encoders concatenated), one head
    count per level ((5, 10, 20), a 64-wide head everywhere), (1, 2, 10) transformer
    blocks per Transformer2D, a linear token projection and the ``text_time``
    micro-conditioning (the pooled 1280-d text embedding plus six size / crop ids
    embedded to 256 each, 2816 in all). ``sample_size`` is 128 (1024px); SDXL-Turbo
    ships 64 (512px)."""

    sample_size: int = 128
    down_block_types: tuple = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types: tuple = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D")
    block_out_channels: tuple = (320, 640, 1280)
    cross_attention_dim: int = 2048
    num_attention_heads: int | tuple = (5, 10, 20)
    use_linear_projection: bool = True
    transformer_layers_per_block: int | tuple = (1, 2, 10)
    addition_embed_type: str | None = "text_time"
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int | None = 2816
    num_time_ids: int = 6


class StableDiffusionXLVAEConfig(AutoencoderKLConfig):
    r"""The Stable Diffusion XL VAE: an [`AutoencoderKLConfig`] built for 1024px with
    the SDXL latent scaling (0.13025) and ``force_upcast`` on (this VAE overflows in
    float16, so it is always built in float32)."""

    sample_size: int = 1024
    scaling_factor: float = 0.13025
    force_upcast: bool = True


class StableDiffusionXLTextConfig2(CLIPTextConfig):
    r"""The second Stable Diffusion XL text tower: OpenCLIP ViT-bigG/14's text encoder.

    Same fields as [`CLIPTextConfig`], defaulted to the ViT-bigG/14 sizes (1280 wide,
    20 heads, 32 layers), plus the joint-space projection whose pooled output is the
    UNet's ``text_embeds`` conditioning.

    Args:
        projection_dim (`int`, *optional*, defaults to 1280):
            Width of the ``text_projection`` applied to the pooled (EOT) state.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation of this tower (the first tower uses the top-level
            ``hidden_act``, ``quick_gelu``)."""

    hidden_dim: int = 1280
    num_heads: int = 20
    num_layers: int = 32
    projection_dim: int = 1280
    hidden_act: str = "gelu"


class StableDiffusionXLConfig(StableDiffusionConfig):
    r"""Configuration for [`StableDiffusionXLModel`], the hosted SDXL weights container.

    The [`StableDiffusionConfig`] shape (nested sub-configs plus the scheduler and
    token ids; flat constructor) with a fourth component: SDXL conditions on two
    text encoders, CLIP ViT-L/14's (``text_config``, the SD 1.x tower) and OpenCLIP
    ViT-bigG/14's (``text_config_2``, with a 1280-d projection), whose penultimate
    hidden states are concatenated into the UNet's 2048-d context while the second
    tower's projected pooled state feeds the ``text_time`` micro-conditioning. The
    flat constructor prefixes the second tower's fields ``text_2_``
    (``text_2_hidden_dim``, ...).

    Args:
        unet_config ([`StableDiffusionXLUNetConfig`], *optional*): The denoiser.
        vae_config ([`StableDiffusionXLVAEConfig`], *optional*): The VAE.
        text_config ([`StableDiffusionTextConfig`], *optional*): The CLIP ViT-L/14
            text encoder.
        text_config_2 ([`StableDiffusionXLTextConfig2`], *optional*): The OpenCLIP
            ViT-bigG/14 text encoder.
        pad_token_id_2 (`int`, *optional*, defaults to 0):
            The second tokenizer pads with ``!`` (id 0) where the first pads with
            ``<|endoftext|>``; the model derives the second tower's ids from the
            first tokenizer's output and this id.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to True):
            Use zero embeddings, rather than the encoded empty prompt, as the
            unconditional branch of classifier-free guidance when no negative
            prompt is given (the SDXL repos' ``model_index.json`` setting).
        requires_aesthetics_score (`bool`, *optional*, defaults to False):
            The refiner's micro-conditioning: an aesthetic score as the fifth time
            id instead of the target size (``num_time_ids`` 5).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_xl import StableDiffusionXLConfig, StableDiffusionXLModel

    >>> config = StableDiffusionXLConfig()
    >>> model = StableDiffusionXLModel(config)  # random weights, SDXL shapes
    >>> model.config.unet_config.transformer_layers_per_block
    (1, 2, 10)
    ```"""

    model_type = "stable_diffusion_xl"

    sub_configs = {
        "unet_config": StableDiffusionXLUNetConfig,
        "vae_config": StableDiffusionXLVAEConfig,
        "text_config": StableDiffusionTextConfig,
        "text_config_2": StableDiffusionXLTextConfig2,
    }
    sub_config_prefixes = {
        "unet_config": "unet_",
        "vae_config": "vae_",
        "text_config": "text_",
        "text_config_2": "text_2_",
    }
    group_extras = {"text_config": ("vocab_size", "max_seq_len")}

    unet_config: StableDiffusionXLUNetConfig | dict | None = None
    vae_config: StableDiffusionXLVAEConfig | dict | None = None
    text_config_2: StableDiffusionXLTextConfig2 | dict | None = None
    pad_token_id_2: int = 0
    force_zeros_for_empty_prompt: bool = True
    requires_aesthetics_score: bool = False


class StableDiffusionXLRefinerUNetConfig(StableDiffusionXLUNetConfig):
    r"""The Stable Diffusion XL refiner's denoiser: a [`StableDiffusionXLUNetConfig`]
    with the refiner layout.

    Four levels ((384, 768, 1536, 1536) wide, attention at the middle two), 4
    transformer blocks per Transformer2D, (6, 12, 24, 24) heads, a 1280-d context
    (the OpenCLIP ViT-bigG/14 tower alone) and five micro-conditioning values (size,
    crop and the aesthetic score, 2560 with the pooled embedding). 2.3B parameters."""

    down_block_types: tuple = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: tuple = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    block_out_channels: tuple = (384, 768, 1536, 1536)
    cross_attention_dim: int = 1280
    num_attention_heads: int | tuple = (6, 12, 24, 24)
    transformer_layers_per_block: int | tuple = 4
    projection_class_embeddings_input_dim: int | None = 2560
    num_time_ids: int = 5


class StableDiffusionXLRefinerConfig(StableDiffusionXLConfig):
    r"""Configuration for [`StableDiffusionXLRefinerModel`], the hosted SDXL refiner
    weights container.

    The [`StableDiffusionXLConfig`] shape with the refiner's parts: no first text
    tower (``text_config`` is ``None``; the OpenCLIP ViT-bigG/14 ``text_config_2``
    alone conditions the UNet, 1280-d), the [`StableDiffusionXLRefinerUNetConfig`]
    denoiser and the aesthetic-score micro-conditioning
    (``requires_aesthetics_score``); the empty negative prompt is encoded rather than
    zeroed (``force_zeros_for_empty_prompt`` off), as the refiner repos declare.

    Args:
        unet_config ([`StableDiffusionXLRefinerUNetConfig`], *optional*): The denoiser.
        vae_config ([`StableDiffusionXLVAEConfig`], *optional*): The VAE.
        text_config (*optional*): Absent (``None``): the refiner has no CLIP ViT-L/14
            tower.
        text_config_2 ([`StableDiffusionXLTextConfig2`], *optional*): The OpenCLIP
            ViT-bigG/14 text encoder.
        requires_aesthetics_score (`bool`, *optional*, defaults to True):
            Condition on an aesthetic score (the fifth time id) instead of a target
            size.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to False):
            Encode the empty prompt for the unconditional branch.

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_xl import StableDiffusionXLRefinerConfig, StableDiffusionXLRefinerModel

    >>> config = StableDiffusionXLRefinerConfig()
    >>> model = StableDiffusionXLRefinerModel(config)  # random weights, refiner shapes
    >>> model.config.unet_config.block_out_channels
    (384, 768, 1536, 1536)
    ```"""

    model_type = "stable_diffusion_xl_refiner"

    sub_configs = {
        "unet_config": StableDiffusionXLRefinerUNetConfig,
        "vae_config": StableDiffusionXLVAEConfig,
        "text_config": StableDiffusionTextConfig,
        "text_config_2": StableDiffusionXLTextConfig2,
    }
    optional_sub_configs = ("text_config",)

    unet_config: StableDiffusionXLRefinerUNetConfig | dict | None = None
    text_config: StableDiffusionTextConfig | dict | None = None
    requires_aesthetics_score: bool = True
    force_zeros_for_empty_prompt: bool = False
