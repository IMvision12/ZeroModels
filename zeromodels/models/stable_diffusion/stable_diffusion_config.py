from zeromodels.base import BaseConfig
from zeromodels.models.clip.clip_config import CLIPTextConfig


class UNet2DConditionConfig(BaseConfig):
    r"""Configuration for [`UNet2DConditionModel`], the Stable Diffusion denoiser.

    The defaults match the Stable Diffusion 1.x UNet (860M parameters). Fields
    mirror the model constructor and serialize flat; build a model from it with
    ``UNet2DConditionModel(config)``. The same class serves later UNets that only
    change the widths (Stable Diffusion 2.x: 1024-d cross-attention, one head count
    per level, a linear token projection).

    Args:
        sample_size (`int`, *optional*, defaults to 64):
            Latent spatial size the graph is built for (image size / 8).
        in_channels / out_channels (`int`, *optional*, defaults to 4):
            Latent channel counts in and out.
        down_block_types / up_block_types (`tuple`):
            Per-level block kind, ``"CrossAttnDownBlock2D"`` / ``"DownBlock2D"`` and
            their up counterparts.
        block_out_channels (`tuple`, *optional*, defaults to (320, 640, 1280, 1280)):
            Channel width per level.
        layers_per_block (`int`, *optional*, defaults to 2):
            ResNet blocks per down level (up levels use one more).
        cross_attention_dim (`int`, *optional*, defaults to 768):
            Width of the text ``encoder_hidden_states``.
        num_attention_heads (`int` or `tuple`, *optional*, defaults to 8):
            Attention heads in the ``CrossAttn`` blocks, one value for every level
            or a tuple with one per level.
        norm_num_groups (`int`, *optional*, defaults to 32):
            GroupNorm group count.
        use_linear_projection (`bool`, *optional*, defaults to False):
            Project the Transformer2D tokens with a linear layer instead of a 1x1
            convolution on the feature map.
        transformer_layers_per_block (`int` or `tuple`, *optional*, defaults to 1):
            Transformer blocks stacked in each Transformer2D, one value for every
            level or one per level (SDXL: (1, 2, 10)).
        addition_embed_type (`str`, *optional*):
            `"text_time"` adds SDXL's micro-conditioning to the timestep embedding
            (the pooled text embedding plus the sinusoidally embedded size / crop
            `time_ids`, through the `add_embedding` MLP); `None` for SD 1.x / 2.x.
        addition_time_embed_dim (`int`, *optional*, defaults to 256):
            Sinusoidal embedding width of each time id.
        projection_class_embeddings_input_dim (`int`, *optional*):
            Input width of the `add_embedding` MLP: the pooled text width plus
            `num_time_ids * addition_time_embed_dim` (SDXL: 1280 + 6 * 256 = 2816).
        num_time_ids (`int`, *optional*, defaults to 6):
            Micro-conditioning values per image (original size, crop offset, target
            size).
        text_seq_len (`int`, *optional*, defaults to 77):
            Static text sequence length (CLIP pads to 77).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion import UNet2DConditionConfig, UNet2DConditionModel

    >>> config = UNet2DConditionConfig()
    >>> model = UNet2DConditionModel(config)
    >>> model.config.block_out_channels
    (320, 640, 1280, 1280)
    ```"""

    model_type = "unet_2d_condition"

    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: tuple = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: tuple = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    block_out_channels: tuple = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    cross_attention_dim: int = 768
    num_attention_heads: int | tuple = 8
    norm_num_groups: int = 32
    use_linear_projection: bool = False
    transformer_layers_per_block: int | tuple = 1
    addition_embed_type: str | None = None
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int | None = None
    num_time_ids: int = 6
    text_seq_len: int = 77


class AutoencoderKLConfig(BaseConfig):
    r"""Configuration for [`AutoencoderKL`], the Stable Diffusion VAE.

    The defaults match the Stable Diffusion 1.x VAE (84M parameters). Fields mirror
    the model constructor and serialize flat.

    Args:
        in_channels / out_channels (`int`, *optional*, defaults to 3):
            Image channel counts in and out.
        latent_channels (`int`, *optional*, defaults to 4):
            Latent channel count.
        block_out_channels (`tuple`, *optional*, defaults to (128, 256, 512, 512)):
            Channel width per level; the ``2 ** (len - 1)`` downsample factor is 8.
        layers_per_block (`int`, *optional*, defaults to 2):
            ResNet blocks per encoder level.
        norm_num_groups (`int`, *optional*, defaults to 32):
            GroupNorm group count.
        sample_size (`int`, *optional*, defaults to 512):
            Image resolution the encoder/decoder graphs are built for.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            Latent scaling applied by the pipeline around the VAE.
        force_upcast (`bool`, *optional*, defaults to False):
            Build the VAE in float32 whatever dtype the rest of the model loads in
            (the SDXL VAE overflows in float16).
        shift_factor (`float`, *optional*, defaults to 0.0):
            Latent offset applied with the scaling (SD3: ``(z - shift) * scale``).
        use_quant_conv / use_post_quant_conv (`bool`, *optional*, defaults to True):
            The 1x1 convolutions around the latent (absent in the SD3 VAE).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion import AutoencoderKLConfig, AutoencoderKL

    >>> config = AutoencoderKLConfig()
    >>> vae = AutoencoderKL.from_config(config)
    >>> vae.vae_scale_factor
    8
    ```"""

    model_type = "autoencoder_kl"

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    sample_size: int = 512
    scaling_factor: float = 0.18215
    force_upcast: bool = False
    shift_factor: float = 0.0
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True


class StableDiffusionTextConfig(CLIPTextConfig):
    r"""The Stable Diffusion text tower: CLIP ViT-L/14's text encoder.

    Same fields as [`CLIPTextConfig`], defaulted to the ViT-L/14 sizes SD 1.x
    conditions on (768 wide, 12 heads, 12 layers) rather than CLIP-B's."""

    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12


class StableDiffusionConfig(BaseConfig):
    r"""Configuration for [`StableDiffusionModel`], the hosted SD 1.x weights container.

    One config for the three trained components, so a repo carries a single
    ``zm_config.json`` (nested ``unet_config`` / ``vae_config`` / ``text_config``)
    plus one ``model.weights.h5``, and loads with the standard
    ``from_weights("zeromodels/<variant>")``. The constructor stays flat, like every
    other zeromodels model: sub-config fields are prefixed ``unet_`` / ``vae_`` /
    ``text_`` (``vocab_size`` and ``max_seq_len`` keep their names). The defaults
    are SD 1.x (1.1 through 1.5 share this architecture; only the weights differ).

    Args:
        unet_config ([`UNet2DConditionConfig`], *optional*): The denoiser.
        vae_config ([`AutoencoderKLConfig`], *optional*): The VAE.
        text_config ([`StableDiffusionTextConfig`], *optional*): The CLIP ViT-L/14
            text encoder.
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            Text encoder activation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Text encoder LayerNorm epsilon.
        bos_token_id / eos_token_id / pad_token_id (`int`, *optional*):
            CLIP's ``<|startoftext|>`` (49406) and ``<|endoftext|>`` (49407, also the
            pad); ``generate`` builds the empty prompt for classifier-free guidance
            from them.
        scheduler_config (`dict`, *optional*):
            The checkpoint's default sampler and its training noise schedule, in the
            diffusers ``scheduler_config.json`` form (``_class_name``,
            ``beta_schedule``, ``beta_start`` / ``beta_end``, ``steps_offset``, ...).
            Stored inside ``zm_config.json`` rather than as a separate file;
            ``StableDiffusionTextToImage`` builds its scheduler from it (PNDM with
            the SD 1.x schedule when absent).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion import StableDiffusionConfig, StableDiffusionModel

    >>> config = StableDiffusionConfig()
    >>> model = StableDiffusionModel(config)  # random weights, SD 1.x shapes
    >>> model.config.unet_config.block_out_channels
    (320, 640, 1280, 1280)
    ```"""

    model_type = "stable_diffusion"

    sub_configs = {
        "unet_config": UNet2DConditionConfig,
        "vae_config": AutoencoderKLConfig,
        "text_config": StableDiffusionTextConfig,
    }
    sub_config_prefixes = {
        "unet_config": "unet_",
        "vae_config": "vae_",
        "text_config": "text_",
    }
    group_extras = {"text_config": ("vocab_size", "max_seq_len")}

    unet_config: UNet2DConditionConfig | dict | None = None
    vae_config: AutoencoderKLConfig | dict | None = None
    text_config: StableDiffusionTextConfig | dict | None = None
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-05
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    pad_token_id: int = 49407
    scheduler_config: dict | None = None
