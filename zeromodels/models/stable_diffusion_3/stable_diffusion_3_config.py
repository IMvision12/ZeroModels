from zeromodels.base import BaseConfig
from zeromodels.models.stable_diffusion.stable_diffusion_config import (
    AutoencoderKLConfig,
)
from zeromodels.models.stable_diffusion_xl.stable_diffusion_xl_config import (
    StableDiffusionXLTextConfig2,
)


class StableDiffusion3TransformerConfig(BaseConfig):
    r"""Configuration for [`SD3Transformer2DModel`], the Stable Diffusion 3 MMDiT.

    The defaults match the SD 3 medium transformer (2B parameters). Fields mirror
    the model constructor and serialize flat; build a model from it with
    ``SD3Transformer2DModel(config)``.

    Args:
        sample_size (`int`, *optional*, defaults to 128):
            Latent spatial size the graph is built for (image size / 8).
        patch_size (`int`, *optional*, defaults to 2):
            Patch side; the token grid is ``sample_size / patch_size``.
        in_channels / out_channels (`int`, *optional*, defaults to 16):
            Latent channel counts in and out.
        num_layers (`int`, *optional*, defaults to 24):
            Joint transformer blocks.
        attention_head_dim (`int`, *optional*, defaults to 64):
            Width of one attention head.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Attention heads; the token width is ``num_attention_heads *
            attention_head_dim``.
        joint_attention_dim (`int`, *optional*, defaults to 4096):
            Width of the text features (the T5 width; the CLIP features are padded
            to it).
        caption_projection_dim (`int`, *optional*, defaults to 1536):
            Width the text features are projected to (the token width).
        pooled_projection_dim (`int`, *optional*, defaults to 2048):
            Width of the pooled CLIP embeddings added to the timestep embedding.
        pos_embed_max_size (`int`, *optional*, defaults to 192):
            Side of the precomputed position grid the latent grid is cropped from.
        qk_norm (`str`, *optional*):
            ``"rms_norm"`` normalizes q and k per head (SD 3.5); `None` for SD 3.
        dual_attention_layers (`tuple`, *optional*):
            Blocks with a second, latent-only attention (SD 3.5 medium: 0 to 12).
        text_seq_len (`int`, *optional*, defaults to 333):
            Static text sequence length (77 CLIP + 256 T5 tokens).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_3 import StableDiffusion3TransformerConfig, SD3Transformer2DModel

    >>> config = StableDiffusion3TransformerConfig()
    >>> model = SD3Transformer2DModel(config)
    >>> model.config.num_layers
    24
    ```"""

    model_type = "sd3_transformer_2d"

    sample_size: int = 128
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 24
    attention_head_dim: int = 64
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1536
    pooled_projection_dim: int = 2048
    pos_embed_max_size: int = 192
    qk_norm: str | None = None
    dual_attention_layers: tuple = ()
    text_seq_len: int = 333


class StableDiffusion3T5EncoderConfig(BaseConfig):
    r"""Configuration for [`SD3T5EncoderModel`], the third Stable Diffusion 3 text
    encoder: the encoder half of T5 v1.1 XXL.

    The defaults are the T5-XXL encoder every SD 3 / 3.5 checkpoint conditions on
    (24 layers, 4096 wide, 64 heads of 64, gated-GELU feed-forward of 10240, 4.7B
    parameters), hosted once as ``zeromodels/t5-v1_1-xxl-encoder``. Fields mirror
    the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 32128):
            SentencePiece vocabulary.
        embed_dim (`int`, *optional*, defaults to 4096):
            Token width (``d_model``).
        key_value_dim (`int`, *optional*, defaults to 64):
            Width of one attention head (``d_kv``).
        mlp_dim (`int`, *optional*, defaults to 10240):
            Feed-forward inner width (``d_ff``).
        num_layers (`int`, *optional*, defaults to 24):
            Encoder blocks.
        num_heads (`int`, *optional*, defaults to 64):
            Attention heads.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Relative-position-bias buckets.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            Largest bucketed distance.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            RMSNorm epsilon.
        pad_token_id / eos_token_id (`int`, *optional*, defaults to 0 / 1):
            The tokenizer's ``<pad>`` and ``</s>``.

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_3 import StableDiffusion3T5EncoderConfig, SD3T5EncoderModel

    >>> config = StableDiffusion3T5EncoderConfig(num_layers=2, embed_dim=64, mlp_dim=128, num_heads=4)
    >>> model = SD3T5EncoderModel(config)
    >>> model.config.num_layers
    2
    ```"""

    model_type = "stable_diffusion_3_t5_encoder"

    vocab_size: int = 32128
    embed_dim: int = 4096
    key_value_dim: int = 64
    mlp_dim: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_eps: float = 1e-06
    pad_token_id: int = 0
    eos_token_id: int = 1


class StableDiffusion3VAEConfig(AutoencoderKLConfig):
    r"""The Stable Diffusion 3 VAE: an [`AutoencoderKLConfig`] with 16 latent channels,
    no quant convolutions, the SD 3 latent scaling (1.5305) and shift (0.0609),
    built for 1024px in float32 (``force_upcast``)."""

    latent_channels: int = 16
    sample_size: int = 1024
    scaling_factor: float = 1.5305
    shift_factor: float = 0.0609
    force_upcast: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False


class StableDiffusion3TextConfig(StableDiffusionXLTextConfig2):
    r"""The first Stable Diffusion 3 text tower: CLIP ViT-L/14's text encoder with
    its 768-d projection (``CLIPTextModelWithProjection``), ``quick_gelu``."""

    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    projection_dim: int = 768
    hidden_act: str = "quick_gelu"


class StableDiffusion3Config(BaseConfig):
    r"""Configuration for [`StableDiffusion3Model`], the hosted SD 3 weights container.

    One config for the container's components, so a repo carries a single
    ``zm_config.json`` (nested ``transformer_config`` / ``vae_config`` /
    ``text_config`` / ``text_config_2``) plus the weights, and loads with the
    standard ``from_weights("zeromodels/<variant>")``. The constructor stays flat:
    sub-config fields are prefixed ``transformer_`` / ``vae_`` / ``text_`` /
    ``text_2_``. The third text encoder (T5-XXL) is not a component: it is a
    separately hosted :class:`SD3T5EncoderModel` (its own
    [`StableDiffusion3T5EncoderConfig`]) the task attaches on demand;
    ``max_sequence_length`` and the T5 token ids describe its inputs.

    Args:
        transformer_config ([`StableDiffusion3TransformerConfig`], *optional*): The
            MMDiT.
        vae_config ([`StableDiffusion3VAEConfig`], *optional*): The 16-channel VAE.
        text_config ([`StableDiffusion3TextConfig`], *optional*): The CLIP ViT-L/14
            text encoder (with projection).
        text_config_2 ([`StableDiffusionXLTextConfig2`], *optional*): The OpenCLIP
            ViT-bigG/14 text encoder (with projection).
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            First text tower activation (the second carries its own).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Text encoder LayerNorm epsilon.
        bos_token_id / eos_token_id / pad_token_id (`int`, *optional*):
            CLIP's ``<|startoftext|>`` (49406) and ``<|endoftext|>`` (49407, also the
            first tokenizer's pad).
        pad_token_id_2 (`int`, *optional*, defaults to 0):
            The second CLIP tokenizer pads with ``!`` (id 0).
        max_sequence_length (`int`, *optional*, defaults to 256):
            T5 token length.
        eos_token_id_3 / pad_token_id_3 (`int`, *optional*, defaults to 1 / 0):
            The T5 tokenizer's ``</s>`` and ``<pad>``.
        scheduler_config (`dict`, *optional*):
            The checkpoint's sampler in the diffusers ``scheduler_config.json`` form
            (``FlowMatchEulerDiscreteScheduler`` with its ``shift``).

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_3 import StableDiffusion3Config, StableDiffusion3Model

    >>> config = StableDiffusion3Config()
    >>> model = StableDiffusion3Model(config)  # random weights, SD 3 medium shapes
    >>> model.config.transformer_config.num_layers
    24
    ```"""

    model_type = "stable_diffusion_3"

    sub_configs = {
        "transformer_config": StableDiffusion3TransformerConfig,
        "vae_config": StableDiffusion3VAEConfig,
        "text_config": StableDiffusion3TextConfig,
        "text_config_2": StableDiffusionXLTextConfig2,
    }
    sub_config_prefixes = {
        "transformer_config": "transformer_",
        "vae_config": "vae_",
        "text_config": "text_",
        "text_config_2": "text_2_",
    }
    group_extras = {"text_config": ("vocab_size", "max_seq_len")}

    transformer_config: StableDiffusion3TransformerConfig | dict | None = None
    vae_config: StableDiffusion3VAEConfig | dict | None = None
    text_config: StableDiffusion3TextConfig | dict | None = None
    text_config_2: StableDiffusionXLTextConfig2 | dict | None = None
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-05
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    pad_token_id: int = 49407
    pad_token_id_2: int = 0
    max_sequence_length: int = 256
    eos_token_id_3: int = 1
    pad_token_id_3: int = 0
    scheduler_config: dict | None = None
