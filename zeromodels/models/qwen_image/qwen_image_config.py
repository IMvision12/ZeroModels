from zeromodels.base import BaseConfig
from zeromodels.models.qwen2_5_vl.qwen2_5_vl_config import Qwen2_5VLTextConfig

DEFAULT_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
DEFAULT_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


class QwenImageVAEConfig(BaseConfig):
    """Configuration for :class:`AutoencoderKLQwenImage`.

    Fields match Diffusers ``AutoencoderKLQwenImage``; ``sample_size`` is the
    ZeroModels graph-build resolution (weights are resolution-independent).
    """

    model_type = "autoencoder_kl_qwen_image"

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple = ()
    temperal_downsample: tuple = (False, True, True)
    dropout: float = 0.0
    input_channels: int = 3
    latents_mean: tuple = DEFAULT_LATENTS_MEAN
    latents_std: tuple = DEFAULT_LATENTS_STD
    sample_size: int = 1024


class QwenImageTransformerConfig(BaseConfig):
    r"""Configuration for [`QwenImageTransformer2DModel`].

    Defaults match ``Qwen/Qwen-Image`` ``transformer/config.json`` (60-layer
    double-stream DiT, 24 heads × 128, joint text width 3584).

    Args:
        patch_size: Latent patch side (2); packed tokens use ``in_channels``.
        in_channels: Packed latent width (64 = 16 × 2 × 2).
        out_channels: Unpacked latent channels (16).
        num_layers: Dual-stream DiT blocks (60).
        attention_head_dim / num_attention_heads: Head geometry (inner dim =
            heads × head_dim = 3072).
        joint_attention_dim: Text feature width from Qwen2.5-VL (3584).
        axes_dims_rope: MS-RoPE axis splits ``(16, 56, 56)``.
        sample_size: Latent spatial side the graph is built for (image / 8).
        text_seq_len: Static text sequence length for the graph.
    """

    model_type = "qwen_image_transformer_2d"

    patch_size: int = 2
    in_channels: int = 64
    out_channels: int = 16
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    axes_dims_rope: tuple = (16, 56, 56)
    guidance_embeds: bool = False
    sample_size: int = 128
    text_seq_len: int = 512


class QwenImageTextConfig(Qwen2_5VLTextConfig):
    r"""Qwen2.5-VL text tower used as the Qwen-Image prompt encoder.

    Defaults match ``Qwen/Qwen-Image`` ``text_encoder/config.json`` (7B Instruct
    text half: 28 layers, 3584-d, 28 heads / 4 KV).
    """

    model_type = "qwen_image_text"

    vocab_size: int = 152064
    embed_dim: int = 3584
    mlp_dim: int = 18944
    num_layers: int = 28
    num_heads: int = 28
    num_kv_heads: int = 4
    norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    mrope_section: tuple = (16, 24, 24)
    tie_embeddings: bool = False
    max_seq_len: int = 1024


class QwenImageConfig(BaseConfig):
    r"""Configuration for [`QwenImageModel`] / [`QwenImageTextToImage`].

    One hosted container: MMDiT transformer + Qwen-Image VAE + Qwen2.5-VL text
    encoder. Nested serialize (``transformer_config`` / ``vae_config`` /
    ``text_config``); flat constructor with ``transformer_`` / ``vae_`` /
    ``text_`` prefixes.

    Args:
        transformer_config: The double-stream DiT.
        vae_config: The Wan-derived 16-channel VAE.
        text_config: The Qwen2.5-VL text tower.
        scheduler_config: Diffusers ``scheduler_config.json`` dict
            (``FlowMatchEulerDiscreteScheduler`` with dynamic shifting).
        prompt_template_encode_start_idx: Tokens dropped from the ChatML
            template prefix when building prompt embeds (34 in Diffusers).
        max_sequence_length: Prompt embed length after the template drop (512).
        default_sample_size: Latent grid side used when height/width omitted
            (128 → 1024px at VAE scale 8).
        bos_token_id / eos_token_id / pad_token_id: Qwen2 special token ids.
    """

    model_type = "qwen_image"

    sub_configs = {
        "transformer_config": QwenImageTransformerConfig,
        "vae_config": QwenImageVAEConfig,
        "text_config": QwenImageTextConfig,
    }
    sub_config_prefixes = {
        "transformer_config": "transformer_",
        "vae_config": "vae_",
        "text_config": "text_",
    }
    group_extras = {"text_config": ("vocab_size", "max_seq_len")}

    transformer_config: QwenImageTransformerConfig | dict | None = None
    vae_config: QwenImageVAEConfig | dict | None = None
    text_config: QwenImageTextConfig | dict | None = None
    scheduler_config: dict | None = None
    prompt_template_encode_start_idx: int = 34
    max_sequence_length: int = 512
    default_sample_size: int = 128
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
