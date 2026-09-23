from zeromodels.base import BaseConfig
from zeromodels.models.qwen3_vl.qwen3_vl_config import Qwen3VLTextConfig

DEFAULT_LATENTS_MEAN = (
    0.5126,
    0.7721,
    -0.0631,
    1.3506,
    -0.7855,
    -2.1025,
    -0.3458,
    1.3722,
    1.8873,
    -1.7177,
    -0.6510,
    0.2732,
    0.7562,
    -0.6163,
    -1.0277,
    3.8363,
    2.0210,
    0.0472,
    0.9320,
    2.0087,
    2.4954,
    -0.1391,
    -1.4249,
    1.8464,
    -0.5236,
    1.2826,
    3.7046,
    -1.3035,
    2.7286,
    -1.4518,
    -1.9036,
    -1.9955,
    -0.0342,
    -1.0265,
    -0.7636,
    3.0555,
    0.0746,
    -3.0751,
    -0.1076,
    1.7376,
    -1.0914,
    -1.9435,
    -0.2784,
    -1.3680,
    0.4809,
    -0.4433,
    0.3764,
    0.5729,
    -2.0595,
    1.0960,
    -1.3260,
    -2.0211,
    -5.0179,
    0.5275,
    4.0162,
    1.8505,
    0.3026,
    1.9373,
    1.4937,
    0.2632,
    0.5547,
    -1.7121,
    -0.1562,
    0.0304,
)
DEFAULT_LATENTS_STD = (
    3.2001,
    3.2936,
    3.4321,
    3.0091,
    3.1061,
    4.0379,
    4.0705,
    3.7910,
    3.0785,
    3.6500,
    3.9308,
    3.0904,
    2.8778,
    3.7675,
    3.7320,
    5.0756,
    3.2864,
    4.0397,
    3.1317,
    4.0443,
    2.9249,
    3.9454,
    3.0988,
    4.2489,
    3.4896,
    3.8513,
    3.9323,
    3.4719,
    3.7498,
    4.2830,
    3.5694,
    4.2467,
    3.9037,
    3.2947,
    5.0770,
    3.5075,
    3.2700,
    3.4767,
    2.8063,
    5.1125,
    3.5327,
    4.7833,
    3.1286,
    4.1819,
    3.8527,
    3.8312,
    3.5605,
    4.3875,
    3.9624,
    4.0168,
    3.5643,
    4.0550,
    5.5614,
    4.2963,
    4.4080,
    3.4959,
    3.8747,
    3.7608,
    3.5735,
    3.1490,
    3.7662,
    3.6746,
    3.4563,
    3.8161,
)


class QwenImage21VAEConfig(BaseConfig):
    """Configuration for :class:`AutoencoderKLQwenImage21`.

    Defaults match ``Qwen/Qwen-Image-2.1`` ``vae/config.json`` (64-channel latent,
    16× spatial compression, residual encoder/decoder, 4-channel RGBA IO).
    """

    model_type = "autoencoder_kl_qwen_image_21"

    base_dim: int = 96
    decoder_base_dim: int = 144
    z_dim: int = 64
    dim_mult: tuple = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    attn_scales: tuple = ()
    temperal_downsample: tuple = (False, True, True, True)
    dropout: float = 0.0
    input_channels: int = 4
    out_channels: int = 4
    is_residual: bool = True
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 8
    latents_mean: tuple = DEFAULT_LATENTS_MEAN
    latents_std: tuple = DEFAULT_LATENTS_STD
    sample_size: int = 1024


class QwenImage21TransformerConfig(BaseConfig):
    """Configuration for :class:`QwenImage21Transformer2DModel`.

    Defaults match ``Qwen/Qwen-Image-2.1`` ``transformer/config.json`` (32-layer
    single-stream DiT, 32 heads × 128, context width 4096, unpatched latents).
    """

    model_type = "qwen_image_21_transformer_2d"

    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True
    sample_size: int = 64
    text_seq_len: int = 512


class QwenImage21TextConfig(Qwen3VLTextConfig):
    """Qwen3-VL text tower used as the Qwen-Image-2.1 prompt encoder.

    Defaults match ``Qwen/Qwen-Image-2.1`` ``text_encoder/config.json`` text half
    (36 layers, 4096-d, 32 heads / 8 KV).
    """

    model_type = "qwen_image_21_text"

    vocab_size: int = 151936
    embed_dim: int = 4096
    mlp_dim: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple = (24, 20, 20)
    tie_embeddings: bool = False
    max_seq_len: int = 1024


class QwenImage21Config(BaseConfig):
    """Configuration for :class:`QwenImage21Model` / :class:`QwenImage21TextToImage`.

    One hosted container: single-stream DiT + Qwen-Image-2.1 VAE + Qwen3-VL text
    encoder. Nested serialize; flat constructor with ``transformer_`` / ``vae_`` /
    ``text_`` prefixes.
    """

    model_type = "qwen_image_21"

    sub_configs = {
        "transformer_config": QwenImage21TransformerConfig,
        "vae_config": QwenImage21VAEConfig,
        "text_config": QwenImage21TextConfig,
    }
    sub_config_prefixes = {
        "transformer_config": "transformer_",
        "vae_config": "vae_",
        "text_config": "text_",
    }
    group_extras = {"text_config": ("vocab_size", "max_seq_len")}

    transformer_config: QwenImage21TransformerConfig | dict | None = None
    vae_config: QwenImage21VAEConfig | dict | None = None
    text_config: QwenImage21TextConfig | dict | None = None
    scheduler_config: dict | None = None
    prompt_template_encode_start_idx: int = 14
    max_sequence_length: int = 512
    default_sample_size: int = 64
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    image_token_id: int = 151655
