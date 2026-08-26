from zeromodels.base import BaseConfig


class Qwen3_5TextConfig(BaseConfig):
    r"""Text-decoder config for Qwen3.5 (the ``text_config`` sub-config).

    The dense Qwen3.5 hybrid decoder: mostly Gated-DeltaNet linear-attention layers with a
    gated full-attention block every ``full_attention_interval`` (GQA, per-head QK-norm,
    partial-rotary interleaved M-RoPE), each with a dense GeGLU MLP. Also the config for the
    standalone text head [`Qwen3_5TextGenerate`] (which uses 1-D rope and ignores
    ``mrope_section``)."""

    model_type = "qwen3_5_text"

    vocab_size: int = 248320
    embed_dim: int = 5120
    mlp_dim: int = 17408
    num_layers: int = 64
    num_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    tie_embeddings: bool = False
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48


class Qwen3_5VisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen3.5 (the ``vision_config`` sub-config).

    The Qwen3-VL ViT (no DeepStack): full attention over the packed patch sequence, learned
    (bilinearly interpolated) position embeddings, GELU MLP blocks, and a 2x2 spatial-merge
    projector to the text ``out_dim``."""

    model_type = "qwen3_5_vision"

    depth: int = 27
    embed_dim: int = 1152
    mlp_dim: int = 4304
    num_heads: int = 16
    out_dim: int = 5120
    act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen3_5Config(BaseConfig):
    r"""Configuration for the dense Qwen3.5 VLM: [`Qwen3_5VLModel`] and
    [`Qwen3_5ConditionalGenerate`].

    A composite config: the dense hybrid text decoder lives in a [`Qwen3_5TextConfig`]
    (``text_config``) and the ViT in a [`Qwen3_5VisionConfig`] (``vision_config``); the four
    vision token ids are the top-level image/video glue. Flattened to the model constructor
    with the ``vision_`` prefix on the vision fields, except the geometry fields which keep
    their own name."""

    model_type = "qwen3_5"

    sub_configs = {
        "text_config": Qwen3_5TextConfig,
        "vision_config": Qwen3_5VisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "num_position_embeddings",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
        )
    }

    text_config: Qwen3_5TextConfig | dict | None = None
    vision_config: Qwen3_5VisionConfig | dict | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
