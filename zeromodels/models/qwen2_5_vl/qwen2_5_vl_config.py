from zeromodels.base import BaseConfig


class Qwen2_5VLTextConfig(BaseConfig):
    r"""Text-decoder config for Qwen2.5-VL (the ``text_config`` sub-config).

    A Qwen2 decoder (GQA with biased QKV, SwiGLU, RMSNorm) with multimodal rotary
    positions (M-RoPE): the rotary dims split across temporal / height / width by
    ``mrope_section``.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 11008):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 36):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Key/value heads per layer (GQA).
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        mrope_section (`tuple`, *optional*, defaults to `(16, 24, 24)`):
            Per-axis (temporal, height, width) split of the rotary dimensions.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding."""

    model_type = "qwen2_5_vl_text"

    vocab_size: int = 151936
    embed_dim: int = 2048
    mlp_dim: int = 11008
    num_layers: int = 36
    num_heads: int = 16
    num_kv_heads: int = 2
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    mrope_section: tuple = (16, 24, 24)
    tie_embeddings: bool = True


class Qwen2_5VLVisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen2.5-VL (the ``vision_config`` sub-config).

    The Qwen2.5-VL ViT: windowed attention with periodic full-attention blocks, a
    patch/temporal-patch embed, and a spatial-merge projector to ``out_dim``.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of ViT blocks.
        embed_dim (`int`, *optional*, defaults to 1280):
            ViT tower width.
        mlp_dim (`int`, *optional*, defaults to 3420):
            ViT MLP width.
        num_heads (`int`, *optional*, defaults to 16):
            ViT attention heads.
        out_dim (`int`, *optional*, defaults to 2048):
            Merger output width (matches the text `embed_dim`).
        window_size (`int`, *optional*, defaults to 112):
            Local-attention window (in pixels).
        fullatt_block_indexes (`tuple`, *optional*, defaults to `(7, 15, 23, 31)`):
            Block indices that use full (not windowed) attention.
        tokens_per_second (`int`, *optional*, defaults to 2):
            Video temporal rate used by the M-RoPE position ids.
        patch_size (`int`, *optional*, defaults to 14):
            Vision patch side length.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the spatial patch merge.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames merged per temporal patch.
        in_channels (`int`, *optional*, defaults to 3):
            Input image channels."""

    model_type = "qwen2_5_vl_vision"

    depth: int = 32
    embed_dim: int = 1280
    mlp_dim: int = 3420
    num_heads: int = 16
    out_dim: int = 2048
    window_size: int = 112
    fullatt_block_indexes: tuple = (7, 15, 23, 31)
    tokens_per_second: int = 2
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen2_5VLConfig(BaseConfig):
    r"""Configuration for Qwen2.5-VL: [`Qwen2_5VLModel`] and [`Qwen2_5VLConditionalGenerate`].

    A composite config: the text decoder lives in a [`Qwen2_5VLTextConfig`]
    (``text_config``) and the ViT in a [`Qwen2_5VLVisionConfig`] (``vision_config``);
    the four vision token ids are the top-level image/video glue. The flat model
    constructor is fed by flattening the sub-configs: vision fields gain the
    ``vision_`` prefix, except the geometry / video fields (`window_size`,
    `patch_size`, ...) that keep their own name.

    Args:
        text_config (`Qwen2_5VLTextConfig | dict`, *optional*):
            Text-decoder config.
        vision_config (`Qwen2_5VLVisionConfig | dict`, *optional*):
            Vision-tower config.
        image_token_id (`int`, *optional*, defaults to 151655):
            Placeholder token id replaced by image patch embeddings.
        video_token_id (`int`, *optional*, defaults to 151656):
            Placeholder token id replaced by video patch embeddings.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            `<|vision_start|>` marker id.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            `<|vision_end|>` marker id.

    Examples:

    ```python
    >>> from zeromodels.models.qwen2_5_vl import Qwen2_5VLConfig, Qwen2_5VLConditionalGenerate

    >>> configuration = Qwen2_5VLConfig(
    ...     text_config={"embed_dim": 3584, "num_layers": 28},
    ...     vision_config={"depth": 32},
    ... )
    >>> model = Qwen2_5VLConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_vl"

    sub_configs = {
        "text_config": Qwen2_5VLTextConfig,
        "vision_config": Qwen2_5VLVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "window_size",
            "fullatt_block_indexes",
            "tokens_per_second",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
        )
    }

    text_config: Qwen2_5VLTextConfig | dict | None = None
    vision_config: Qwen2_5VLVisionConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
