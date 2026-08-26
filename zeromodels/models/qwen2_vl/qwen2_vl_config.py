from zeromodels.base import BaseConfig


class Qwen2VLTextConfig(BaseConfig):
    r"""Text-decoder config for Qwen2-VL (the ``text_config`` sub-config).

    A Qwen2 decoder (GQA with biased QKV, SwiGLU, RMSNorm) with multimodal rotary
    positions (M-RoPE): the rotary dims split across temporal / height / width by
    ``mrope_section``.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 1536):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 8960):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 28):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 12):
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

    model_type = "qwen2_vl_text"

    vocab_size: int = 151936
    embed_dim: int = 1536
    mlp_dim: int = 8960
    num_layers: int = 28
    num_heads: int = 12
    num_kv_heads: int = 2
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    mrope_section: tuple = (16, 24, 24)
    tie_embeddings: bool = True


class Qwen2VLVisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen2-VL (the ``vision_config`` sub-config).

    The Qwen2-VL ViT: full attention over the packed patch sequence (no windowing),
    a patch/temporal-patch embed, and a 2x2 spatial-merge projector to the text
    ``embed_dim``. Unlike Qwen2.5-VL the MLP width is given as a ratio of the tower
    width (``mlp_ratio``) rather than an explicit hidden size.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of ViT blocks.
        embed_dim (`int`, *optional*, defaults to 1280):
            ViT tower width.
        num_heads (`int`, *optional*, defaults to 16):
            ViT attention heads.
        mlp_ratio (`int`, *optional*, defaults to 4):
            ViT MLP width as a multiple of ``embed_dim``.
        patch_size (`int`, *optional*, defaults to 14):
            Vision patch side length.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the spatial patch merge.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames merged per temporal patch.
        in_channels (`int`, *optional*, defaults to 3):
            Input image channels."""

    model_type = "qwen2_vl_vision"

    depth: int = 32
    embed_dim: int = 1280
    num_heads: int = 16
    mlp_ratio: int = 4
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen2VLConfig(BaseConfig):
    r"""Configuration for Qwen2-VL: [`Qwen2VLModel`] and [`Qwen2VLConditionalGenerate`].

    A composite config: the text decoder lives in a [`Qwen2VLTextConfig`]
    (``text_config``) and the ViT in a [`Qwen2VLVisionConfig`] (``vision_config``);
    the four vision token ids are the top-level image/video glue. The flat model
    constructor is fed by flattening the sub-configs: vision fields gain the
    ``vision_`` prefix, except the geometry fields (`patch_size`,
    `spatial_merge_size`, `temporal_patch_size`, `in_channels`) that keep their own
    name.

    Args:
        text_config (`Qwen2VLTextConfig | dict`, *optional*):
            Text-decoder config.
        vision_config (`Qwen2VLVisionConfig | dict`, *optional*):
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
    >>> from zeromodels.models.qwen2_vl import Qwen2VLConfig, Qwen2VLConditionalGenerate

    >>> configuration = Qwen2VLConfig(
    ...     text_config={"embed_dim": 3584, "num_layers": 28},
    ...     vision_config={"depth": 32},
    ... )
    >>> model = Qwen2VLConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_vl"

    sub_configs = {
        "text_config": Qwen2VLTextConfig,
        "vision_config": Qwen2VLVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
        )
    }

    text_config: Qwen2VLTextConfig | dict | None = None
    vision_config: Qwen2VLVisionConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
