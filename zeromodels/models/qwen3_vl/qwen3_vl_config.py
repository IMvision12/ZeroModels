from zeromodels.base import BaseConfig


class Qwen3VLTextConfig(BaseConfig):
    r"""Text-decoder config for Qwen3-VL (the ``text_config`` sub-config).

    A Qwen3 decoder (GQA with per-head QK-norm, bias-free QKV, SwiGLU, RMSNorm) with
    interleaved multimodal rotary positions (M-RoPE): the rotary dims split across
    temporal / height / width by ``mrope_section``.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 6144):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 28):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 8):
            Key/value heads per layer (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head attention dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon (shared by the per-head QK-norms too).
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            Rotary base frequency.
        mrope_section (`tuple`, *optional*, defaults to `(24, 20, 20)`):
            Per-axis (temporal, height, width) split of the interleaved M-RoPE.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding."""

    model_type = "qwen3_vl_text"

    vocab_size: int = 151936
    embed_dim: int = 2048
    mlp_dim: int = 6144
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple = (24, 20, 20)
    tie_embeddings: bool = True


class Qwen3VLVisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen3-VL (the ``vision_config`` sub-config).

    The Qwen3-VL ViT: full attention over the packed patch sequence, learned
    (bilinearly interpolated) position embeddings, GELU MLP blocks, a 2x2
    spatial-merge projector to the text ``out_dim``, and DeepStack (features from
    ``deepstack_visual_indexes`` blocks are injected into the text decoder's early
    layers).

    Args:
        depth (`int`, *optional*, defaults to 24):
            Number of ViT blocks.
        embed_dim (`int`, *optional*, defaults to 1024):
            ViT tower width.
        mlp_dim (`int`, *optional*, defaults to 4096):
            ViT MLP hidden width.
        num_heads (`int`, *optional*, defaults to 16):
            ViT attention heads.
        out_dim (`int`, *optional*, defaults to 2048):
            Output width of the vision merger (the text ``embed_dim``).
        act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            ViT MLP activation.
        num_position_embeddings (`int`, *optional*, defaults to 2304):
            Size of the learned position-embedding grid (interpolated per image).
        deepstack_visual_indexes (`tuple`, *optional*, defaults to `(5, 11, 17)`):
            ViT block indices whose features are injected into the text decoder.
        patch_size (`int`, *optional*, defaults to 16):
            Vision patch side length.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the spatial patch merge.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames merged per temporal patch.
        in_channels (`int`, *optional*, defaults to 3):
            Input image channels."""

    model_type = "qwen3_vl_vision"

    depth: int = 24
    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 16
    out_dim: int = 2048
    act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (5, 11, 17)
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen3VLConfig(BaseConfig):
    r"""Configuration for Qwen3-VL: [`Qwen3VLModel`] and [`Qwen3VLConditionalGenerate`].

    A composite config: the Qwen3 text decoder lives in a [`Qwen3VLTextConfig`]
    (``text_config``) and the ViT in a [`Qwen3VLVisionConfig`] (``vision_config``);
    the four vision token ids are the top-level image/video glue. The flat model
    constructor is fed by flattening the sub-configs: vision fields gain the
    ``vision_`` prefix, except the geometry / DeepStack fields (`patch_size`,
    `spatial_merge_size`, `temporal_patch_size`, `in_channels`,
    `num_position_embeddings`, `deepstack_visual_indexes`) that keep their own name.

    Args:
        text_config (`Qwen3VLTextConfig | dict`, *optional*):
            Text-decoder config.
        vision_config (`Qwen3VLVisionConfig | dict`, *optional*):
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
    >>> from zeromodels.models.qwen3_vl import Qwen3VLConfig, Qwen3VLConditionalGenerate

    >>> configuration = Qwen3VLConfig(
    ...     text_config={"embed_dim": 2560, "num_layers": 36},
    ...     vision_config={"depth": 24, "out_dim": 2560},
    ... )
    >>> model = Qwen3VLConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl"

    sub_configs = {
        "text_config": Qwen3VLTextConfig,
        "vision_config": Qwen3VLVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "num_position_embeddings",
            "deepstack_visual_indexes",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
        )
    }

    text_config: Qwen3VLTextConfig | dict | None = None
    vision_config: Qwen3VLVisionConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653


QWEN3_VL_TOKENS = {
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
