from zeromodels.base import BaseConfig


class Qwen3VLMoeTextConfig(BaseConfig):
    r"""Text-decoder config for Qwen3-VL-MoE (the ``text_config`` sub-config).

    A Qwen3-VL text decoder (GQA with per-head QK-norm, bias-free QKV, interleaved
    M-RoPE) whose MLP is a sparse Mixture-of-Experts: a float32-softmax top-k router
    over ``num_experts`` fused SwiGLU experts (**no shared expert**). A dense SwiGLU
    MLP is used instead on any layer in ``mlp_only_layers`` or off the
    ``decoder_sparse_step`` cadence.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 5632):
            Dense-MLP hidden width (used on non-MoE layers).
        num_layers (`int`, *optional*, defaults to 24):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 16):
            Key/value heads per layer (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            Rotary base frequency.
        mrope_section (`tuple`, *optional*, defaults to `(24, 20, 20)`):
            Per-axis (temporal, height, width) split of the interleaved M-RoPE.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding.
        num_experts (`int`, *optional*, defaults to 60):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Experts activated per token.
        moe_mlp_dim (`int`, *optional*, defaults to 1408):
            Per-routed-expert hidden width (``moe_intermediate_size``).
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Renormalize the selected router weights to sum to one.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Use MoE every Nth layer (1 = every layer).
        mlp_only_layers (`tuple`, *optional*, defaults to `()`):
            Layer indices forced to a dense MLP instead of MoE."""

    model_type = "qwen3_vl_moe_text"

    vocab_size: int = 151936
    embed_dim: int = 2048
    mlp_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple = (24, 20, 20)
    tie_embeddings: bool = True
    num_experts: int = 60
    num_experts_per_tok: int = 4
    moe_mlp_dim: int = 1408
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()


class Qwen3VLMoeVisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen3-VL-MoE (the ``vision_config`` sub-config).

    Identical to the Qwen3-VL ViT: full attention over the packed patch sequence,
    learned (bilinearly interpolated) position embeddings, GELU blocks, a 2x2
    spatial-merge projector to the text ``out_dim``, and DeepStack (features from
    ``deepstack_visual_indexes`` blocks are injected into the text decoder's early
    layers).

    Args:
        depth (`int`, *optional*, defaults to 27):
            Number of ViT blocks.
        embed_dim (`int`, *optional*, defaults to 1152):
            ViT tower width.
        mlp_dim (`int`, *optional*, defaults to 4304):
            ViT MLP hidden width.
        num_heads (`int`, *optional*, defaults to 16):
            ViT attention heads.
        out_dim (`int`, *optional*, defaults to 3584):
            Output width of the vision merger (the text ``embed_dim``).
        act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            ViT MLP activation.
        num_position_embeddings (`int`, *optional*, defaults to 2304):
            Size of the learned position-embedding grid (interpolated per image).
        deepstack_visual_indexes (`tuple`, *optional*, defaults to `(8, 16, 24)`):
            ViT block indices whose features are injected into the text decoder.
        patch_size (`int`, *optional*, defaults to 16):
            Vision patch side length.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the spatial patch merge.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames merged per temporal patch.
        in_channels (`int`, *optional*, defaults to 3):
            Input image channels."""

    model_type = "qwen3_vl_moe_vision"

    depth: int = 27
    embed_dim: int = 1152
    mlp_dim: int = 4304
    num_heads: int = 16
    out_dim: int = 3584
    act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (8, 16, 24)
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen3VLMoeConfig(BaseConfig):
    r"""Configuration for Qwen3-VL-MoE: [`Qwen3VLMoeModel`] and [`Qwen3VLMoeConditionalGenerate`].

    A composite config: the Qwen3-VL MoE text decoder lives in a
    [`Qwen3VLMoeTextConfig`] (``text_config``) and the ViT in a
    [`Qwen3VLMoeVisionConfig`] (``vision_config``); the four vision token ids are the
    top-level image/video glue. The flat model constructor is fed by flattening the
    sub-configs: vision fields gain the ``vision_`` prefix, except the geometry /
    DeepStack fields (`num_position_embeddings`, `deepstack_visual_indexes`,
    `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `in_channels`) which
    keep their own name.

    Args:
        text_config (`Qwen3VLMoeTextConfig | dict`, *optional*):
            Text-decoder config.
        vision_config (`Qwen3VLMoeVisionConfig | dict`, *optional*):
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
    >>> from zeromodels.models.qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeConditionalGenerate

    >>> configuration = Qwen3VLMoeConfig(
    ...     text_config={"embed_dim": 2048, "num_layers": 48, "num_experts": 128},
    ...     vision_config={"depth": 27, "out_dim": 2048},
    ... )
    >>> model = Qwen3VLMoeConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_moe"

    sub_configs = {
        "text_config": Qwen3VLMoeTextConfig,
        "vision_config": Qwen3VLMoeVisionConfig,
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

    text_config: Qwen3VLMoeTextConfig | dict | None = None
    vision_config: Qwen3VLMoeVisionConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653


QWEN3_VL_MOE_TOKENS = {
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
