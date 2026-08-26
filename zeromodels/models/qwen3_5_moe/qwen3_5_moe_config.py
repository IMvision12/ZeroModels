from zeromodels.base import BaseConfig


class Qwen3_5MoeTextConfig(BaseConfig):
    r"""Text-decoder config for Qwen3.5-MoE (the ``text_config`` sub-config).

    The Qwen3-Next hybrid decoder used as a sparse Mixture-of-Experts language model:
    most blocks are Gated-DeltaNet linear-attention layers, with a gated full-attention
    block every ``full_attention_interval`` (GQA, per-head QK-norm, partial-rotary
    interleaved M-RoPE). Every block's MLP is a softmax-routed expert bank plus a
    sigmoid-gated shared expert; RMSNorm is zero-centered.

    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 512):
            Dense-MLP width (unused: every layer is MoE here, kept for the shared
            decoder-layer constructor).
        num_layers (`int`, *optional*, defaults to 40):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads in the full-attention layers.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Key/value heads in the full-attention layers (GQA).
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dim of the full-attention layers.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon (shared everywhere, incl. the QK-norms).
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            Rotary base frequency.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Fraction of ``head_dim`` that receives rotary
            (``rotary_dim = int(head_dim * partial_rotary_factor)``).
        mrope_section (`tuple`, *optional*, defaults to `(11, 11, 10)`):
            Per-axis (temporal, height, width) split of the interleaved M-RoPE (sums
            to ``rotary_dim // 2``).
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the LM head is tied to the token embedding.
        full_attention_interval (`int`, *optional*, defaults to 4):
            A full-attention block is placed every Nth layer; the rest are
            Gated-DeltaNet linear-attention layers.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Causal conv1d kernel width in the linear-attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Per-head key dim of the linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Per-head value dim of the linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Key head count of the linear attention.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Value head count of the linear attention.
        num_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Experts activated per token.
        moe_mlp_dim (`int`, *optional*, defaults to 512):
            Per-routed-expert hidden width (``moe_intermediate_size``).
        shared_mlp_dim (`int`, *optional*, defaults to 512):
            Shared-expert hidden width.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Renormalize the selected router weights to sum to one.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Use MoE every Nth layer (1 = every layer).
        mlp_only_layers (`tuple`, *optional*, defaults to `()`):
            Layer indices forced to a dense MLP instead of MoE."""

    model_type = "qwen3_5_moe_text"

    vocab_size: int = 248320
    embed_dim: int = 2048
    mlp_dim: int = 512
    num_layers: int = 40
    num_heads: int = 16
    num_kv_heads: int = 2
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
    linear_num_value_heads: int = 32
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_mlp_dim: int = 512
    shared_mlp_dim: int = 512
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()


class Qwen3_5MoeVisionConfig(BaseConfig):
    r"""Vision-tower config for Qwen3.5-MoE (the ``vision_config`` sub-config).

    The Qwen3-VL ViT (no DeepStack): full attention over the packed patch sequence,
    learned (bilinearly interpolated) position embeddings, GELU MLP blocks, and a 2x2
    spatial-merge projector to the text ``out_dim``.

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
        patch_size (`int`, *optional*, defaults to 16):
            Vision patch side length.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Side length of the spatial patch merge.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames merged per temporal patch.
        in_channels (`int`, *optional*, defaults to 3):
            Input image channels."""

    model_type = "qwen3_5_moe_vision"

    depth: int = 27
    embed_dim: int = 1152
    mlp_dim: int = 4304
    num_heads: int = 16
    out_dim: int = 3584
    act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen3_5MoeConfig(BaseConfig):
    r"""Configuration for Qwen3.5-MoE: [`Qwen3_5MoeModel`] and [`Qwen3_5MoeConditionalGenerate`].

    A composite config: the Qwen3-Next MoE text decoder lives in a
    [`Qwen3_5MoeTextConfig`] (``text_config``) and the ViT in a
    [`Qwen3_5MoeVisionConfig`] (``vision_config``); the four vision token ids are the
    top-level image/video glue. The flat model constructor is fed by flattening the
    sub-configs: vision fields gain the ``vision_`` prefix, except the geometry fields
    (`num_position_embeddings`, `patch_size`, `spatial_merge_size`,
    `temporal_patch_size`, `in_channels`) which keep their own name.

    Args:
        text_config (`Qwen3_5MoeTextConfig | dict`, *optional*):
            Text-decoder config.
        vision_config (`Qwen3_5MoeVisionConfig | dict`, *optional*):
            Vision-tower config.
        image_token_id (`int`, *optional*, defaults to 248056):
            Placeholder token id replaced by image patch embeddings.
        video_token_id (`int`, *optional*, defaults to 248057):
            Placeholder token id replaced by video patch embeddings.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            `<|vision_start|>` marker id.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            `<|vision_end|>` marker id.

    Examples:

    ```python
    >>> from zeromodels.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeConditionalGenerate

    >>> configuration = Qwen3_5MoeConfig(
    ...     text_config={"embed_dim": 2048, "num_layers": 40},
    ...     vision_config={"depth": 27, "out_dim": 2048},
    ... )
    >>> model = Qwen3_5MoeConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_5_moe"

    sub_configs = {
        "text_config": Qwen3_5MoeTextConfig,
        "vision_config": Qwen3_5MoeVisionConfig,
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

    text_config: Qwen3_5MoeTextConfig | dict | None = None
    vision_config: Qwen3_5MoeVisionConfig | dict | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


QWEN3_5_MOE_TOKENS = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}
