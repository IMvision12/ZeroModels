from zeromodels.base import BaseConfig


class Inkling2LoopTextConfig(BaseConfig):
    r"""Text-decoder config for Inkling-2-Mini (``inkling2_loop_text``).

    Looped sparse Mixture-of-Experts hybrid transformer:
    - 24 physical decoder layers where the middle 12 layers (``6..17``) execute
      ``loop_num_passes=3`` times sequentially with shared weights (48 logical layers),
      passing cross-pass messages through ``Inkling2LoopMessageAttentionResidual`` and
      rank-64 ``Inkling2LoopMessageGate`` controllers.
    - Block attention residuals (``n_layers_per_attn_res_block=6``) with unweighted
      RMSNorm over accumulated block states.
    - Hybrid attention: 5 sliding-window layers (``sliding_window_size=1024``) followed
      by 1 full-attention layer every 6 layers, with per-token attention sinks,
      hidden-state-conditioned relative position bias, L2-clipped Q/K/Rel states,
      logarithmic query/rel scaling on full-attention layers, and causal short
      convolutions (static on sliding K/V and post-attn/post-mlp; dynamic on full K/V).
    - Sparse MoE (128 experts, top-8 sigmoid routing with ``e_score_correction_bias``
      and ``route_scale=8.0``) with softcapped SwiGLU and final logit softcapping.
    """

    model_type = "inkling2_loop_text"

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    embed_dim: int = 2304
    num_layers: int = 24
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    swa_num_heads: int = 24
    swa_num_kv_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    layer_types: tuple | None = None
    max_position_embeddings: int = 1048576
    norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    moe_mlp_dim: int = 1152
    num_experts: int = 128
    num_experts_per_tok: int = 8
    route_scale: float = 8.0
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64
    eos_token_id: int = 200006


class Inkling2VisionConfig(BaseConfig):
    r"""Vision-tower config for Inkling-2-Mini (``inkling2_vision``).

    23-layer Vision Transformer with ``20x20`` native patches, ``2x2`` spatial merge
    (``40x40`` pixels per merged image token), bilinearly interpolated ``32x32``
    learned position embeddings, per-image CLS token, 2D axial pairwise RoPE,
    LayerScale, and a 2-layer GELU merger projecting to ``text_hidden_size``.
    """

    model_type = "inkling2_vision"

    embed_dim: int = 1024
    depth: int = 23
    num_heads: int = 16
    mlp_ratio: float = 4.0
    mlp_dim: int | None = None
    in_channels: int = 3
    patch_size: int = 20
    spatial_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    out_dim: int = 2304
    norm_eps: float = 1e-5
    layer_scale_init_value: float = 0.1
    rope_theta: float = 10000.0


class Inkling2LoopConfig(BaseConfig):
    r"""Composite configuration for Inkling-2-Mini:
    [`Inkling2LoopModel`] and [`Inkling2LoopConditionalGenerate`].

    Wraps [`Inkling2LoopTextConfig`] (``text_config``) and
    [`Inkling2VisionConfig`] (``vision_config``) plus ``image_token_id``.
    """

    model_type = "inkling2_loop"

    sub_configs = {
        "text_config": Inkling2LoopTextConfig,
        "vision_config": Inkling2VisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "posemb_grid_size",
            "projection_hidden_dim",
            "projector_bias",
        )
    }

    text_config: Inkling2LoopTextConfig | dict | None = None
    vision_config: Inkling2VisionConfig | dict | None = None
    image_token_id: int = 200054


INKLING2_LOOP_TOKENS = {
    "image_token_id": 200054,
    "eos_token_id": 200006,
}
