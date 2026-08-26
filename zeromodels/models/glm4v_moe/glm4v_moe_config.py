from zeromodels.base import BaseConfig


class Glm4vMoeConfig(BaseConfig):
    """Configuration for GLM-4.5V / 4.6V (MoE VLM): [`Glm4vMoeModel`] and
    [`Glm4vMoeConditionalGenerate`].

    A GLM-4V vision tower feeds image embeddings into the ``image_token_id`` slots of
    a GLM-4.5 MoE decoder (grouped-topk sigmoid router with a learned
    ``e_score_correction_bias`` over a fused-einsum expert bank plus a shared expert,
    partial NeoX rope; the first ``first_k_dense`` layers are dense), fused by 3D
    merged M-RoPE.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        moe_mlp_dim: Per-expert width (``moe_intermediate_size``).
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim.
        num_experts / num_experts_per_tok / n_shared_experts: MoE shape.
        n_group / topk_group: Group-limited routing (1/1 = plain top-k).
        norm_topk_prob: Renormalize the selected router weights.
        routed_scaling_factor: Scale applied to the routed-expert output.
        first_k_dense: Leading dense (non-MoE) layers.
        partial_rotary_factor: Fraction of each head that receives rotary.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        mrope_section: Per-axis channel split of the merged M-RoPE.
        tie_embeddings: Whether the LM head ties to the token embedding.
        vision_depth / vision_embed_dim / vision_num_heads / vision_mlp_dim /
        vision_out_dim / vision_norm_eps: Vision-tower geometry.
        image_size / patch_size / spatial_merge_size / temporal_patch_size /
        in_channels: Patch/grid geometry.
        image_token_id / video_token_id / image_start_token_id / image_end_token_id /
        video_start_token_id / video_end_token_id: Multimodal special tokens.
    """

    model_type = "glm4v_moe"

    vocab_size: int = 151424
    embed_dim: int = 4096
    mlp_dim: int = 10944
    moe_mlp_dim: int = 1408
    num_layers: int = 46
    num_heads: int = 96
    num_kv_heads: int = 8
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense: int = 1
    partial_rotary_factor: float = 0.5
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    mrope_section: tuple = (8, 12, 12)
    tie_embeddings: bool = False
    vision_depth: int = 24
    vision_embed_dim: int = 1536
    vision_num_heads: int = 12
    vision_mlp_dim: int = 13696
    vision_out_dim: int = 4096
    image_size: int = 336
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3
    vision_norm_eps: float = 1e-5
    image_token_id: int = 151363
    video_token_id: int = 151364
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
