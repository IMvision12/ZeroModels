from zeromodels.base import BaseConfig


class Glm4vConfig(BaseConfig):
    """Configuration for GLM-4.1V / 4.6V-Flash (dense VLM): [`Glm4vModel`] and
    [`Glm4vConditionalGenerate`].

    A GLM-4V vision tower (Conv3d patch embed, bicubic-interpolated learned positions,
    packed-attention rotary blocks, 2x2 downsample, SwiGLU merger) feeds image
    embeddings into the ``image_token_id`` slots of a GLM-4 dense decoder (sandwich
    norms, biased q/k/v, partial interleaved rope), fused by 3D merged M-RoPE.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Text SwiGLU width (``intermediate_size``).
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        partial_rotary_factor: Fraction of each head that receives rotary.
        norm_eps: Text RMSNorm epsilon.
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

    model_type = "glm4v"

    vocab_size: int = 151552
    embed_dim: int = 4096
    mlp_dim: int = 13696
    num_layers: int = 40
    num_heads: int = 32
    num_kv_heads: int = 2
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
    image_token_id: int = 151343
    video_token_id: int = 151344
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
