from zeromodels.base import BaseConfig


class MiniMaxM3VLConfig(BaseConfig):
    """Typed config for MiniMax-M3-VL (hybrid-attn MoE decoder + vision tower)."""

    model_type = "minimax_m3_vl"

    vocab_size: int = 200064
    embed_dim: int = 6144
    mlp_dim: int = 3072
    dense_mlp_dim: int = 12288
    shared_mlp_dim: int = 3072
    num_layers: int = 60
    num_heads: int = 64
    num_kv_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 2.0
    layer_types: tuple = None
    mlp_layer_types: tuple = None
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    partial_rotary_factor: float = 0.5
    rope_theta: float = 5000000.0
    norm_eps: float = 1e-6
    vision_embed_dim: int = 1280
    vision_mlp_dim: int = 5120
    vision_num_layers: int = 32
    vision_num_heads: int = 16
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    vision_rope_theta: float = 10000.0
    vision_norm_eps: float = 1e-5
    projector_dim: int = 6144
    image_token_id: int = 200025
    video_token_id: int = 200026
    tie_embeddings: bool = False


MINIMAX_M3_VL_CONFIG = {
    "minimax-m3": {
        "vocab_size": 200064,
        "embed_dim": 6144,
        "mlp_dim": 3072,
        "dense_mlp_dim": 12288,
        "shared_mlp_dim": 3072,
        "num_layers": 60,
        "num_heads": 64,
        "num_kv_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "routed_scaling_factor": 2.0,
        "layer_types": tuple(
            "full_attention" if i < 3 else "minimax_m3_sparse" for i in range(60)
        ),
        "mlp_layer_types": tuple("dense" if i < 3 else "sparse" for i in range(60)),
        "index_n_heads": 4,
        "index_head_dim": 128,
        "index_block_size": 128,
        "index_topk_blocks": 16,
        "index_local_blocks": 1,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "partial_rotary_factor": 0.5,
        "rope_theta": 5000000.0,
        "norm_eps": 1e-6,
        "vision_embed_dim": 1280,
        "vision_mlp_dim": 5120,
        "vision_num_layers": 32,
        "vision_num_heads": 16,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "vision_rope_theta": 10000.0,
        "vision_norm_eps": 1e-5,
        "projector_dim": 6144,
        "image_token_id": 200025,
        "video_token_id": 200026,
        "tie_embeddings": False,
    },
}

MINIMAX_M3_VL_WEIGHTS_URLS = {
    "minimax-m3": {
        "hf_id": "MiniMaxAI/MiniMax-M3",
        "gated": False,
        "safetensors": True,
    },
}
