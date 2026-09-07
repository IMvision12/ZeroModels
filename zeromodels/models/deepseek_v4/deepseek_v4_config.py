from zeromodels.base import BaseConfig


class DeepseekV4Config(BaseConfig):
    """Typed config for DeepSeek-V4 (MLA + DeepSeekMoE + sparse-indexer attention)."""

    model_type = "deepseek_v4"

    vocab_size: int = 129280
    embed_dim: int = 4096
    num_layers: int = 43
    num_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    layer_types: tuple = None
    mlp_layer_types: tuple = None
    num_experts: int = 256
    num_experts_per_tok: int = 6
    moe_mlp_dim: int = 2048
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict = {
        "type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }
    norm_eps: float = 1e-6
    tie_embeddings: bool = False


DEEPSEEK_V4_CONFIG = {
    "deepseek-v4-flash": {
        "embed_dim": 4096,
        "num_layers": 43,
        "num_heads": 64,
        "o_groups": 8,
        "num_experts": 256,
        "moe_mlp_dim": 2048,
        "routed_scaling_factor": 1.5,
        "index_topk": 512,
        "layer_types": tuple(
            ["sliding_attention"] * 2
            + [
                "compressed_sparse_attention"
                if i % 2 == 0
                else "heavily_compressed_attention"
                for i in range(41)
            ]
        ),
        "mlp_layer_types": tuple(["hash_moe"] * 3 + ["moe"] * 40),
    },
    "deepseek-v4-flash-base": {
        "embed_dim": 4096,
        "num_layers": 43,
        "num_heads": 64,
        "o_groups": 8,
        "num_experts": 256,
        "moe_mlp_dim": 2048,
        "routed_scaling_factor": 1.5,
        "index_topk": 512,
        "layer_types": tuple(
            ["sliding_attention"] * 2
            + [
                "compressed_sparse_attention"
                if i % 2 == 0
                else "heavily_compressed_attention"
                for i in range(41)
            ]
        ),
        "mlp_layer_types": tuple(["hash_moe"] * 3 + ["moe"] * 40),
    },
    "deepseek-v4-pro": {
        "embed_dim": 7168,
        "num_layers": 61,
        "num_heads": 128,
        "o_groups": 16,
        "num_experts": 384,
        "moe_mlp_dim": 3072,
        "routed_scaling_factor": 2.5,
        "index_topk": 1024,
        "layer_types": None,
        "mlp_layer_types": None,
    },
    "deepseek-v4-pro-base": {
        "embed_dim": 7168,
        "num_layers": 61,
        "num_heads": 128,
        "o_groups": 16,
        "num_experts": 384,
        "moe_mlp_dim": 3072,
        "routed_scaling_factor": 2.5,
        "index_topk": 1024,
        "layer_types": None,
        "mlp_layer_types": None,
    },
}

DEEPSEEK_V4_WEIGHTS_URLS = {
    "deepseek-v4-flash": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-flash-base": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash-Base",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-pro": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-pro-base": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro-Base",
        "gated": False,
        "safetensors": True,
    },
}
