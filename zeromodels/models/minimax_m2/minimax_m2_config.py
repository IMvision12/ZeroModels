from zeromodels.base import BaseConfig


class MiniMaxM2Config(BaseConfig):
    """Typed config for MiniMax M2 (MoE)."""

    model_type = "minimax_m2"

    vocab_size: int = 200064
    embed_dim: int = 3072
    mlp_dim: int = 1536
    num_layers: int = 62
    num_heads: int = 48
    num_kv_heads: int = 8
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    partial_rotary_factor: float = 1.0
    rope_theta: float = 5000000.0
    norm_eps: float = 1e-06
    tie_embeddings: bool = False


MINIMAX_M2_CONFIG = {
    "minimax-m2": {
        "vocab_size": 200064,
        "embed_dim": 3072,
        "mlp_dim": 1536,
        "num_layers": 62,
        "num_heads": 48,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "rope_theta": 5000000.0,
        "norm_eps": 1e-6,
        "tie_embeddings": False,
    },
}

MINIMAX_M2_WEIGHTS_URLS = {
    "minimax-m2": {
        "hf_id": "MiniMaxAI/MiniMax-M2",
        "gated": False,
        "safetensors": True,
    },
}
