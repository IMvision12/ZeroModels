from zeromodels.base import BaseConfig


class MiniMaxConfig(BaseConfig):
    """Typed config for MiniMax (MoE + lightning/full hybrid attention)."""

    model_type = "minimax"

    vocab_size: int = 200064
    embed_dim: int = 6144
    mlp_dim: int = 9216
    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    num_experts: int = 32
    num_experts_per_tok: int = 2
    layer_types: tuple = None
    block_size: int = 256
    full_attn_alpha: float = 1.0
    full_attn_beta: float = 1.0
    linear_attn_alpha: float = 1.0
    linear_attn_beta: float = 1.0
    mlp_alpha: float = 1.0
    mlp_beta: float = 1.0
    partial_rotary_factor: float = 1.0
    rope_theta: float = 10000000.0
    norm_eps: float = 1e-05
    tie_embeddings: bool = False


MINIMAX_CONFIG = {
    "minimax-text-01": {
        "vocab_size": 200064,
        "embed_dim": 6144,
        "mlp_dim": 9216,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 32,
        "num_experts_per_tok": 2,
        "layer_types": tuple(
            "full_attention" if (i + 1) % 8 == 0 else "linear_attention"
            for i in range(80)
        ),
        "block_size": 256,
        "full_attn_alpha": 3.5565588200778455,
        "full_attn_beta": 1.0,
        "linear_attn_alpha": 3.5565588200778455,
        "linear_attn_beta": 1.0,
        "mlp_alpha": 3.5565588200778455,
        "mlp_beta": 1.0,
        "rope_theta": 10000000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": False,
    },
}

MINIMAX_WEIGHTS_URLS = {
    "minimax-text-01": {
        "hf_id": "MiniMaxAI/MiniMax-Text-01-hf",
        "gated": False,
        "safetensors": True,
    },
}
