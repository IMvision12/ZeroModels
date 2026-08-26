from zeromodels.base import BaseConfig


class Qwen3NextConfig(BaseConfig):
    r"""Configuration for Qwen3-Next: [`Qwen3NextModel`] and [`Qwen3NextTextGenerate`].

    A hybrid decoder: most blocks are Gated-DeltaNet linear-attention layers, with a
    full-attention block every ``full_attention_interval``; both use a sparse MoE MLP
    (a softmax router over ``num_experts`` fused experts plus a sigmoid-gated shared
    expert). Full-attention blocks use partial-rotary GQA with per-head QK-norm.

    Args mirror [`Qwen3NextModel`]; see that class for per-field descriptions."""

    model_type = "qwen3_next"

    vocab_size: int = 151936
    embed_dim: int = 2048
    mlp_dim: int = 5120
    num_layers: int = 48
    num_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    tie_embeddings: bool = False
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_mlp_dim: int = 512
    shared_mlp_dim: int = 512
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()


QWEN3_NEXT_80B = {
    "vocab_size": 151936,
    "embed_dim": 2048,
    "mlp_dim": 5120,
    "num_layers": 48,
    "num_heads": 16,
    "num_kv_heads": 2,
    "head_dim": 256,
    "norm_eps": 1e-6,
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "tie_embeddings": False,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_mlp_dim": 512,
    "shared_mlp_dim": 512,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": (),
}

QWEN3_NEXT_CONFIG = {
    "qwen3-next-80b-a3b-instruct": dict(QWEN3_NEXT_80B),
    "qwen3-next-80b-a3b-thinking": dict(QWEN3_NEXT_80B),
}

QWEN3_NEXT_WEIGHTS_URLS = {
    "qwen3-next-80b-a3b-instruct": {
        "hf_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-next-80b-a3b-thinking": {
        "hf_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "gated": False,
        "safetensors": True,
    },
}
