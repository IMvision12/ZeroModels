from zeromodels.base import BaseConfig


class DeepseekV2Config(BaseConfig):
    """Typed config for DeepSeek-V2 (MLA + DeepSeekMoE)."""

    model_type = "deepseek_v2"

    vocab_size: int = 102400
    embed_dim: int = 2048
    num_layers: int = 27
    num_heads: int = 16
    mlp_dim: int = 10944
    moe_mlp_dim: int = 1408
    num_experts: int = 64
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    topk_method: str = "greedy"
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense: int = 1
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rope_scaling: dict = None
    norm_eps: float = 1e-06
    max_position_embeddings: int = 163840
    tie_embeddings: bool = False


DEEPSEEK_V2_CONFIG = {
    "deepseek-v2-lite": {
        "embed_dim": 2048,
        "num_layers": 27,
        "num_heads": 16,
        "mlp_dim": 10944,
        "moe_mlp_dim": 1408,
        "num_experts": 64,
        "topk_method": "greedy",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "q_lora_rank": None,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    },
    "deepseek-v2-lite-chat": {
        "embed_dim": 2048,
        "num_layers": 27,
        "num_heads": 16,
        "mlp_dim": 10944,
        "moe_mlp_dim": 1408,
        "num_experts": 64,
        "topk_method": "greedy",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "q_lora_rank": None,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    },
    "deepseek-v2": {
        "embed_dim": 5120,
        "num_layers": 60,
        "num_heads": 128,
        "mlp_dim": 12288,
        "moe_mlp_dim": 1536,
        "num_experts": 160,
        "topk_method": "group_limited_greedy",
        "n_group": 8,
        "topk_group": 3,
        "routed_scaling_factor": 16.0,
        "q_lora_rank": 1536,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    },
    "deepseek-v2-chat": {
        "embed_dim": 5120,
        "num_layers": 60,
        "num_heads": 128,
        "mlp_dim": 12288,
        "moe_mlp_dim": 1536,
        "num_experts": 160,
        "topk_method": "group_limited_greedy",
        "n_group": 8,
        "topk_group": 3,
        "routed_scaling_factor": 16.0,
        "q_lora_rank": 1536,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    },
    "deepseek-v2.5": {
        "embed_dim": 5120,
        "num_layers": 60,
        "num_heads": 128,
        "mlp_dim": 12288,
        "moe_mlp_dim": 1536,
        "num_experts": 160,
        "topk_method": "group_limited_greedy",
        "n_group": 8,
        "topk_group": 3,
        "routed_scaling_factor": 16.0,
        "q_lora_rank": 1536,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
    },
}

DEEPSEEK_V2_WEIGHTS_URLS = {
    "deepseek-v2-lite": {
        "hf_id": "deepseek-ai/DeepSeek-V2-Lite",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v2-lite-chat": {
        "hf_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v2": {
        "hf_id": "deepseek-ai/DeepSeek-V2",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v2-chat": {
        "hf_id": "deepseek-ai/DeepSeek-V2-Chat",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v2.5": {
        "hf_id": "deepseek-ai/DeepSeek-V2.5",
        "gated": False,
        "safetensors": True,
    },
}
