DEEPSEEK_COMMON = {
    "vocab_size": 129280,
    "embed_dim": 7168,
    "num_layers": 61,
    "num_heads": 128,
    "mlp_dim": 18432,
    "moe_mlp_dim": 2048,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "first_k_dense": 3,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    "norm_eps": 1e-6,
    "max_position_embeddings": 163840,
    "tie_embeddings": False,
}

DEEPSEEK_V3_CONFIG = {
    "deepseek-v3": dict(DEEPSEEK_COMMON),
    "deepseek-v3-0324": dict(DEEPSEEK_COMMON),
    "deepseek-v3.1": dict(DEEPSEEK_COMMON),
    "deepseek-r1": dict(DEEPSEEK_COMMON),
}

DEEPSEEK_V3_WEIGHTS_URLS = {
    "deepseek-v3": {
        "hf_id": "deepseek-ai/DeepSeek-V3",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v3-0324": {
        "hf_id": "deepseek-ai/DeepSeek-V3-0324",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v3.1": {
        "hf_id": "deepseek-ai/DeepSeek-V3.1",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-r1": {
        "hf_id": "deepseek-ai/DeepSeek-R1",
        "gated": False,
        "safetensors": True,
    },
}
