GLM5_MOE_COMMON = {
    "vocab_size": 154880,
    "embed_dim": 6144,
    "num_layers": 78,
    "num_heads": 64,
    "mlp_dim": 12288,
    "moe_mlp_dim": 2048,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "first_k_dense": 3,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "num_mtp_layers": 1,
    "norm_eps": 1e-5,
    "attention_bias": False,
    "tie_embeddings": False,
}

GLM5_MOE_CONFIG = {
    "glm5": {
        **GLM5_MOE_COMMON,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 202752,
        "index_topk_freq": 1,
    },
    "glm5_1": {
        **GLM5_MOE_COMMON,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 202752,
        "index_topk_freq": 1,
    },
    "glm5_2": {
        **GLM5_MOE_COMMON,
        "rope_theta": 8000000.0,
        "max_position_embeddings": 1048576,
        "index_topk_freq": 4,
    },
}

GLM5_MOE_WEIGHTS_URLS = {
    "glm5": {"hf_id": "zai-org/GLM-5", "gated": False, "safetensors": True},
    "glm5_1": {"hf_id": "zai-org/GLM-5.1", "gated": False, "safetensors": True},
    "glm5_2": {"hf_id": "zai-org/GLM-5.2", "gated": False, "safetensors": True},
}
