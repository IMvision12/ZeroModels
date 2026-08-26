# K2.5 / K2.6 / K2.7-Code are the same architecture (all ship model_type
# "kimi_k25") and differ only in which checkpoint is pulled.
KIMI_K25_COMMON = {
    # text tower (DeepSeek-V3 shaped; HF `text_config`, model_type "kimi_k2")
    "vocab_size": 163840,
    "embed_dim": 7168,
    "num_layers": 61,
    "num_heads": 64,
    "mlp_dim": 18432,
    "moe_mlp_dim": 2048,
    "num_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    # Kimi disables expert grouping (DeepSeek-V3 uses 8/4), so grouped-topk
    # degenerates to a plain top-k over all 384 experts.
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.827,
    "first_k_dense": 1,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_theta": 50000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 64.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    # 1e-5, NOT the 1e-6 DeepSeek-V3 default.
    "norm_eps": 1e-5,
    "max_position_embeddings": 262144,
    # The published configs set tie_word_embeddings false at both levels and the
    # checkpoints carry language_model.lm_head.weight (the True in HF's config
    # dataclass is only a default).
    "tie_embeddings": False,
    # vision tower (MoonViT + temporal axis)
    "vision_embed_dim": 1152,
    "vision_depth": 27,
    "vision_num_heads": 16,
    "vision_mlp_dim": 4304,
    "vision_patch_size": 14,
    "pos_emb_height": 64,
    "pos_emb_width": 64,
    "pos_emb_time": 4,
    "merge_kernel": (2, 2),
    "vision_rope_theta": 10000.0,
    # multimodal projector
    "projection_hidden_size": 1152,
    "projection_norm_eps": 1e-5,
    # placeholder ids (note: video_token_id == vocab_size, so it is out of the
    # embedding range and must be zeroed before the token lookup)
    "image_token_id": 163605,
    "video_token_id": 163840,
    "vision_start_token_id": 163602,
    "vision_end_token_id": 163604,
}

KIMI_K25_CONFIG = {
    "kimi-k2.5": dict(KIMI_K25_COMMON),
    "kimi-k2.6": dict(KIMI_K25_COMMON),
    "kimi-k2.7-code": dict(KIMI_K25_COMMON),
}

KIMI_K25_WEIGHTS_URLS = {
    "kimi-k2.5": {
        "hf_id": "moonshotai/Kimi-K2.5",
        "gated": False,
        "safetensors": True,
    },
    "kimi-k2.6": {
        "hf_id": "moonshotai/Kimi-K2.6",
        "gated": False,
        "safetensors": True,
    },
    "kimi-k2.7-code": {
        "hf_id": "moonshotai/Kimi-K2.7-Code",
        "gated": False,
        "safetensors": True,
    },
}
