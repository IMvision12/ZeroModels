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
