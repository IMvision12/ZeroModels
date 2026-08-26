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
