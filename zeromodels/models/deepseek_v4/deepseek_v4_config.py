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
