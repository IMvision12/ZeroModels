from zeromodels.base import BaseConfig


class LlamaConfig(BaseConfig):
    """Typed config for the Llama family (Llama 3 / 3.1 / 3.2; rope_factor=None -> Llama 2)."""

    model_type = "llama"

    vocab_size: int = 128256
    embed_dim: int = 2048
    mlp_dim: int = 8192
    num_layers: int = 16
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_pos: int = 8192
    tie_embeddings: bool = True


LLAMA_CONFIG = {
    "llama3-8b": {
        "vocab_size": 128256,
        "embed_dim": 4096,
        "mlp_dim": 14336,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": None,
        "tie_embeddings": False,
    },
    "llama3-8b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 4096,
        "mlp_dim": 14336,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": None,
        "tie_embeddings": False,
    },
    "llama3-70b": {
        "vocab_size": 128256,
        "embed_dim": 8192,
        "mlp_dim": 28672,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": None,
        "tie_embeddings": False,
    },
    "llama3-70b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 8192,
        "mlp_dim": 28672,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": None,
        "tie_embeddings": False,
    },
    "llama3.1-8b": {
        "vocab_size": 128256,
        "embed_dim": 4096,
        "mlp_dim": 14336,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 8.0,
        "tie_embeddings": False,
    },
    "llama3.1-8b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 4096,
        "mlp_dim": 14336,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 8.0,
        "tie_embeddings": False,
    },
    "llama3.1-70b": {
        "vocab_size": 128256,
        "embed_dim": 8192,
        "mlp_dim": 28672,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 8.0,
        "tie_embeddings": False,
    },
    "llama3.1-70b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 8192,
        "mlp_dim": 28672,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 8.0,
        "tie_embeddings": False,
    },
    "llama3.2-1b": {
        "vocab_size": 128256,
        "embed_dim": 2048,
        "mlp_dim": 8192,
        "num_layers": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 32.0,
        "tie_embeddings": True,
    },
    "llama3.2-1b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 2048,
        "mlp_dim": 8192,
        "num_layers": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 32.0,
        "tie_embeddings": True,
    },
    "llama3.2-3b": {
        "vocab_size": 128256,
        "embed_dim": 3072,
        "mlp_dim": 8192,
        "num_layers": 28,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 32.0,
        "tie_embeddings": True,
    },
    "llama3.2-3b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 3072,
        "mlp_dim": 8192,
        "num_layers": 28,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 32.0,
        "tie_embeddings": True,
    },
    "llama3.3-70b-instruct": {
        "vocab_size": 128256,
        "embed_dim": 8192,
        "mlp_dim": 28672,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_factor": 8.0,
        "tie_embeddings": False,
    },
}

LLAMA_WEIGHTS_URLS = {
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "gated": True,
        "safetensors": True,
    },
    "llama3-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3-70b": {
        "hf_id": "meta-llama/Meta-Llama-3-70B",
        "gated": True,
        "safetensors": True,
    },
    "llama3-70b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "gated": True,
        "safetensors": True,
    },
    "llama3.1-8b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3.1-70b": {
        "hf_id": "meta-llama/Llama-3.1-70B",
        "gated": True,
        "safetensors": True,
    },
    "llama3.1-70b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "gated": True,
        "safetensors": True,
    },
    "llama3.2-1b-instruct": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "gated": True,
        "safetensors": True,
    },
    "llama3.2-3b-instruct": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "gated": True,
        "safetensors": True,
    },
    "llama3.3-70b-instruct": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "gated": True,
        "safetensors": True,
    },
}
