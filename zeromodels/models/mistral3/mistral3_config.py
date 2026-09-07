from zeromodels.base import BaseConfig


class Mistral3Config(BaseConfig):
    """Typed config for Mistral 3 (Mistral text decoder + Pixtral vision tower)."""

    model_type = "mistral3"

    vocab_size: int = 131072
    embed_dim: int = 5120
    mlp_dim: int = 32768
    num_layers: int = 40
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-5
    rope_theta: float = 1000000000.0
    tie_embeddings: bool = False
    vision_embed_dim: int = 1024
    vision_mlp_dim: int = 4096
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    image_size: int = 1540
    patch_size: int = 14
    vision_rope_theta: float = 10000.0
    spatial_merge_size: int = 2
    projector_bias: bool = False
    image_token_id: int = 10


MISTRAL3_CONFIG = {
    "mistral-small-3.1-24b-instruct": {
        "vocab_size": 131072,
        "embed_dim": 5120,
        "mlp_dim": 32768,
        "num_layers": 40,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 1000000000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 1540,
        "patch_size": 14,
        "vision_rope_theta": 10000.0,
        "spatial_merge_size": 2,
        "projector_bias": False,
        "image_token_id": 10,
    },
    "mistral-small-3.2-24b-instruct": {
        "vocab_size": 131072,
        "embed_dim": 5120,
        "mlp_dim": 32768,
        "num_layers": 40,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-5,
        "rope_theta": 1000000000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 1540,
        "patch_size": 14,
        "vision_rope_theta": 10000.0,
        "spatial_merge_size": 2,
        "projector_bias": False,
        "image_token_id": 10,
    },
}

MISTRAL3_WEIGHTS_URLS = {
    "mistral-small-3.1-24b-instruct": {
        "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "gated": True,
        "safetensors": True,
    },
    "mistral-small-3.2-24b-instruct": {
        "hf_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "gated": True,
        "safetensors": True,
    },
}
