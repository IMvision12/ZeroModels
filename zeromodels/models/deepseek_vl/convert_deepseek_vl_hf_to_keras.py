import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "language_model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

VISION_MAPPING = {
    "vision_model.patch_embed": "vision_model.vision_model.embeddings.patch_embedding",
    "vision_model.position_embedding.embeddings": (
        "vision_model.vision_model.embeddings.position_embedding.weight"
    ),
    "vision_model.post_layernorm": "vision_model.vision_model.post_layernorm",
    "vision_model.blocks_": "vision_model.vision_model.encoder.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.out_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

ALIGNER_MAPPING = {
    "aligner_linear1": "aligner.linear1",
    "aligner_linear2": "aligner.linear2",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def transfer_deepseek_vl_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        size = keras_model.image_size
        n_tokens = (size // keras_model.patch_size) ** 2
        ids = np.array(
            [[0] + [keras_model.image_token_id] * n_tokens + [1]], dtype="int32"
        )
        keras_model(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "pixel_values": np.zeros((1, size, size, 3), dtype="float32"),
            }
        )
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        if name.startswith("vision_model."):
            mapping = VISION_MAPPING
        elif name.startswith("aligner_"):
            mapping = ALIGNER_MAPPING
        else:
            mapping = TEXT_MAPPING
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if name.endswith("patch_embedding.weight"):
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, state[name])


# Per-variant recipes (relocated from deepseek_vl_config.py). Models load from
# the Hub by repo id; these build the arch for conversion + drive the backfill.
# Only the model_type "deepseek_vl" repos (the 1.3B chat/base) are loadable
# here. The 7B repos are "deepseek_vl_hybrid" (SAM branch) -- a different
# architecture -- and are intentionally absent.
DEEPSEEK_VL_VARIANTS = {
    "deepseek_vl_1.3b_chat": {
        "vocab_size": 102400,
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
    "deepseek_vl_1.3b_base": {
        "vocab_size": 102400,
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
}


if __name__ == "__main__":
    import gc

    import keras

    from zeromodels.models.deepseek_vl import DeepseekVLModel

    HF_SOURCES = {
        "deepseek_vl_1.3b_chat": "deepseek-community/deepseek-vl-1.3b-chat",
        "deepseek_vl_1.3b_base": "deepseek-community/deepseek-vl-1.3b-base",
    }
    MAX_SHARD_GB = 1.7

    for variant in DEEPSEEK_VL_VARIANTS:
        hf_id = HF_SOURCES[variant]
        out_path = f"{variant}.weights.json"
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = DeepseekVLModel.from_weights("hf:" + hf_id)

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB)")

        del model
        keras.backend.clear_session()
        gc.collect()
