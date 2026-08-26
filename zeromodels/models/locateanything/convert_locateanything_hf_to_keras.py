import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "vision_model.patch_proj": "vision_model.patch_embed.proj",
    "vision_model.pos_emb": "vision_model.patch_embed.pos_emb.weight",
    "vision_model.final_norm": "vision_model.encoder.final_layernorm",
    "block_": "encoder.blocks.",
    "mlp1_norm": "mlp1.0",
    "mlp1_fc1": "mlp1.1",
    "mlp1_fc2": "mlp1.3",
    "token_embedding.embeddings": "language_model.model.embed_tokens.weight",
    "final_norm.weight": "language_model.model.norm.weight",
    "decoder_layer_": "language_model.model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def hf_name_for(path):
    for old, new in WEIGHT_NAME_MAPPING.items():
        path = path.replace(old, new)
    return path


def build_for_transfer(keras_model):
    grid = np.array([[2, 2]], dtype="int64")
    pixel_values = np.zeros((4, 3, 14, 14), dtype="float32")
    img = keras_model.image_token_index
    input_ids = np.array([[img, 0, 0, 0]], dtype="int64")
    keras_model(
        {"input_ids": input_ids, "pixel_values": pixel_values, "image_grid_hws": grid}
    )


def transfer_locateanything_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_for_transfer(keras_model)
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        path = weight.path.replace("/", ".")
        name = hf_name_for(path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        value = hf_state_dict[name]
        if path.endswith("patch_proj.kernel"):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif path.endswith("pos_emb"):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)


# Single-variant recipe (relocated from locateanything_config.py). The conversion
# below builds from the HF config.json via config_from_hf; this recipe drives the
# kf_config backfill so the repo declares LocateAnythingConditionalGenerate.
LOCATEANYTHING_RECIPES = {
    "locateanything_3b": {
        "vocab_size": 152681,
        "embed_dim": 2048,
        "mlp_dim": 11008,
        "num_layers": 36,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_embeddings": True,
        "vision_embed_dim": 1152,
        "vision_depth": 27,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4304,
        "vision_patch_size": 14,
        "vision_init_pos_h": 64,
        "vision_init_pos_w": 64,
        "merge_kernel": (2, 2),
        "vision_rope_theta": 10000.0,
        "image_token_index": 151665,
        "block_size": 6,
    },
}


def safetensors_state_dict(files):
    from safetensors import safe_open

    handles = {}
    for f in files:
        fh = safe_open(f, framework="pt")
        for k in fh.keys():
            handles[k] = fh

    class _View:
        def __contains__(self, k):
            return k in handles

        def __getitem__(self, k):
            return handles[k].get_tensor(k).float().cpu().numpy()

    return _View()


if __name__ == "__main__":
    import gc
    import glob
    import json
    import os

    import keras
    from huggingface_hub import snapshot_download

    from zeromodels.models.locateanything import LocateAnythingConditionalGenerate

    DTYPE = "bfloat16"
    MAX_SHARD_GB = 1.7
    HF_SOURCES = {"locateanything_3b": "nvidia/LocateAnything-3B"}
    OUTPUTS = {"locateanything_3b": "model.weights.json"}

    keras.config.set_dtype_policy(DTYPE)

    for variant, weights_path in OUTPUTS.items():
        hf_id = HF_SOURCES[variant]
        print(
            f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}  ({DTYPE})\n{'=' * 60}"
        )

        local = snapshot_download(
            hf_id, allow_patterns=["*.json", "*.txt", "*.safetensors"]
        )

        with open(os.path.join(local, "config.json")) as f:
            hf_config = json.load(f)
        model = LocateAnythingConditionalGenerate(
            **LocateAnythingConditionalGenerate.config_from_hf(hf_config)
        )
        shards = sorted(glob.glob(os.path.join(local, "*.safetensors")))
        transfer_locateanything_weights(model, safetensors_state_dict(shards))

        if weights_path.endswith(".json"):
            model.save_weights(weights_path, max_shard_size=MAX_SHARD_GB)
        else:
            model.save_weights(weights_path)
        print(f"  Saved weights -> {weights_path}")

        del model
        keras.backend.clear_session()
        gc.collect()
