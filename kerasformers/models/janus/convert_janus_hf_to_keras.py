import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

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
    "vision_model.patch_embed": "vision_model.embeddings.patch_embedding",
    "vision_model.position_embedding.embeddings": (
        "vision_model.embeddings.position_embedding.weight"
    ),
    "vision_model.post_layernorm": "vision_model.post_layernorm",
    "vision_model.blocks_": "vision_model.encoder.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.projection_layer",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

ALIGNER_MAPPING = {
    "aligner_fc1": "aligner.fc1",
    "aligner_hidden": "aligner.hidden_layers.0",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def transfer_janus_weights(keras_model, hf_state_dict):
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


# Per-variant recipes (relocated from janus_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the backfill.
JANUS_VARIANTS = {
    "janus_pro_1b": {
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "image_token_id": 100581,
    },
    "janus_pro_7b": {
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
        "image_token_id": 100594,
    },
}


if __name__ == "__main__":
    import gc

    import keras
    import torch
    import transformers
    from PIL import Image

    from kerasformers.models.janus import JanusModel, JanusProcessor

    HF_SOURCES = {
        "janus_pro_1b": "deepseek-community/Janus-Pro-1B",
        "janus_pro_7b": "deepseek-community/Janus-Pro-7B",
    }
    MAX_SHARD_GB = 1.7
    rng = np.random.default_rng(0)

    def cosine(a, b):
        a, b = a.astype("float64").ravel(), b.astype("float64").ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for variant in JANUS_VARIANTS:
        hf_id = HF_SOURCES[variant]
        out_path = f"{variant}.weights.json"
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = JanusModel.from_weights("hf:" + hf_id)

        img = Image.fromarray(rng.integers(0, 255, (384, 384, 3), dtype="uint8"))
        proc = JanusProcessor.from_weights("hf:" + hf_id)
        kin = proc(text="<image_placeholder>\nDescribe the image.", images=img)
        k_h = np.asarray(keras.ops.convert_to_numpy(model(kin)["last_hidden_state"]))

        hf_model = (
            transformers.JanusForConditionalGeneration.from_pretrained(  # FLAG: class
                hf_id
            ).eval()
        )
        ids = np.asarray(keras.ops.convert_to_numpy(kin["input_ids"])).astype("int64")
        pv = np.transpose(  # FLAG: HF pixel layout (channels-first; add num_images dim if needed)
            np.asarray(keras.ops.convert_to_numpy(kin["pixel_values"])), (0, 3, 1, 2)
        )
        with torch.no_grad():
            hf_out = hf_model(
                input_ids=torch.from_numpy(ids),
                pixel_values=torch.from_numpy(pv),
                output_hidden_states=True,
            )
        hf_h = hf_out.hidden_states[-1].float().numpy()
        cos = cosine(k_h, hf_h)
        print(f"  last_hidden_state cosine: {cos:.6f}")
        if cos < 0.99:
            raise ValueError(f"{variant}: Janus parity failed (cos={cos:.4f})")

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB)")

        del hf_model, model
        keras.backend.clear_session()
        gc.collect()
