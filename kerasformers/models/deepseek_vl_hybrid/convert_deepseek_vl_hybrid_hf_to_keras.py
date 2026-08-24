import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights
from kerasformers.models.deepseek_vl.convert_deepseek_vl_hf_to_keras import (
    TEXT_MAPPING,
    VISION_MAPPING,
    normalize_keys,
)

SAM_LAYER_MAPPING = {
    "mlp_lin1": "mlp.lin1",
    "mlp_lin2": "mlp.lin2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

HIGH_RES_PREFIXES = (
    "high_res_vision_model",
    "high_res_vision_neck",
    "high_res_vision_proj",
    "high_res_vision_alpha",
    "aligner",
)


def transfer_deepseek_vl_hybrid_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        size = keras_model.image_size
        hr = keras_model.high_res_image_size
        n_tokens = (size // keras_model.patch_size) ** 2
        ids = np.array(
            [[0] + [keras_model.image_token_id] * n_tokens + [1]], dtype="int32"
        )
        keras_model(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "pixel_values": np.zeros((1, size, size, 3), dtype="float32"),
                "high_res_pixel_values": np.zeros((1, hr, hr, 3), dtype="float32"),
            }
        )

    for weight in tqdm(keras_model.weights, desc="Transferring text + SigLIP"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        if name.startswith(HIGH_RES_PREFIXES):
            continue
        mapping = VISION_MAPPING if name.startswith("vision_model.") else TEXT_MAPPING
        hf_name = name
        for old, new in mapping.items():
            hf_name = hf_name.replace(old, new)
        if hf_name not in state:
            raise WeightMappingError(weight.path, hf_name)
        if hf_name.endswith("patch_embedding.weight"):
            weight.assign(np.transpose(np.asarray(state[hf_name]), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, state[hf_name])

    enc = keras_model.get_layer("high_res_vision_model")
    p = "high_res_vision_model.vision_encoder"
    transfer_weights(
        "conv_kernel",
        enc.patch_embed.kernel,
        state[f"{p}.patch_embed.projection.weight"],
    )
    enc.patch_embed.bias.assign(state[f"{p}.patch_embed.projection.bias"])
    enc.pos_embed.pos_embed.assign(state[f"{p}.pos_embed"])
    for i, block in enumerate(tqdm(enc.blocks, desc="Transferring SAM encoder layers")):
        for w in block.weights:
            # Suffix relative to the (nested) block, e.g. "attn/qkv/kernel".
            suffix = w.path.split(f"/{block.name}/", 1)[-1].replace("/", ".")
            if "rel_pos" in suffix:
                w.assign(state[f"{p}.layers.{i}.{suffix}"])
                continue
            for old, new in SAM_LAYER_MAPPING.items():
                suffix = suffix.replace(old, new)
            transfer_weights(w.path, w, state[f"{p}.layers.{i}.{suffix}"])
    transfer_weights(
        "conv_kernel", enc.neck_conv1.kernel, state[f"{p}.neck.conv1.weight"]
    )
    enc.neck_ln1.gamma.assign(state[f"{p}.neck.layer_norm1.weight"])
    enc.neck_ln1.beta.assign(state[f"{p}.neck.layer_norm1.bias"])
    transfer_weights(
        "conv_kernel", enc.neck_conv2.kernel, state[f"{p}.neck.conv2.weight"]
    )
    enc.neck_ln2.gamma.assign(state[f"{p}.neck.layer_norm2.weight"])
    enc.neck_ln2.beta.assign(state[f"{p}.neck.layer_norm2.bias"])

    neck = keras_model.get_layer("high_res_vision_neck")
    transfer_weights(
        "conv_kernel", neck.conv1.kernel, state["high_res_vision_neck.conv1.weight"]
    )
    neck.layer_norm1.gamma.assign(state["high_res_vision_neck.layer_norm1.weight"])
    neck.layer_norm1.beta.assign(state["high_res_vision_neck.layer_norm1.bias"])
    transfer_weights(
        "conv_kernel", neck.conv2.kernel, state["high_res_vision_neck.conv2.weight"]
    )
    neck.layer_norm2.gamma.assign(state["high_res_vision_neck.layer_norm2.weight"])
    neck.layer_norm2.beta.assign(state["high_res_vision_neck.layer_norm2.bias"])

    proj = keras_model.get_layer("high_res_vision_proj")
    transfer_weights(
        "conv_kernel", proj.conv1.kernel, state["high_res_vision_proj.conv1.weight"]
    )
    transfer_weights(
        "conv_kernel", proj.conv2.kernel, state["high_res_vision_proj.conv2.weight"]
    )

    keras_model.high_res_vision_alpha.assign(
        np.asarray(state["high_res_vision_alpha"]).reshape(-1)
    )

    al = keras_model.get_layer("aligner")
    for attr, hf in [
        (al.vision_proj, "aligner.vision_proj"),
        (al.high_res_vision_proj, "aligner.high_res_vision_proj"),
        (al.proj, "aligner.proj"),
    ]:
        transfer_weights("kernel", attr.kernel, state[f"{hf}.weight"])
        attr.bias.assign(state[f"{hf}.bias"])


# Per-variant recipes (relocated from deepseek_vl_hybrid_config.py). Models load
# from the Hub by repo id; these build the arch for conversion + drive the backfill.
DEEPSEEK_VL_HYBRID_VARIANTS = {
    "deepseek_vl_7b_chat": {
        "vocab_size": 102400,
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
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
        "high_res_embed_dim": 768,
        "high_res_mlp_dim": 3072,
        "high_res_num_layers": 12,
        "high_res_num_heads": 12,
        "high_res_image_size": 1024,
        "high_res_patch_size": 16,
        "high_res_output_channels": 256,
        "high_res_window_size": 14,
        "high_res_global_attn_indexes": (2, 5, 8, 11),
        "high_res_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
    "deepseek_vl_7b_base": {
        "vocab_size": 102400,
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
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
        "high_res_embed_dim": 768,
        "high_res_mlp_dim": 3072,
        "high_res_num_layers": 12,
        "high_res_num_heads": 12,
        "high_res_image_size": 1024,
        "high_res_patch_size": 16,
        "high_res_output_channels": 256,
        "high_res_window_size": 14,
        "high_res_global_attn_indexes": (2, 5, 8, 11),
        "high_res_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
}


if __name__ == "__main__":
    import gc

    import keras

    from kerasformers.models.deepseek_vl_hybrid import DeepseekVLHybridModel

    HF_SOURCES = {
        "deepseek_vl_7b_chat": "deepseek-community/deepseek-vl-7b-chat",
        "deepseek_vl_7b_base": "deepseek-community/deepseek-vl-7b-base",
    }
    MAX_SHARD_GB = 1.7

    for variant in DEEPSEEK_VL_HYBRID_VARIANTS:
        hf_id = HF_SOURCES[variant]
        out_path = f"{variant}.weights.json"
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = DeepseekVLHybridModel.from_weights("hf:" + hf_id)

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB, sharded)")

        del model
        keras.backend.clear_session()
        gc.collect()
