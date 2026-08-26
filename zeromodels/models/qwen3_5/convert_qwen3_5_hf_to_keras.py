import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "linear_attn.conv_weight": "linear_attn.conv1d.weight",
    "kernel": "weight",
}


def split_gated_deltanet_in_proj(layer, qkvz, ba):
    nk, nv = layer.num_k_heads, layer.num_v_heads
    hk, hv = layer.head_k_dim, layer.head_v_dim
    ratio = nv // nk
    hidden = qkvz.shape[1]
    group = 2 * hk + 2 * ratio * hv
    qkvz = qkvz.reshape(nk, group, hidden)
    q = qkvz[:, :hk].reshape(nk * hk, hidden)
    k = qkvz[:, hk : 2 * hk].reshape(nk * hk, hidden)
    v = qkvz[:, 2 * hk : 2 * hk + ratio * hv].reshape(nv * hv, hidden)
    z = qkvz[:, 2 * hk + ratio * hv :].reshape(nv * hv, hidden)
    ba = ba.reshape(nk, 2 * ratio, hidden)
    b = ba[:, :ratio].reshape(nv, hidden)
    a = ba[:, ratio:].reshape(nv, hidden)
    return np.concatenate([q, k, v], axis=0), z, b, a


def transfer_qwen3_5_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    lm = (
        "model.language_model."
        if "model.language_model.norm.weight" in hf_state_dict
        else "model."
    )
    presplit = any(k.endswith("linear_attn.in_proj_qkv.weight") for k in hf_state_dict)

    handled = set()
    for i, decoder_layer in enumerate(keras_model.decoder_layers):
        if getattr(decoder_layer, "layer_type", None) == "full_attention":
            continue
        gdn = decoder_layer.linear_attn
        prefix = f"{lm}layers.{i}.linear_attn"
        if presplit:
            targets = (
                (gdn.in_proj_qkv, f"{prefix}.in_proj_qkv.weight"),
                (gdn.in_proj_z, f"{prefix}.in_proj_z.weight"),
                (gdn.in_proj_b, f"{prefix}.in_proj_b.weight"),
                (gdn.in_proj_a, f"{prefix}.in_proj_a.weight"),
            )
            for dense, key in targets:
                transfer_weights(dense.kernel.path, dense.kernel, hf_state_dict[key])
                handled.add(dense.kernel.path)
        else:
            qkvz = np.asarray(hf_state_dict[f"{prefix}.in_proj_qkvz.weight"])
            ba = np.asarray(hf_state_dict[f"{prefix}.in_proj_ba.weight"])
            qkv_w, z_w, b_w, a_w = split_gated_deltanet_in_proj(gdn, qkvz, ba)
            for dense, packed in (
                (gdn.in_proj_qkv, qkv_w),
                (gdn.in_proj_z, z_w),
                (gdn.in_proj_b, b_w),
                (gdn.in_proj_a, a_w),
            ):
                transfer_weights(dense.kernel.path, dense.kernel, packed)
                handled.add(dense.kernel.path)

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        if weight.path in handled:
            continue
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if lm != "model." and name.startswith("model."):
            name = lm + name[len("model.") :]
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]
        if "conv_weight" in weight.path:
            weight.assign(np.asarray(torch_weight).squeeze(1))
        else:
            transfer_weights(weight.path, weight, torch_weight)


# ---- Dense Qwen3.5 VLM (Qwen3_5ConditionalGenerate) ----------------------------
# Keras weight-path substring -> HF name, for the multimodal checkpoint. Text keys are
# normalized so ``model.language_model.*`` becomes ``model.*``; vision keys keep
# ``visual.*``. Order matters: specific rules run before the generic fallbacks.
VL_WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.norm.weight",
    "language_model.": "model.",
    "decoder_layer_": "layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "linear_attn.conv_weight": "linear_attn.conv1d.weight",
    "visual.pos_embed": "visual.pos_embed.weight",
    "blocks_": "blocks.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def transfer_qwen3_5_vl_weights(keras_model, hf_state_dict):
    """Load an HF dense Qwen3.5 VLM (``Qwen3_5ForConditionalGeneration``, e.g.
    Qwen3.5-27B / Qwen3.8-27B) state dict into a built :class:`Qwen3_5ConditionalGenerate`.

    Combines the dense text rules (split the fused Gated-DeltaNet ``in_proj_qkvz`` /
    ``in_proj_ba``, squeeze the conv1d weight, dense GeGLU MLP) with the Qwen3-VL vision
    rules (learned ``pos_embed`` assigned directly, Conv3d patch embed reshaped for the
    Keras ``Dense``). HF ``model.language_model.*`` keys are rewritten to ``model.*`` and
    ``model.visual.*`` to ``visual.*`` first.
    """
    if not keras_model.built or not keras_model.weights:
        m = keras_model.spatial_merge_size
        h = w = 2 * m
        n_merged = (h * w) // (m * m)
        keras_model(
            {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n_merged + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros(
                    (h * w, keras_model.patch_dim), dtype="float32"
                ),
                "image_grid_thw": np.array([[1, h, w]], dtype=np.int64),
            }
        )

    # Normalize the VLM key namespaces.
    state = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        state[k] = v

    # Gated-DeltaNet: split the fused HF in_proj into the 4 Keras projections.
    handled = set()
    for i, decoder_layer in enumerate(keras_model.language_model.decoder_layers):
        if getattr(decoder_layer, "layer_type", None) == "full_attention":
            continue
        gdn = decoder_layer.linear_attn
        prefix = f"model.layers.{i}.linear_attn"
        qkvz = np.asarray(state[f"{prefix}.in_proj_qkvz.weight"])
        ba = np.asarray(state[f"{prefix}.in_proj_ba.weight"])
        qkv_w, z_w, b_w, a_w = split_gated_deltanet_in_proj(gdn, qkvz, ba)
        for dense, packed in (
            (gdn.in_proj_qkv, qkv_w),
            (gdn.in_proj_z, z_w),
            (gdn.in_proj_b, b_w),
            (gdn.in_proj_a, a_w),
        ):
            transfer_weights(dense.kernel.path, dense.kernel, packed)
            handled.add(dense.kernel.path)

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        if weight.path in handled:
            continue
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in VL_WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if "conv_weight" in weight.path:
            weight.assign(np.asarray(state[name]).squeeze(1))  # (D,1,K) -> (D,K)
        elif weight.path.endswith("pos_embed"):
            weight.assign(np.asarray(state[name]))
        elif "patch_embed" in weight.path and weight.path.endswith("kernel"):
            tw = np.asarray(state[name])
            transfer_weights(weight.path, weight, tw.reshape(tw.shape[0], -1))
        else:
            transfer_weights(weight.path, weight, state[name])
