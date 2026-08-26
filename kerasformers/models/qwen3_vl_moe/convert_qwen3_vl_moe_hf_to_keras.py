import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# Keras weight-path substring -> HF name. Text keys are normalized so
# ``model.language_model.*`` becomes ``model.*``; vision keys keep ``visual.*``. Order
# matters: the router / dense-MLP / vision rules run before the generic fallbacks, and
# the router ``mlp.gate_weight`` rule precedes the dense ``mlp.gate.kernel`` rule so it
# cannot be corrupted.
WEIGHT_NAME_MAPPING = {
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
    # MoE router (before the dense-gate rule).
    "mlp.gate_weight": "mlp.gate.weight",
    # Dense MLP (mlp_only_layers); explicit ``.kernel`` so it can't match the router.
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
    # Vision tower.
    "deepstack_merger_": "deepstack_merger_list.",
    "visual.pos_embed": "visual.pos_embed.weight",
    "blocks_": "blocks.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def fuse_expert_weights(state):
    """Fuse per-expert ``mlp.experts.N.{gate,up,down}_proj`` into the fused
    ``mlp.experts.gate_up_proj`` (E, 2I, H) / ``down_proj`` (E, H, I)."""
    if not any(EXPERT_RE.match(k) for k in state):
        return state
    out, gate, up, down = {}, {}, {}, {}
    for key, value in state.items():
        match = EXPERT_RE.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"gate_proj": gate, "up_proj": up, "down_proj": down}[which].setdefault(
                layer, {}
            )[expert] = value
        else:
            out[key] = value
    for layer in gate:
        experts = sorted(gate[layer])
        out[f"{layer}.mlp.experts.gate_up_proj"] = np.stack(
            [
                np.concatenate(
                    [np.asarray(gate[layer][e]), np.asarray(up[layer][e])], axis=0
                )
                for e in experts
            ],
            axis=0,
        )
        out[f"{layer}.mlp.experts.down_proj"] = np.stack(
            [np.asarray(down[layer][e]) for e in experts], axis=0
        )
    return out


def transfer_qwen3_vl_moe_weights(keras_model, hf_state_dict):
    """Load an HF Qwen3-VL-MoE (``Qwen3VLMoeForConditionalGeneration``) state dict into
    a freshly built Keras model in place.

    Like the Qwen3-VL converter (learned ``pos_embed`` assigned directly, Conv3d patch
    embed reshaped for the Keras ``Dense``, DeepStack mergers) plus the Qwen3-MoE text
    rules: per-expert ``mlp.experts.N.*`` are fused into ``gate_up_proj`` / ``down_proj``
    and the router ``mlp.gate.weight`` is copied verbatim. HF ``model.language_model.*``
    keys are rewritten to ``model.*`` and ``model.visual.*`` to ``visual.*`` first.
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

    state = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        state[k] = v
    state = fuse_expert_weights(state)

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if (
            ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
        ):
            weight.assign(np.asarray(state[name]))  # fused / router: direct copy
        elif weight.path.endswith("pos_embed"):
            weight.assign(np.asarray(state[name]))
        elif "patch_embed" in weight.path and weight.path.endswith("kernel"):
            tw = np.asarray(state[name])
            transfer_weights(weight.path, weight, tw.reshape(tw.shape[0], -1))
        else:
            transfer_weights(weight.path, weight, state[name])
