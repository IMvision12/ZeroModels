import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

TEXT_MAP = {
    "token_embedding.embeddings": "embed_tokens.weight",
    "embed_tokens_per_layer.embeddings": "embed_tokens_per_layer.weight",
    "final_norm.weight": "norm.weight",
    "decoder_layer_": "layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "post_attention_norm": "post_attention_layernorm",
    "post_feedforward_norm_1": "post_feedforward_layernorm_1",
    "post_feedforward_norm_2": "post_feedforward_layernorm_2",
    "pre_feedforward_norm_2": "pre_feedforward_layernorm_2",
    "pre_feedforward_norm": "pre_feedforward_layernorm",
    "post_feedforward_norm": "post_feedforward_layernorm",
    "attention_norm": "input_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "router.proj": "router.proj",
    "router.scale": "router.scale",
    "router.per_expert_scale": "router.per_expert_scale",
    "kernel": "weight",
}


def resolve_hf_name(keras_path, nested):
    path = keras_path.replace("/", ".")
    if path.startswith("vision_tower."):
        path = path.replace("layers_", "encoder.layers.")
        if "patch_embedder.input_proj.kernel" in path:
            return path.replace(".kernel", ".weight")
        if "position_embedding_table" in path:
            return path
        if path.endswith(".kernel"):
            return path[: -len(".kernel")] + ".linear.weight"
        return path
    if path.startswith("embed_vision.") or path.startswith("embed_audio."):
        return path.replace(".kernel", ".weight")
    if path.startswith("audio_tower."):
        path = path.replace("layers_", "layers.")
        if path.endswith("depthwise_conv1d_kernel"):
            return path.replace("depthwise_conv1d_kernel", "depthwise_conv1d.weight")
        if path.endswith(".gamma"):
            return path.replace(".gamma", ".weight")
        for suffix in (
            "conv.kernel",
            "input_proj_linear.kernel",
            "relative_k_proj.kernel",
            "output_proj.kernel",
        ):
            if path.endswith(suffix):
                return path.replace(".kernel", ".weight")
        if path.endswith(".kernel"):
            return path[: -len(".kernel")] + ".linear.weight"
        return path
    for old, new in TEXT_MAP.items():
        path = path.replace(old, new)
    return ("language_model." + path) if nested else path


def transfer_gemma4_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        if hasattr(keras_model, "build_for_transfer"):
            keras_model.build_for_transfer()
        else:
            keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})

    nested = any(
        k.startswith(("model.language_model.", "language_model."))
        for k in hf_state_dict
    )
    prefix = "model." if any(k.startswith("model.") for k in hf_state_dict) else ""

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = prefix + resolve_hf_name(weight.path, nested)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]
        if weight.path.endswith("depthwise_conv1d_kernel"):
            weight.assign(np.transpose(np.asarray(torch_weight), (2, 0, 1)))
        elif len(weight.shape) == 0:
            weight.assign(np.asarray(torch_weight))
        elif weight.path.endswith("position_embedding_table"):
            weight.assign(np.asarray(torch_weight))
        elif weight.path.endswith("embedding_projection/kernel"):
            weight.assign(np.asarray(torch_weight).T)
        elif ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            weight.assign(np.asarray(torch_weight))
        elif name.endswith("router.scale") or name.endswith("router.per_expert_scale"):
            weight.assign(np.asarray(torch_weight))
        else:
            transfer_weights(weight.path, weight, torch_weight)
