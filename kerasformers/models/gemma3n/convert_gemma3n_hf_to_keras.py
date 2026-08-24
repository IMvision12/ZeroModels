import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# Text-decoder name map (keras-relative -> HF-relative, before any prefix). Order
# matters: longer / more specific keys first so a shorter key does not corrupt a
# longer one (post_attention_norm before attention_norm, query_norm before query).
TEXT_MAP = {
    "token_embedding.embeddings": "embed_tokens.weight",
    "embed_tokens_per_layer.embeddings": "embed_tokens_per_layer.weight",
    "final_norm.weight": "norm.weight",
    "decoder_layer_": "layers.",
    "altup_unembed_projection_": "altup_unembed_projections.",
    "altup_projection_": "altup_projections.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "post_attention_norm": "post_attention_layernorm",
    "pre_feedforward_norm": "pre_feedforward_layernorm",
    "post_feedforward_norm": "post_feedforward_layernorm",
    "attention_norm": "input_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

# Audio-tower name map (keras-relative -> HF-relative). conformer_ / block names and
# the depthwise conv kernel differ; everything else matches HF.
AUDIO_MAP = {
    "conformer_": "conformer.",
    "depthwise_conv1d_kernel": "depthwise_conv1d.weight",
    "kernel": "weight",
}

# Multimodal embedder map (keras-relative -> HF-relative).
EMBEDDER_MAP = {
    "embedding.embeddings": "embedding.weight",
    "kernel": "weight",
}

# MobileNet-V5 vision map: flat keras MQA layer names -> timm's Sequential-nested
# submodule names.
_MQA_MAP = {
    "attn.query_proj": "attn.query.proj",
    "attn.key_down_conv": "attn.key.down_conv",
    "attn.key_norm": "attn.key.norm",
    "attn.key_proj": "attn.key.proj",
    "attn.value_down_conv": "attn.value.down_conv",
    "attn.value_norm": "attn.value.norm",
    "attn.value_proj": "attn.value.proj",
    "attn.output_proj": "attn.output.proj",
}


def _apply(mapping, path):
    for old, new in mapping.items():
        path = path.replace(old, new)
    return path


def resolve_vision_name(body):
    """Map a MobileNet-V5 keras sub-path (relative to ``vision_tower/``) to its timm
    name (relative to ``timm_model.``)."""
    body = re.sub(r"blocks_(\d+)_(\d+)", r"blocks.\1.\2", body)
    body = _apply(_MQA_MAP, body)
    body = body.replace("depthwise_kernel", "weight").replace("kernel", "weight")
    return body


def resolve_text_name(rel, nested):
    rel = _apply(TEXT_MAP, rel)
    return ("language_model." + rel) if nested else rel


def resolve_fusion_name(rel):
    """Map a fusion-model keras path (model-name stripped) to its HF name."""
    if rel.startswith("language_model."):
        body = rel[len("language_model.") :]
        return "model.language_model." + _apply(TEXT_MAP, body)
    if rel.startswith("audio_tower."):
        body = rel[len("audio_tower.") :]
        return "model.audio_tower." + _apply(AUDIO_MAP, body)
    if rel.startswith(("embed_audio.", "embed_vision.")):
        head, body = rel.split(".", 1)
        return "model." + head + "." + _apply(EMBEDDER_MAP, body)
    if rel.startswith("vision_tower."):
        body = rel[len("vision_tower.") :]
        return "model.vision_tower.timm_model." + resolve_vision_name(body)
    if rel.startswith("lm_head."):
        return _apply({"kernel": "weight"}, rel)
    # Bare text decoder (standalone Gemma3nTextGenerate loaded from a full checkpoint).
    return "model.language_model." + _apply(TEXT_MAP, rel)


_FUSION_PREFIXES = (
    "language_model.",
    "audio_tower.",
    "vision_tower.",
    "embed_vision.",
    "embed_audio.",
    "lm_head.",
)


def is_fusion_model(keras_model):
    return any(
        w.path.replace("/", ".").startswith(_FUSION_PREFIXES)
        for w in keras_model.weights
    )


def transfer_gemma3n_weights(keras_model, hf_state_dict):
    """Transfer Gemma 3n weights (standalone text model or full multimodal model).

    The standalone :class:`Gemma3nTextModel` maps to the bare ``model.*`` text
    decoder, or the nested ``model.language_model.*`` decoder of a full checkpoint.
    The multimodal :class:`Gemma3nModel` / :class:`Gemma3nConditionalGenerate`
    routes each tower to its ``model.<tower>.*`` prefix and the tied / untied
    ``lm_head`` to the top level. The AltUp value norm and the KV-shared tail carry
    no weights, matching the Keras model."""
    if hasattr(keras_model, "build_for_transfer"):
        # Multimodal model: force-build every tower + embedder (a text-only
        # forward would leave the vision / audio weights uncreated and skipped).
        keras_model.build_for_transfer()
    elif not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})

    fusion = is_fusion_model(keras_model)
    nested = any(
        k.startswith(("model.language_model.", "language_model."))
        for k in hf_state_dict
    )
    text_prefix = "model." if any(k.startswith("model.") for k in hf_state_dict) else ""

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        rel = weight.path.replace("/", ".")
        if fusion:
            name = resolve_fusion_name(rel)
            # A tied lm_head has no own tensor; skip if absent (it reuses embeddings).
            if name not in hf_state_dict and rel.startswith("lm_head."):
                continue
        else:
            name = text_prefix + resolve_text_name(rel, nested)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        src = np.asarray(hf_state_dict[name])
        if "depthwise_conv1d_kernel" in weight.path:
            # torch depthwise Conv1d [C, 1, K] -> keras depthwise_conv kernel [K, C, 1].
            weight.assign(np.transpose(src, (2, 0, 1)))
        elif weight.path.endswith("/kernel") and src.ndim == 4:
            # 4D conv kernel. Regular Conv2d is torch [out, in, kh, kw] -> keras
            # [kh, kw, in, out]; keras DepthwiseConv2D is [kh, kw, C, 1] from torch
            # [C, 1, kh, kw]. Pick whichever transpose matches the target shape
            # (a regular conv with in==1 would fool a shape[1]==1 heuristic).
            # Handled here (not transfer_weights) because timm's proj/query/key/value
            # conv names lack the "conv" substring the shared util keys off.
            regular = np.transpose(src, (2, 3, 1, 0))
            if tuple(regular.shape) == tuple(weight.shape):
                weight.assign(regular)
            else:
                weight.assign(np.transpose(src, (2, 3, 0, 1)))
        elif weight.path.endswith("/kernel") and src.ndim == 2:
            # Any 2D Dense kernel: torch Linear [out, in] -> keras Dense [in, out].
            # Handled here (not via transfer_weights) because a "/kernel" whose path
            # contains "embedding" (embedding_projection, relative_position_embedding)
            # would otherwise be mistaken for an untransposed embedding table.
            weight.assign(src.T)
        else:
            transfer_weights(weight.path, weight, src)
