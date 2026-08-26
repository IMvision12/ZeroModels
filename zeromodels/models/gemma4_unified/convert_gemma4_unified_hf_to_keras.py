import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.gemma4.convert_gemma4_hf_to_keras import TEXT_MAP


def resolve_hf_name(keras_path):
    path = keras_path.replace("/", ".")
    if path.startswith("embed_vision."):
        rest = path[len("embed_vision.") :]
        if rest.startswith("multimodal_embedder."):
            sub = rest[len("multimodal_embedder.") :].replace(".kernel", ".weight")
            return "embed_vision." + sub
        if rest == "pos_embedding":
            return "vision_embedder.pos_embedding"
        rest = (
            rest.replace(".gamma", ".weight")
            .replace(".beta", ".bias")
            .replace(".kernel", ".weight")
        )
        return "vision_embedder." + rest
    if path.startswith("embed_audio."):
        return "embed_audio." + path[len("embed_audio.") :].replace(
            ".kernel", ".weight"
        )
    for old, new in TEXT_MAP.items():
        path = path.replace(old, new)
    return "language_model." + path


def transfer_gemma4_unified_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        if hasattr(keras_model, "build_for_transfer"):
            keras_model.build_for_transfer()
        else:
            keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})

    prefix = "model." if any(k.startswith("model.") for k in hf_state_dict) else ""

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = prefix + resolve_hf_name(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]
        if weight.path.endswith("pos_embedding"):
            weight.assign(np.asarray(torch_weight))
        elif weight.path.endswith("embedding_projection/kernel"):
            # A Dense whose name trips transfer_weights' "embedding" heuristic
            # (direct copy, no transpose); assign the transpose ourselves.
            weight.assign(np.asarray(torch_weight).T)
        elif weight.path.endswith(("/gamma", "/beta", "/bias")):
            # LayerNorm scale / shift and Dense bias: 1-D direct copies.
            weight.assign(np.asarray(torch_weight))
        else:
            transfer_weights(weight.path, weight, torch_weight)
