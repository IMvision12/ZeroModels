import collections.abc
import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate_weight": "mlp.gate.weight",
    "kernel": "weight",
}

EXPERT_PATTERN = re.compile(
    r"^(model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
)
GATE_UP_SUFFIX = ".mlp.experts.gate_up_proj"
DOWN_SUFFIX = ".mlp.experts.down_proj"


class LazyFusedExperts(collections.abc.Mapping):
    """Fuse Mixtral per-expert ``w1``/``w2``/``w3`` into the model's banks lazily.

    A drop-in for the eager fused ``{name: array}`` dict, but each layer's fused
    ``gate_up_proj`` ``(E, 2I, H)`` / ``down_proj`` ``(E, H, I)`` bank is built
    only when the converter looks it up, and freed as soon as it is assigned (or
    quantized into int storage). So peak host memory is ~one layer's experts, not
    every layer's fused banks at once (the difference between fitting an 8x7B on
    a small box and not). Non-expert keys pass through unchanged (router ``gate``
    renamed to match the keras path). Reads one source tensor at a time, so it
    composes with the lazy / streaming ``hf:`` state dict.
    """

    def __init__(self, hf_state_dict):
        self._src = hf_state_dict
        self._experts = {}  # (layer, which) -> {expert_index: source_key}
        self._passthrough = {}  # logical_name -> source_key
        for key in hf_state_dict:
            match = EXPERT_PATTERN.match(key)
            if match:
                layer, expert, which = (
                    match.group(1),
                    int(match.group(2)),
                    match.group(3),
                )
                self._experts.setdefault((layer, which), {})[expert] = key
            elif ".block_sparse_moe.gate." in key:
                self._passthrough[
                    key.replace(".block_sparse_moe.gate.", ".mlp.gate.")
                ] = key
            else:
                self._passthrough[key] = key
        self._fused = set()
        for layer, _which in self._experts:
            self._fused.add(f"{layer}{GATE_UP_SUFFIX}")
            self._fused.add(f"{layer}{DOWN_SUFFIX}")

    def _fuse_gate_up(self, layer):
        w1, w3 = self._experts[(layer, "w1")], self._experts[(layer, "w3")]
        return np.stack(
            [
                np.concatenate(
                    [np.asarray(self._src[w1[e]]), np.asarray(self._src[w3[e]])], axis=0
                )
                for e in sorted(w1)
            ],
            axis=0,
        )  # (E, 2I, H)

    def _fuse_down(self, layer):
        w2 = self._experts[(layer, "w2")]
        return np.stack([np.asarray(self._src[w2[e]]) for e in sorted(w2)], axis=0)

    def __getitem__(self, name):
        if name.endswith(GATE_UP_SUFFIX):
            return self._fuse_gate_up(name[: -len(GATE_UP_SUFFIX)])
        if name.endswith(DOWN_SUFFIX):
            return self._fuse_down(name[: -len(DOWN_SUFFIX)])
        return self._src[self._passthrough[name]]

    def __contains__(self, name):
        return name in self._fused or name in self._passthrough

    def __iter__(self):
        return iter(list(self._passthrough) + sorted(self._fused))

    def __len__(self):
        return len(self._passthrough) + len(self._fused)


def transfer_mixtral_weights(keras_model, hf_state_dict):
    state = LazyFusedExperts(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            weight.assign(np.asarray(state[name]))
        elif name.endswith("mlp.gate.weight"):
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
