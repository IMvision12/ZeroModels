import collections.abc
import re

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.deepseek_v3.convert_deepseek_v3_hf_to_keras import (
    WEIGHT_NAME_MAPPING as TEXT_NAME_MAPPING,
)

# The published checkpoints keep MoonViT's original names (fused ``wqkv``,
# ``norm0``/``norm1``, ``mlp.fc0``/``fc1``), not the split, renamed ones the
# modeling code uses -- transformers rewrites them at load time. These map the
# keras paths onto the *raw* names; ordering matters (norm1 before norm2).
VISION_NAME_MAPPING = {
    "final_norm": "encoder.final_layernorm",
    "patch_proj": "patch_embed.proj",
    "pos_emb": "patch_embed.pos_emb.weight",
    "block_": "encoder.blocks.",
    "norm1": "norm0",
    "norm2": "norm1",
    "mlp.fc1": "mlp.fc0",
    "mlp.fc2": "mlp.fc1",
    "attn.proj": "wo",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

PROJECTOR_NAME_MAPPING = {
    "pre_norm.gamma": "pre_norm.weight",
    "pre_norm.beta": "pre_norm.bias",
    "in_proj.kernel": "proj.0.weight",
    "in_proj.bias": "proj.0.bias",
    "out_proj.kernel": "proj.2.weight",
    "out_proj.bias": "proj.2.bias",
}

FUSED_EXPERT_PATTERN = re.compile(
    r"^(language_model\.model\.layers\.\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
EXPERT_PATTERN = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.experts\.\d+\.")
VISION_QKV_PATTERN = re.compile(
    r"^(vision_tower\.encoder\.blocks\.\d+)\.attn\.([qkv])_proj\.(weight|bias)$"
)
WQKV_PATTERN = re.compile(
    r"^vision_tower\.encoder\.blocks\.(\d+)\.wqkv\.(weight|bias)$"
)

PACK_FACTOR = 8  # int4 values per int32 word
GROUP_SIZE = 32  # compressed-tensors "group" strategy, group_size 32


def as_numpy(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if "bfloat16" in str(value.dtype) or "float8" in str(value.dtype):
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None and array.dtype != np.dtype(dtype):
        array = array.astype(dtype)
    return array


def unpack_int4(packed, scale, shape):
    """Dequantize one compressed-tensors ``pack-quantized`` int4 weight.

    ``packed`` holds ``PACK_FACTOR`` unsigned 4-bit values per int32, densely and
    LSB-first along the input axis (``packed_dim=1``): element ``e`` of each
    32-element group lives in word ``e // 8`` at bit ``4 * (e % 8)``. Values are
    stored offset by ``+8``; the quantization is symmetric ``group`` strategy, so
    there is no zero point and one scale covers each ``GROUP_SIZE`` inputs.
    """
    rows, cols = int(shape[0]), int(shape[1])
    words = as_numpy(packed).view(np.uint32).reshape(rows, cols // 32, 4)
    shifts = 4 * np.arange(PACK_FACTOR, dtype=np.uint32)
    nibbles = (words[..., None] >> shifts) & np.uint32(0xF)
    quantized = nibbles.reshape(rows, cols).astype("float32") - 8.0
    scale = as_numpy(scale, "float32")
    return quantized * np.repeat(scale, GROUP_SIZE, axis=1)[:, :cols]


def permute_for_rope(tensor, num_heads):
    """Interleaved-pair rope layout -> split-half (``rotate_half``) layout.

    MoonViT rotates adjacent pairs ``(x[2i], x[2i+1])``; the vision tower here
    mirrors transformers and rotates halves instead. Reordering each head's rows
    from ``[p0a, p0b, p1a, p1b, ...]`` to ``[p0a, p1a, ..., p0b, p1b, ...]`` makes
    the two identical. Applies to the ``q`` and ``k`` projections only -- both
    their kernels and their biases -- never to ``v``.
    """
    dim = tensor.shape[0]
    half = dim // num_heads // 2
    if tensor.ndim == 2:
        reordered = tensor.reshape(num_heads, half, 2, tensor.shape[1])
        return reordered.transpose(0, 2, 1, 3).reshape(dim, -1)
    reordered = tensor.reshape(num_heads, half, 2)
    return reordered.transpose(0, 2, 1).reshape(dim)


def hf_name_for(path):
    if path.startswith("vision_tower."):
        name = path[len("vision_tower.") :]
        for old, new in VISION_NAME_MAPPING.items():
            name = name.replace(old, new)
        return f"vision_tower.{name}"
    if path.startswith("mm_projector."):
        suffix = PROJECTOR_NAME_MAPPING.get(path[len("mm_projector.") :])
        return f"mm_projector.{suffix}" if suffix else path
    name = path
    for old, new in TEXT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return f"language_model.{name}"


class KimiK25State(collections.abc.Mapping):
    """Lazy view that reshapes the raw checkpoint into what the keras model wants.

    Three derivations happen on lookup rather than up front, so a trillion-
    parameter checkpoint is never materialized at once (only the tensor a single
    keras weight needs):

    - routed experts are int4 ``pack-quantized`` and stored one Linear per
      expert; they are dequantized and fused into the ``(E, 2I, H)`` /
      ``(E, H, I)`` banks the einsum experts layer uses,
    - the vision blocks' fused ``wqkv`` is split into ``q``/``k``/``v``, with
      ``q`` and ``k`` permuted into the ``rotate_half`` rope layout,
    - everything else passes straight through.
    """

    def __init__(self, state, num_experts, vision_num_heads):
        self.state = state
        self.num_experts = num_experts
        self.vision_num_heads = vision_num_heads

    def expert_weight(self, layer, expert, which):
        prefix = f"{layer}.mlp.experts.{expert}.{which}"
        if f"{prefix}.weight" in self.state:
            return as_numpy(self.state[f"{prefix}.weight"], "float32")
        return unpack_int4(
            self.state[f"{prefix}.weight_packed"],
            self.state[f"{prefix}.weight_scale"],
            as_numpy(self.state[f"{prefix}.weight_shape"]),
        )

    def fuse_experts(self, layer, which):
        first = self.expert_weight(
            layer, 0, "down_proj" if "down" in which else "gate_proj"
        )
        rows, cols = first.shape
        if which == "down_proj":
            bank = np.empty((self.num_experts, rows, cols), dtype="float32")
            bank[0] = first
            for expert in range(1, self.num_experts):
                bank[expert] = self.expert_weight(layer, expert, "down_proj")
            return bank
        bank = np.empty((self.num_experts, 2 * rows, cols), dtype="float32")
        bank[0, :rows] = first
        bank[0, rows:] = self.expert_weight(layer, 0, "up_proj")
        for expert in range(1, self.num_experts):
            bank[expert, :rows] = self.expert_weight(layer, expert, "gate_proj")
            bank[expert, rows:] = self.expert_weight(layer, expert, "up_proj")
        return bank

    def vision_qkv(self, block, which, suffix):
        fused = as_numpy(self.state[f"{block}.wqkv.{suffix}"], "float32")
        part = dict(zip("qkv", np.split(fused, 3, axis=0)))[which]
        if which == "v":
            return part
        return permute_for_rope(part, self.vision_num_heads)

    def derived_keys(self):
        keys = []
        for key in self.state:
            match = WQKV_PATTERN.match(key)
            if match:
                block = f"vision_tower.encoder.blocks.{match.group(1)}"
                keys += [f"{block}.attn.{p}_proj.{match.group(2)}" for p in "qkv"]
                continue
            match = re.match(
                r"^(language_model\.model\.layers\.\d+)\.mlp\.experts\.0"
                r"\.gate_proj\.(weight|weight_packed)$",
                key,
            )
            if match:
                keys += [
                    f"{match.group(1)}.mlp.experts.gate_up_proj",
                    f"{match.group(1)}.mlp.experts.down_proj",
                ]
        return keys

    def __getitem__(self, key):
        match = FUSED_EXPERT_PATTERN.match(key)
        if match:
            return self.fuse_experts(match.group(1), match.group(2))
        match = VISION_QKV_PATTERN.match(key)
        if match:
            return self.vision_qkv(match.group(1), match.group(2), match.group(3))
        return self.state[key]

    def __contains__(self, key):
        match = FUSED_EXPERT_PATTERN.match(key)
        if match:
            which = "down_proj" if match.group(2) == "down_proj" else "gate_proj"
            prefix = f"{match.group(1)}.mlp.experts.0.{which}"
            return (
                f"{prefix}.weight" in self.state
                or f"{prefix}.weight_packed" in self.state
            )
        match = VISION_QKV_PATTERN.match(key)
        if match:
            return f"{match.group(1)}.wqkv.{match.group(3)}" in self.state
        return key in self.state

    def __iter__(self):
        for key in self.state:
            if EXPERT_PATTERN.match(key) or WQKV_PATTERN.match(key):
                continue
            yield key
        yield from self.derived_keys()

    def __len__(self):
        return sum(1 for _ in self)


def transfer_kimi_k25_weights(keras_model, hf_state_dict):
    state = KimiK25State(
        hf_state_dict, keras_model.num_experts, keras_model.vision_num_heads
    )
    if not keras_model.built or not keras_model.weights:
        keras_model.build_for_transfer()
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        path = weight.path.replace("/", ".")
        name = hf_name_for(path)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        value = state[name]
        if path == "vision_tower.patch_proj.kernel":
            # (embed_dim, C, P, P) conv kernel -> (C * P * P, embed_dim) dense
            kernel = as_numpy(value, "float32")
            weight.assign(kernel.reshape(kernel.shape[0], -1).T)
        elif (
            path == "vision_tower.pos_emb"
            or ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
            or name.endswith("e_score_correction_bias")
        ):
            weight.assign(as_numpy(value, "float32"))
        else:
            transfer_weights(weight.path, weight, value)
