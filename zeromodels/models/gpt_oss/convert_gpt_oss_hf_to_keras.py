import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "kernel": "weight",
}

FP4_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def dequantize_mxfp4(blocks, scales):
    """Dequantize GPT-OSS MXFP4 packed expert tensors to float32.

    Port of Hugging Face's ``convert_moe_packed_tensors``: each uint8 byte holds
    two 4-bit indices into ``FP4_VALUES`` (low nibble first), and every 32-value
    block (16 bytes) shares an e8m0 power-of-two scale. Output is transposed to
    the dequantized ``(E, H, 2I)`` / ``(E, I, H)`` layout used by the experts.
    """
    blocks = np.asarray(blocks).astype(np.uint8)
    scales = np.asarray(scales).astype(np.int32) - 127
    *prefix, g, b = blocks.shape
    rows = int(np.prod(prefix)) * g
    blk = blocks.reshape(rows, b)
    exp = scales.reshape(rows, 1).astype(np.float32)
    out = np.empty((rows, b * 2), dtype=np.float32)
    out[:, 0::2] = FP4_VALUES[blk & 0x0F]
    out[:, 1::2] = FP4_VALUES[blk >> 4]
    out = out * np.exp2(exp)
    out = out.reshape(*prefix, g * b * 2)
    return np.ascontiguousarray(np.swapaxes(out, 1, 2))


def hf_name_for(path):
    # Functional model weight paths are flat (no model-name root to strip).
    name = path.replace("/", ".")
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def transfer_gpt_oss_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = hf_name_for(weight.path)
        if (name.endswith("_blocks") or name.endswith("_scales")) and "experts" in name:
            # MXFP4 expert matrix, kept packed: copy the uint8 nibble blocks /
            # e8m0 scales straight across (shapes match the checkpoint). The
            # GptOssMXFP4Experts layer dequantizes them on the fly at run time.
            if name not in hf_state_dict:
                raise WeightMappingError(weight.path, name)
            weight.assign(np.asarray(hf_state_dict[name]))
        elif name.endswith(".gate_up_proj") or name.endswith(".down_proj"):
            # Dequantized (bf16) expert matrix: plain bf16 repo, or MXFP4 blocks
            # expanded to float when the model was built without mxfp4_experts.
            if name in hf_state_dict:
                weight.assign(np.asarray(hf_state_dict[name]))
            elif f"{name}_blocks" in hf_state_dict:
                weight.assign(
                    dequantize_mxfp4(
                        hf_state_dict[f"{name}_blocks"], hf_state_dict[f"{name}_scales"]
                    )
                )
            else:
                raise WeightMappingError(weight.path, name)
        elif name.endswith("_bias") and "experts" in name:
            # Per-expert bias (E, 2I) / (E, H): direct copy, no Dense transpose.
            if name not in hf_state_dict:
                raise WeightMappingError(weight.path, name)
            weight.assign(np.asarray(hf_state_dict[name]))
        else:
            if name not in hf_state_dict:
                raise WeightMappingError(weight.path, name)
            transfer_weights(weight.path, weight, hf_state_dict[name])
