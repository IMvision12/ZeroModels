import re

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError

E2M1_VALUES = np.array(
    (
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
    ),
    dtype=np.float32,
)
_BYTE_IDX = np.arange(256, dtype=np.int32)
BYTE_TO_E2M1_PAIR = np.stack(
    (E2M1_VALUES[_BYTE_IDX & 0xF], E2M1_VALUES[_BYTE_IDX >> 4]), axis=-1
)  # (256, 2) float32


def deinterleave_dim1(arr):
    """Deinterleave a 3D expert tensor ``(E, 2I, H)`` along axis 1.

    Matches HF ``Interleave(dim=1)``: ``reshape(E, I, 2, H).transpose(0, 2, 1, 3).reshape(E, 2I, H)``.
    """
    e, d1, h = arr.shape
    return (
        arr.reshape(e, d1 // 2, 2, h)
        .transpose(0, 2, 1, 3)
        .reshape(e, d1, h)
        .copy()
    )


def dequantize_inkling2_experts(
    weight,
    scale=None,
    amax=None,
    outer_scale_denominator=None,
    out_dtype=np.float32,
    experts_per_chunk=8,
):
    """Dequantize NVFP4 or MXFP8 Inkling-2-Mini expert weights in pure NumPy.

    - Unquantized (``scale is None``): returns ``weight`` cast to ``out_dtype``.
    - NVFP4 (``amax is not None``): two E2M1 nibbles per uint8 byte (low nibble first),
      one ``float8_e4m3fn`` scale per 16 values, multiplied by ``amax / outer_scale_denominator``.
    - MXFP8 (``scale is not None`` and ``amax is None``): ``float8_e4m3fn`` values
      with one ``ue8m0`` power-of-two scale ``2 ** (scale - 127)`` per 32 values.
    """
    if scale is None:
        return np.asarray(weight, dtype=out_dtype)

    weight = np.asarray(weight)
    scale = np.asarray(scale)
    num_experts, rows = weight.shape[:2]
    cols = weight.shape[2] * 2 if amax is not None else weight.shape[2]
    output = np.empty((num_experts, rows, cols), dtype=out_dtype)

    if amax is not None:
        expert_factor = np.asarray(amax, dtype=np.float32) / np.asarray(
            outer_scale_denominator, dtype=np.float32
        )

    for start in range(0, num_experts, experts_per_chunk):
        sl = slice(start, min(start + experts_per_chunk, num_experts))
        block_scale = scale[sl].astype(np.float32)
        if amax is None:
            vals = weight[sl].astype(np.float32).reshape(*block_scale.shape, 32)
            vals = vals * np.exp2(block_scale - 127.0)[..., None]
        else:
            codes = weight[sl].astype(np.uint8)
            vals = BYTE_TO_E2M1_PAIR[codes].reshape(*block_scale.shape, 16)
            vals = vals * block_scale[..., None]
            vals = vals * expert_factor[sl, None, None, None]
        output[sl] = vals.reshape(-1, rows, cols).astype(out_dtype)

    return output


def merge_message_queries(queries, kv_queries):
    """Merge ``(2*L, H)`` [q, mlp] and ``(2*L, H)`` [k, v] into ``(L, 4, H)`` [q, k, v, mlp]."""
    q = np.asarray(queries, dtype=np.float32)
    kv = np.asarray(kv_queries, dtype=np.float32)
    num_loop = q.shape[0] // 2
    q = q.reshape(num_loop, 2, -1)
    kv = kv.reshape(num_loop, 2, -1)
    return np.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)


def _target_numpy_dtype(weight):
    dtype_str = str(getattr(weight, "dtype", "float32"))
    if "bfloat16" in dtype_str:
        try:
            import ml_dtypes

            return ml_dtypes.bfloat16
        except ImportError:
            return np.float32
    if "float16" in dtype_str:
        return np.float16
    return np.float32


def _resolve_key(hf_state_dict, *candidates):
    for key in candidates:
        if key in hf_state_dict:
            return key
    return None


def _assign_tensor(weight, arr, transpose_2d=False):
    arr = np.asarray(arr)
    if transpose_2d and arr.ndim == 2:
        arr = arr.T
    target_dtype = _target_numpy_dtype(weight)
    weight.assign(arr.astype(target_dtype, copy=False))


def _map_vision_weight(subpath, hf_state_dict):
    """Map a relative ``vision_tower/...`` weight path to its HF tensor + transpose flag."""
    direct_map = {
        "patch_embedding/kernel": (
            "model.visual.backbone.embeddings.patch_embedding.weight",
            True,
        ),
        "position_embedding": (
            "model.visual.backbone.embeddings.position_embedding",
            False,
        ),
        "class_embedding": (
            "model.visual.backbone.embeddings.class_embedding",
            False,
        ),
        "cls_position_embedding": (
            "model.visual.backbone.embeddings.cls_position_embedding",
            False,
        ),
        "pre_layernorm/gamma": ("model.visual.backbone.ln_pre.weight", False),
        "pre_layernorm/beta": ("model.visual.backbone.ln_pre.bias", False),
        "merger/linear_1/kernel": ("model.visual.proj_in.weight", True),
        "merger/linear_1/bias": ("model.visual.proj_in.bias", False),
        "merger/linear_2/kernel": ("model.visual.proj_out.weight", True),
        "merger/linear_2/bias": ("model.visual.proj_out.bias", False),
    }
    if subpath in direct_map:
        hf_key, do_t = direct_map[subpath]
        if hf_key in hf_state_dict:
            return hf_state_dict[hf_key], do_t
        raise WeightMappingError(f"vision_tower/{subpath}", hf_key)

    m = re.match(r"^layers_(\d+)/(.+)$", subpath)
    if not m:
        raise WeightMappingError(f"vision_tower/{subpath}", subpath)
    idx, rest = m.group(1), m.group(2)
    base = f"model.visual.backbone.layers.{idx}"
    layer_map = {
        "layernorm_before/gamma": (f"{base}.ln_1.weight", False),
        "layernorm_before/beta": (f"{base}.ln_1.bias", False),
        "layernorm_after/gamma": (f"{base}.ln_2.weight", False),
        "layernorm_after/beta": (f"{base}.ln_2.bias", False),
        "lambda_1": (f"{base}.ls_1.gamma", False),
        "lambda_2": (f"{base}.ls_2.gamma", False),
        "self_attn/q_proj/kernel": (f"{base}.self_attn.q_proj.weight", True),
        "self_attn/q_proj/bias": (f"{base}.self_attn.q_proj.bias", False),
        "self_attn/k_proj/kernel": (f"{base}.self_attn.k_proj.weight", True),
        "self_attn/k_proj/bias": (f"{base}.self_attn.k_proj.bias", False),
        "self_attn/v_proj/kernel": (f"{base}.self_attn.v_proj.weight", True),
        "self_attn/v_proj/bias": (f"{base}.self_attn.v_proj.bias", False),
        "self_attn/out_proj/kernel": (f"{base}.self_attn.out_proj.weight", True),
        "self_attn/out_proj/bias": (f"{base}.self_attn.out_proj.bias", False),
        "mlp/fc1/kernel": (f"{base}.mlp.c_fc.weight", True),
        "mlp/fc1/bias": (f"{base}.mlp.c_fc.bias", False),
        "mlp/fc2/kernel": (f"{base}.mlp.c_proj.weight", True),
        "mlp/fc2/bias": (f"{base}.mlp.c_proj.bias", False),
    }
    if rest not in layer_map:
        raise WeightMappingError(f"vision_tower/{subpath}", rest)
    hf_key, do_t = layer_map[rest]
    if hf_key not in hf_state_dict:
        raise WeightMappingError(f"vision_tower/{subpath}", hf_key)
    return hf_state_dict[hf_key], do_t


def _map_decoder_layer_weight(layer_idx, rest, weight, hf_state_dict):
    base = f"model.llm.layers.{layer_idx}"

    # Attention residual queries (attn_queries is offset by -1 because layer 0 has none)
    if rest == "attention_input/query":
        hf_key = f"model.llm.attn_res.attn_queries.{layer_idx - 1}"
        if hf_key not in hf_state_dict:
            raise WeightMappingError(weight.path, hf_key)
        return hf_state_dict[hf_key], False
    if rest == "mlp_input/query":
        hf_key = f"model.llm.attn_res.mlp_queries.{layer_idx}"
        if hf_key not in hf_state_dict:
            raise WeightMappingError(weight.path, hf_key)
        return hf_state_dict[hf_key], False

    # MoE experts (NVFP4 / MXFP8 dequantize on the fly, one layer at a time)
    if rest == "mlp/experts/gate_up_proj":
        w_key = f"{base}.mlp.experts.w13_weight"
        if w_key not in hf_state_dict:
            raise WeightMappingError(weight.path, w_key)
        s_key = f"{base}.mlp.experts.w13_weight_scale"
        a_key = f"{base}.mlp.experts.w13_weight_amax"
        d_key = f"{base}.mlp.experts.w13_weight_outer_scale_denominator"
        arr = dequantize_inkling2_experts(
            hf_state_dict[w_key],
            scale=hf_state_dict[s_key] if s_key in hf_state_dict else None,
            amax=hf_state_dict[a_key] if a_key in hf_state_dict else None,
            outer_scale_denominator=(
                hf_state_dict[d_key] if d_key in hf_state_dict else None
            ),
            out_dtype=_target_numpy_dtype(weight),
        )
        return deinterleave_dim1(arr), False

    if rest == "mlp/experts/down_proj":
        w_key = f"{base}.mlp.experts.w2_weight"
        if w_key not in hf_state_dict:
            raise WeightMappingError(weight.path, w_key)
        s_key = f"{base}.mlp.experts.w2_weight_scale"
        a_key = f"{base}.mlp.experts.w2_weight_amax"
        d_key = f"{base}.mlp.experts.w2_weight_outer_scale_denominator"
        arr = dequantize_inkling2_experts(
            hf_state_dict[w_key],
            scale=hf_state_dict[s_key] if s_key in hf_state_dict else None,
            amax=hf_state_dict[a_key] if a_key in hf_state_dict else None,
            outer_scale_denominator=(
                hf_state_dict[d_key] if d_key in hf_state_dict else None
            ),
            out_dtype=_target_numpy_dtype(weight),
        )
        return arr, False

    table = {
        "input_layernorm/weight": (f"{base}.attn_norm.weight", False),
        "q_input_layernorm/weight": (f"{base}.q_input_norm.weight", False),
        "k_input_layernorm/weight": (f"{base}.k_input_norm.weight", False),
        "v_input_layernorm/weight": (f"{base}.v_input_norm.weight", False),
        "post_attention_layernorm/weight": (f"{base}.mlp_norm.weight", False),
        "post_attention_convolution/conv_weight": (f"{base}.attn_sconv.weight", False),
        "post_mlp_convolution/conv_weight": (f"{base}.mlp_sconv.weight", False),
        "self_attn/q_linear/kernel": (f"{base}.attn.wq_du.weight", True),
        "self_attn/q_norm/scale": (f"{base}.attn.q_l2_clip.scale", False),
        "self_attn/k_linear/kernel": (f"{base}.attn.wk_dv.weight", True),
        "self_attn/v_linear/kernel": (f"{base}.attn.wv_dv.weight", True),
        "self_attn/k_sconv/conv_weight": (f"{base}.attn.k_sconv.weight", False),
        "self_attn/v_sconv/conv_weight": (f"{base}.attn.v_sconv.weight", False),
        "self_attn/k_sconv/static_weight": (f"{base}.attn.k_sconv.static_w", False),
        "self_attn/k_sconv/weight_proj/kernel": (
            f"{base}.attn.k_sconv.weight_proj",
            True,
        ),
        "self_attn/v_sconv/static_weight": (f"{base}.attn.v_sconv.static_w", False),
        "self_attn/v_sconv/weight_proj/kernel": (
            f"{base}.attn.v_sconv.weight_proj",
            True,
        ),
        "self_attn/gate_proj/kernel": (f"{base}.attn.gate_attn.weight", True),
        "self_attn/o_linear/kernel": (f"{base}.attn.wo_ud.weight", True),
        "self_attn/sink_proj/kernel": (f"{base}.attn.attn_sink_proj.weight", True),
        "self_attn/relative_position_bias/linear/kernel": (
            f"{base}.attn.wr_du.weight",
            True,
        ),
        "self_attn/relative_position_bias/norm/scale": (
            f"{base}.attn.rel_l2_clip.scale",
            False,
        ),
        "self_attn/relative_position_bias/proj": (
            f"{base}.attn.rel_logits_proj.proj",
            False,
        ),
        "mlp/gate_weight": (f"{base}.mlp.gate.weight", False),
        "mlp/e_score_correction_bias": (f"{base}.mlp.gate.bias", False),
        # Loop message gates
        "q_message_gate/down_proj/kernel": (f"{base}.controllers.attn.down_weight", True),
        "q_message_gate/forget_gate_proj/kernel": (
            f"{base}.controllers.attn.forget_weight",
            True,
        ),
        "q_message_gate/message_gate_proj/kernel": (
            f"{base}.controllers.attn.gain_weight",
            True,
        ),
        "k_message_gate/down_proj/kernel": (f"{base}.controllers.k.down_weight", True),
        "k_message_gate/forget_gate_proj/kernel": (
            f"{base}.controllers.k.forget_weight",
            True,
        ),
        "k_message_gate/message_gate_proj/kernel": (
            f"{base}.controllers.k.gain_weight",
            True,
        ),
        "v_message_gate/down_proj/kernel": (f"{base}.controllers.v.down_weight", True),
        "v_message_gate/forget_gate_proj/kernel": (
            f"{base}.controllers.v.forget_weight",
            True,
        ),
        "v_message_gate/message_gate_proj/kernel": (
            f"{base}.controllers.v.gain_weight",
            True,
        ),
        "mlp_message_gate/down_proj/kernel": (
            f"{base}.controllers.mlp.down_weight",
            True,
        ),
        "mlp_message_gate/forget_gate_proj/kernel": (
            f"{base}.controllers.mlp.forget_weight",
            True,
        ),
        "mlp_message_gate/message_gate_proj/kernel": (
            f"{base}.controllers.mlp.gain_weight",
            True,
        ),
    }
    if rest not in table:
        raise WeightMappingError(weight.path, rest)
    hf_key, do_t = table[rest]
    if hf_key not in hf_state_dict:
        raise WeightMappingError(weight.path, hf_key)
    arr = np.asarray(hf_state_dict[hf_key])
    if rest.endswith("conv_weight") and arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    return arr, do_t


def transfer_inkling2_loop_weights(keras_model, hf_state_dict):
    """Stream-convert an Inkling-2-Mini HF state dict (``LazyStateDict``) into Keras.

    Reads each tensor on demand so peak host RAM stays bounded to a single MoE
    expert bank even on the 26.4B checkpoint. Handles NVFP4 (layers 0-19) and
    MXFP8 (layers 20-23) expert dequantization, ``w13`` deinterleaving, unembedding
    unpadding, cross-pass message query merging, and ``attn_res`` index offsetting.
    """
    if not keras_model.built or not keras_model.weights:
        keras_model.build(None)

    for weight in tqdm(keras_model.weights, desc="Transferring Inkling2Loop weights"):
        # Strip top-level model name prefix: e.g. "inkling2_loop_conditional_generate/..."
        parts = weight.path.split("/", 1)
        rel = parts[1] if len(parts) > 1 else parts[0]

        if rel == "lm_head/kernel":
            hf_key = _resolve_key(hf_state_dict, "model.llm.unembed.weight", "lm_head.weight")
            if hf_key is None:
                raise WeightMappingError(weight.path, "model.llm.unembed.weight")
            arr = np.asarray(hf_state_dict[hf_key])[: keras_model.unpadded_vocab_size]
            _assign_tensor(weight, arr, transpose_2d=True)
            continue

        if rel.startswith("vision_tower/"):
            arr, do_t = _map_vision_weight(rel[len("vision_tower/") :], hf_state_dict)
            _assign_tensor(weight, arr, transpose_2d=do_t)
            continue

        if rel.startswith("language_model/"):
            lm_rel = rel[len("language_model/") :]
            if lm_rel == "token_embedding/embeddings":
                hf_key = "model.llm.embed.weight"
                if hf_key not in hf_state_dict:
                    raise WeightMappingError(weight.path, hf_key)
                _assign_tensor(weight, hf_state_dict[hf_key], transpose_2d=False)
                continue
            if lm_rel == "embed_norm/weight":
                hf_key = "model.llm.embed_norm.weight"
                if hf_key not in hf_state_dict:
                    raise WeightMappingError(weight.path, hf_key)
                _assign_tensor(weight, hf_state_dict[hf_key], transpose_2d=False)
                continue
            if lm_rel == "final_norm/weight":
                hf_key = "model.llm.norm.weight"
                if hf_key not in hf_state_dict:
                    raise WeightMappingError(weight.path, hf_key)
                _assign_tensor(weight, hf_state_dict[hf_key], transpose_2d=False)
                continue
            if lm_rel == "final_input/query":
                hf_key = "model.llm.attn_res.output_query"
                if hf_key not in hf_state_dict:
                    raise WeightMappingError(weight.path, hf_key)
                _assign_tensor(weight, hf_state_dict[hf_key], transpose_2d=False)
                continue
            if lm_rel == "message_input/query":
                q_key = "model.llm.middle_cycle_cross_pass_queries"
                kv_key = "model.llm.middle_cycle_cross_pass_kv_queries"
                if q_key not in hf_state_dict or kv_key not in hf_state_dict:
                    raise WeightMappingError(weight.path, f"{q_key} + {kv_key}")
                merged = merge_message_queries(hf_state_dict[q_key], hf_state_dict[kv_key])
                _assign_tensor(weight, merged, transpose_2d=False)
                continue

            m = re.match(r"^decoder_layer_(\d+)/(.+)$", lm_rel)
            if m:
                layer_idx, rest = int(m.group(1)), m.group(2)
                arr, do_t = _map_decoder_layer_weight(
                    layer_idx, rest, weight, hf_state_dict
                )
                _assign_tensor(weight, arr, transpose_2d=do_t)
                continue

        raise WeightMappingError(weight.path, rel)
