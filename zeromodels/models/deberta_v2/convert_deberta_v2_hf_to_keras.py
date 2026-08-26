import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "rel_embeddings/embeddings": "encoder.rel_embeddings.weight",
    "rel_embeddings_layernorm/gamma": "encoder.LayerNorm.weight",
    "rel_embeddings_layernorm/beta": "encoder.LayerNorm.bias",
    "conv/conv/kernel": "encoder.conv.conv.weight",
    "conv/conv/bias": "encoder.conv.conv.bias",
    "conv/LayerNorm/gamma": "encoder.conv.LayerNorm.weight",
    "conv/LayerNorm/beta": "encoder.conv.LayerNorm.bias",
    "lm_head_dense/kernel": "lm_predictions.lm_head.dense.weight",
    "lm_head_dense/bias": "lm_predictions.lm_head.dense.bias",
    "lm_head_layernorm/gamma": "lm_predictions.lm_head.LayerNorm.weight",
    "lm_head_layernorm/beta": "lm_predictions.lm_head.LayerNorm.bias",
    "lm_head_decoder/kernel": "embeddings.word_embeddings.weight",
    "lm_head_decoder/bias": "lm_predictions.lm_head.bias",
    "pooler_dense/kernel": "pooler.dense.weight",
    "pooler_dense/bias": "pooler.dense.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
}

_OPTIONAL_WEIGHTS = ("classifier", "qa_outputs", "lm_head", "pooler_dense")

_LAYER_MAP = {
    "attention_output_dense": "attention.output.dense",
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_output_layernorm": "attention.output.LayerNorm",
    "output_layernorm": "output.LayerNorm",
}


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention_self/blocks_\d+_(query_proj|key_proj|value_proj|pos_key_proj|pos_query_proj)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.self.{proj}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_dense|intermediate_dense|output_dense)/(kernel|bias)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.{_LAYER_MAP[layer]}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_layernorm|output_layernorm)/(gamma|beta)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{_LAYER_MAP[layer]}.{suffix}"

    return None


def transfer_deberta_v2_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    hf = {k.removeprefix("deberta."): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_WEIGHTS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        value = hf[hf_name]
        if weight.path == "conv/conv/kernel":
            value = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )
            weight.assign(np.transpose(value, (2, 1, 0)))
            continue
        transfer_weights(weight.path, weight, value)


# Architecture per variant, moved here from deberta_v2_config.py: the package config no longer
# carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer the HF weights into.
DEBERTA_V2_MODEL_CONFIG = {
    "deberta_v2_xlarge": {
        "vocab_size": 128100,
        "embed_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "mlp_dim": 6144,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 3,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_v2_xxlarge": {
        "vocab_size": 128100,
        "embed_dim": 1536,
        "num_layers": 48,
        "num_heads": 24,
        "mlp_dim": 6144,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 3,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
}


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import DebertaV2Model as HFDebertaV2Model

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.deberta_v2 import DebertaV2MaskedLM, DebertaV2Model

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_SOURCES = {
        "deberta_v2_xlarge": "microsoft/deberta-v2-xlarge",
        "deberta_v2_xxlarge": "microsoft/deberta-v2-xxlarge",
    }
    MAX_SHARD_GB = 1.7

    def raw_state_dict(hf_id):
        try:
            from safetensors.torch import load_file

            return load_file(
                hf_hub_download(hf_id, "model.safetensors", token=HF_TOKEN)
            )
        except Exception:
            path = hf_hub_download(hf_id, "pytorch_model.bin", token=HF_TOKEN)
            return torch.load(path, map_location="cpu", weights_only=True)

    def cosine(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rng = np.random.default_rng(0)

    for variant, arch in DEBERTA_V2_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
        eps = arch["layer_norm_eps"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(5, arch["vocab_size"], (2, 16)).astype("int64")
        mask = np.ones((2, 16), dtype="int64")
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "token_type_ids": np.zeros((2, 16), dtype="int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }
        sd = raw_state_dict(hf_id)

        hf_model = (
            HFDebertaV2Model.from_pretrained(hf_id, token=HF_TOKEN).float().eval()
        )
        print(f"  HF reference dtype: {hf_model.dtype}  (must be float32)")
        with torch.no_grad():
            seq_ref = hf_model(**pt).last_hidden_state
            dense_w = sd["lm_predictions.lm_head.dense.weight"].float()
            dense_b = sd["lm_predictions.lm_head.dense.bias"].float()
            ln_w = sd["lm_predictions.lm_head.LayerNorm.weight"].float()
            ln_b = sd["lm_predictions.lm_head.LayerNorm.bias"].float()
            dec_b = sd["lm_predictions.lm_head.bias"].float()
            word_emb = sd["deberta.embeddings.word_embeddings.weight"].float()
            h = seq_ref @ dense_w.T + dense_b
            h = F.gelu(h)
            h = F.layer_norm(h, (arch["embed_dim"],), ln_w, ln_b, eps=eps)
            mlm_ref = (h @ word_emb.T + dec_b).numpy()
        ref_np = seq_ref.numpy()

        # One superset checkpoint = encoder + MLM head, saved from DebertaV2MaskedLM.
        # DebertaV2Model (no pooler) and the task heads load their subset out of it.
        keras_full = DebertaV2MaskedLM(**arch)
        transfer_deberta_v2_weights(keras_full, sd)
        kl = keras_full(k_inputs, training=False)
        kl = kl.detach().cpu().numpy() if hasattr(kl, "detach") else np.asarray(kl)
        c_mlm = cosine(mlm_ref, kl)
        print(f"  full checkpoint  mlm cosine: {c_mlm:.6f}")
        if c_mlm < 0.999:
            raise ValueError(
                f"{variant}: DebertaV2MaskedLM parity failed (cosine {c_mlm:.4f})"
            )
        keras_full.save_weights(f"{variant}.weights.json", max_shard_size=MAX_SHARD_GB)
        print(f"  Saved single file -> {variant}.weights.json")

        # Verify the ONE file serves both views: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check parity end to end.
        ref = DebertaV2MaskedLM(**arch)
        ref.load_weights(f"{variant}.weights.json")
        bm = DebertaV2Model(**arch)
        copy_weights_by_path_suffix(ref, bm)
        seq = bm(k_inputs, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        c_seq = cosine(ref_np, seq)
        mlm2 = DebertaV2MaskedLM(**arch)
        copy_weights_by_path_suffix(ref, mlm2)
        kl2 = mlm2(k_inputs, training=False)
        kl2 = kl2.detach().cpu().numpy() if hasattr(kl2, "detach") else np.asarray(kl2)
        c_mlm2 = cosine(mlm_ref, kl2)
        print(f"  reload   DebertaV2Model cosine: {c_seq:.6f}  MaskedLM: {c_mlm2:.6f}")
        if min(c_seq, c_mlm2) < 0.999:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_model, keras_full, ref, bm, mlm2, sd
        keras.backend.clear_session()
        gc.collect()
