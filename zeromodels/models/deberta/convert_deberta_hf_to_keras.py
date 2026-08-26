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
        r"blocks_(\d+)_attention_self/blocks_\d+_(in_proj|pos_proj|pos_q_proj)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.self.{proj}.{suffix}"

    m = re.match(r"blocks_(\d+)_attention_self/(q_bias|v_bias)$", path)
    if m:
        idx, b = m.groups()
        return f"encoder.layer.{idx}.attention.self.{b}"

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


def transfer_deberta_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {k.removeprefix("deberta."): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_WEIGHTS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


# Architecture per variant, moved here from deberta_config.py: the package config no longer
# carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer the HF weights into.
DEBERTA_MODEL_CONFIG = {
    "deberta_base": {
        "vocab_size": 50265,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "pos_att_type": ["c2p", "p2c"],
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_large": {
        "vocab_size": 50265,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "pos_att_type": ["c2p", "p2c"],
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
    from transformers import DebertaModel as HFDebertaModel

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.deberta import DebertaMaskedLM, DebertaModel

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_SOURCES = {
        "deberta_base": "microsoft/deberta-base",
        "deberta_large": "microsoft/deberta-large",
    }

    def raw_state_dict(hf_id):
        try:
            from safetensors.torch import load_file

            return load_file(
                hf_hub_download(hf_id, "model.safetensors", token=HF_TOKEN)
            )
        except Exception:
            path = hf_hub_download(hf_id, "pytorch_model.bin", token=HF_TOKEN)
            return torch.load(path, map_location="cpu", weights_only=True)

    rng = np.random.default_rng(0)

    for variant, arch in DEBERTA_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
        eps = arch["layer_norm_eps"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(5, arch["vocab_size"], (2, 16)).astype("int64")
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "token_type_ids": np.zeros((2, 16), dtype="int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }
        valid = mask[..., None].astype(bool)
        sd = raw_state_dict(hf_id)

        hf_model = HFDebertaModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        with torch.no_grad():
            seq_ref = hf_model(**pt).last_hidden_state
            h = seq_ref @ sd["lm_predictions.lm_head.dense.weight"].T
            h = h + sd["lm_predictions.lm_head.dense.bias"]
            h = F.gelu(h)
            h = F.layer_norm(
                h,
                (arch["embed_dim"],),
                sd["lm_predictions.lm_head.LayerNorm.weight"],
                sd["lm_predictions.lm_head.LayerNorm.bias"],
                eps=eps,
            )
            mlm_ref = (
                h @ sd["deberta.embeddings.word_embeddings.weight"].T
                + sd["lm_predictions.lm_head.bias"]
            ).numpy()
        ref_np = seq_ref.numpy()

        # One superset checkpoint = encoder + MLM head, saved from DebertaMaskedLM.
        # DebertaModel (no pooler) and the task heads load their subset out of it.
        keras_full = DebertaMaskedLM(**arch)
        transfer_deberta_weights(keras_full, sd)
        kl = keras_full(k_inputs, training=False)
        kl = kl.detach().cpu().numpy() if hasattr(kl, "detach") else np.asarray(kl)
        d_mlm = float(np.abs((mlm_ref - kl) * valid).max())
        print(f"  full checkpoint  mlm max diff (non-pad): {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(f"{variant}: DebertaMaskedLM parity failed")
        keras_full.save_weights(f"{variant}.weights.h5")
        print(f"  Saved single file -> {variant}.weights.h5")

        # Verify the ONE file serves both views: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check parity end to end.
        ref = DebertaMaskedLM(**arch)
        ref.load_weights(f"{variant}.weights.h5")
        bm2 = DebertaModel(**arch)
        copy_weights_by_path_suffix(ref, bm2)
        seq = bm2(k_inputs, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        d_seq = float(np.abs((ref_np - seq) * valid).max())
        mlm2 = DebertaMaskedLM(**arch)
        copy_weights_by_path_suffix(ref, mlm2)
        kl2 = mlm2(k_inputs, training=False)
        kl2 = kl2.detach().cpu().numpy() if hasattr(kl2, "detach") else np.asarray(kl2)
        d_mlm2 = float(np.abs((mlm_ref - kl2) * valid).max())
        print(f"  reload   DebertaModel: {d_seq:.3e}  MaskedLM: {d_mlm2:.3e}")
        if max(d_seq, d_mlm2) > 1e-3:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_model, keras_full, ref, bm2, mlm2, sd
        keras.backend.clear_session()
        gc.collect()
