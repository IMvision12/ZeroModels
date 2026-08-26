import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/position_embeddings/embeddings": "embeddings.position_embeddings.weight",
    "embeddings/token_type_embeddings/embeddings": "embeddings.token_type_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "attention_output_dense": "attention.output.dense",
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_output_layernorm": "attention.output.LayerNorm",
    "output_layernorm": "output.LayerNorm",
    "pooler_dense/kernel": "pooler.dense.weight",
    "pooler_dense/bias": "pooler.dense.bias",
    "lm_head_dense/kernel": "lm_head.dense.weight",
    "lm_head_dense/bias": "lm_head.dense.bias",
    "lm_head_layernorm/gamma": "lm_head.layer_norm.weight",
    "lm_head_layernorm/beta": "lm_head.layer_norm.bias",
    # The MLM decoder kernel is tied to the input word embeddings; HF strips the
    # tied `lm_head.decoder.weight` from safetensors, so map to the embedding
    # table (transfer_weights transposes it into the Dense kernel). Mapping to
    # the tied key instead would be silently skipped (lm_head is optional) and
    # leave a random decoder. Mirrors the BERT / DeBERTa converters.
    "lm_head_decoder/kernel": "embeddings.word_embeddings.weight",
    "lm_head_decoder/bias": "lm_head.bias",
    "classifier_dense/kernel": "classifier.dense.weight",
    "classifier_dense/bias": "classifier.dense.bias",
    "classifier_out_proj/kernel": "classifier.out_proj.weight",
    "classifier_out_proj/bias": "classifier.out_proj.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
}

_OPTIONAL_WEIGHTS = ("classifier", "qa_outputs", "lm_head", "pooler_dense")


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention_self/blocks_\d+_(query|key|value)/(kernel|bias)$",
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
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_layernorm|output_layernorm)/(gamma|beta)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("roberta."):
        key = key[len("roberta.") :]
    return key.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
        "LayerNorm.beta", "LayerNorm.bias"
    )


def transfer_roberta_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_WEIGHTS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


# Architecture per variant, moved here from roberta_config.py: the package config no longer
# carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer the HF weights into.
ROBERTA_MODEL_CONFIG = {
    "roberta_base": {
        "vocab_size": 50265,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 514,
        "type_vocab_size": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
    },
    "roberta_large": {
        "vocab_size": 50265,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 514,
        "type_vocab_size": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
    },
}


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import RobertaForMaskedLM
    from transformers import RobertaModel as HFRobertaModel

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.roberta import RobertaMaskedLM, RobertaModel

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "roberta_base": "FacebookAI/roberta-base",
        "roberta_large": "FacebookAI/roberta-large",
    }

    rng = np.random.default_rng(0)

    for variant, arch in ROBERTA_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
        pad = arch["pad_token_id"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(3, arch["vocab_size"], (2, 16)).astype("int64")
        ids[:, 0] = 0
        ids[0, 12:] = pad
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        types = np.zeros((2, 16), dtype="int64")
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "token_type_ids": types.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }

        hf_model = HFRobertaModel.from_pretrained(
            hf_id, token=HF_TOKEN
        ).eval()  # + pooler
        hf_mlm = RobertaForMaskedLM.from_pretrained(
            hf_id, token=HF_TOKEN
        ).eval()  # + MLM head

        # One superset checkpoint = encoder + pooler + MLM head. HF splits these across
        # RobertaModel (pooler) and RobertaForMaskedLM (MLM head), so merge both state
        # dicts. The tied MLM decoder kernel is stripped from safetensors -- drop it so the
        # converter reconstructs it from the word embeddings (mirrors the `hf:` path). Each
        # zeromodels class then loads its own subset out of this single file.
        merged = {**dict(hf_mlm.state_dict()), **dict(hf_model.state_dict())}
        merged.pop("lm_head.decoder.weight", None)
        keras_full = RobertaMaskedLM(**arch, add_pooler=True)
        transfer_roberta_weights(keras_full, merged)

        with torch.no_grad():
            hf_out = hf_model(**pt)
            hf_logits = hf_mlm(**pt).logits
        full_out = keras_full(k_inputs, training=False)
        d_mlm = float(
            np.abs(
                hf_logits.detach().cpu().numpy()
                - full_out["logits"].detach().cpu().numpy()
            ).max()
        )
        d_pool = float(
            np.abs(
                hf_out.pooler_output.detach().cpu().numpy()
                - full_out["pooler_output"].detach().cpu().numpy()
            ).max()
        )
        print(f"  full checkpoint   mlm diff: {d_mlm:.3e}   pooler diff: {d_pool:.3e}")
        if max(d_mlm, d_pool) > 1e-3:
            raise ValueError(f"{variant}: full-checkpoint parity failed")

        out_path = f"{variant}.weights.h5"
        keras_full.save_weights(out_path)
        print(f"  Saved single file -> {out_path}")

        # Verify the ONE file serves every view: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check parity end to end.
        ref = RobertaMaskedLM(**arch, add_pooler=True)
        ref.load_weights(out_path)
        rm = RobertaModel(**arch)
        copy_weights_by_path_suffix(ref, rm)
        rm_out = rm(k_inputs, training=False)
        d_seq = float(
            np.abs(
                hf_out.last_hidden_state.detach().cpu().numpy()
                - rm_out["last_hidden_state"].detach().cpu().numpy()
            ).max()
        )
        d_rmpool = float(
            np.abs(
                hf_out.pooler_output.detach().cpu().numpy()
                - rm_out["pooler_output"].detach().cpu().numpy()
            ).max()
        )
        mlm2 = RobertaMaskedLM(**arch)
        copy_weights_by_path_suffix(ref, mlm2)
        d_mlm2 = float(
            np.abs(
                hf_logits.detach().cpu().numpy()
                - mlm2(k_inputs, training=False).detach().cpu().numpy()
            ).max()
        )
        print(
            f"  reload   RobertaModel seq: {d_seq:.3e}  pooler: {d_rmpool:.3e}  |  "
            f"RobertaMaskedLM mlm: {d_mlm2:.3e}"
        )
        if max(d_seq, d_rmpool, d_mlm2) > 1e-3:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_model, hf_mlm, keras_full, ref, rm, mlm2
        keras.backend.clear_session()
        gc.collect()
