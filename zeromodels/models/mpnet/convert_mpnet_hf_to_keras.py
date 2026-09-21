import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/position_embeddings/embeddings": "embeddings.position_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "encoder_relative_attention_bias/embeddings": (
        "encoder.relative_attention_bias.weight"
    ),
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_layernorm": "attention.LayerNorm",
    "output_layernorm": "output.LayerNorm",
    "pooler_dense/kernel": "pooler.dense.weight",
    "pooler_dense/bias": "pooler.dense.bias",
    "lm_head_dense/kernel": "lm_head.dense.weight",
    "lm_head_dense/bias": "lm_head.dense.bias",
    "lm_head_layernorm/gamma": "lm_head.layer_norm.weight",
    "lm_head_layernorm/beta": "lm_head.layer_norm.bias",
    "lm_head_decoder/kernel": "embeddings.word_embeddings.weight",
    "lm_head_decoder/bias": "lm_head.decoder.bias",
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

_QKVO_RE = re.compile(
    r"blocks_(\d+)_attention_attn/blocks_\d+_(q|k|v|o)/(kernel|bias)$"
)
_DENSE_RE = re.compile(r"blocks_(\d+)_(intermediate_dense|output_dense)/(kernel|bias)$")
_NORM_RE = re.compile(
    r"blocks_(\d+)_(attention_layernorm|output_layernorm)/(gamma|beta)$"
)


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = _QKVO_RE.match(path)
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.attn.{proj}.{suffix}"

    m = _DENSE_RE.match(path)
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    m = _NORM_RE.match(path)
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("mpnet."):
        key = key[len("mpnet.") :]
    return key.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
        "LayerNorm.beta", "LayerNorm.bias"
    )


def transfer_mpnet_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
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


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import MPNetForMaskedLM
    from transformers import MPNetModel as HFMPNetModel

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.mpnet import MPNetMaskedLM, MPNetModel

    HF_TOKEN = os.environ.get("HF_TOKEN")
    MPNET_VARIANTS = {"mpnet_base": "microsoft/mpnet-base"}

    rng = np.random.default_rng(0)

    for variant, hf_id in MPNET_VARIANTS.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        hf_model = HFMPNetModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_mlm = MPNetForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        arch = MPNetModel.config_from_hf(hf_model.config.to_dict())

        # Pad id 1 so the pad-offset position ids exercise the masked-cumsum path.
        ids = rng.integers(2, arch["vocab_size"], (2, 16)).astype("int64")
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        ids[0, 12:] = arch["pad_token_id"]
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }
        valid = mask.astype(bool)

        with torch.no_grad():
            hf_out = hf_model(**pt)
            hf_logits = hf_mlm(**pt).logits.detach().cpu().numpy()
        hf_seq = hf_out.last_hidden_state.detach().cpu().numpy()
        hf_pool = hf_out.pooler_output.detach().cpu().numpy()

        # One superset checkpoint = encoder + pooler + MLM head. HF splits these across
        # MPNetModel (pooler) and MPNetForMaskedLM (MLM head), so merge both state dicts;
        # keys differ only by the "mpnet." prefix normalize_hf_key strips. Each zeromodels
        # class then loads its own subset out of this single file.
        merged = {**dict(hf_mlm.state_dict()), **dict(hf_model.state_dict())}
        keras_full = MPNetMaskedLM(**arch, add_pooler=True)
        transfer_mpnet_weights(keras_full, merged)

        full_out = keras_full(k_inputs, training=False)
        full_logits = keras.ops.convert_to_numpy(full_out["logits"])
        full_pool = keras.ops.convert_to_numpy(full_out["pooler_output"])
        d_mlm = float(np.abs(hf_logits[valid] - full_logits[valid]).max())
        d_pool = float(np.abs(hf_pool - full_pool).max())
        print(f"  full checkpoint   mlm diff: {d_mlm:.3e}   pooler diff: {d_pool:.3e}")
        if max(d_mlm, d_pool) > 1e-3:
            raise ValueError(f"{variant}: full-checkpoint parity failed")

        out_path = f"{variant}.weights.h5"
        keras_full.save_weights(out_path)
        print(f"  Saved single file -> {out_path}")

        # Verify the ONE file serves every view: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check parity end to end.
        ref = MPNetMaskedLM(**arch, add_pooler=True)
        ref.load_weights(out_path)

        mm = MPNetModel(**arch)
        copy_weights_by_path_suffix(ref, mm)
        mm_out = mm(k_inputs, training=False)
        d_seq = float(
            np.abs(
                hf_seq[valid]
                - keras.ops.convert_to_numpy(mm_out["last_hidden_state"])[valid]
            ).max()
        )
        d_mmpool = float(
            np.abs(hf_pool - keras.ops.convert_to_numpy(mm_out["pooler_output"])).max()
        )

        mlm2 = MPNetMaskedLM(**arch)
        copy_weights_by_path_suffix(ref, mlm2)
        d_mlm2 = float(
            np.abs(
                hf_logits[valid]
                - keras.ops.convert_to_numpy(mlm2(k_inputs, training=False))[valid]
            ).max()
        )
        print(
            f"  reload   MPNetModel seq: {d_seq:.3e}  pooler: {d_mmpool:.3e}  |  "
            f"MPNetMaskedLM mlm: {d_mlm2:.3e}"
        )
        if max(d_seq, d_mmpool, d_mlm2) > 1e-3:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_model, hf_mlm, keras_full, ref, mm, mlm2
        keras.backend.clear_session()
        gc.collect()
