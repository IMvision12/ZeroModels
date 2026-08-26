import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/tok_embeddings/embeddings": "embeddings.tok_embeddings.weight",
    "embeddings/norm/gamma": "embeddings.norm.weight",
    "final_norm/gamma": "final_norm.weight",
    "head_dense/kernel": "head.dense.weight",
    "head_norm/gamma": "head.norm.weight",
    "mlm_decoder/kernel": "embeddings.tok_embeddings.weight",
    "mlm_decoder/bias": "decoder.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
}

_OPTIONAL_HEADS = ("classifier", "head_dense", "head_norm", "mlm_decoder")


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(r"blocks_(\d+)_attn/blocks_\d+_attn_(Wqkv|Wo)/kernel$", path)
    if m:
        idx, proj = m.groups()
        return f"layers.{idx}.attn.{proj}.weight"

    m = re.match(r"blocks_(\d+)_mlp/blocks_\d+_mlp_(Wi|Wo)/kernel$", path)
    if m:
        idx, proj = m.groups()
        return f"layers.{idx}.mlp.{proj}.weight"

    m = re.match(r"blocks_(\d+)_(attn_norm|mlp_norm)/gamma$", path)
    if m:
        idx, norm = m.groups()
        return f"layers.{idx}.{norm}.weight"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("model."):
        key = key[len("model.") :]
    return key


def transfer_modernbert_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_HEADS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import ModernBertForMaskedLM
    from transformers import ModernBertModel as HFModernBertModel

    from zeromodels.models.modernbert import ModernBertMaskedLM, ModernBertModel

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "modernbert_base": "answerdotai/ModernBERT-base",
        "modernbert_large": "answerdotai/ModernBERT-large",
    }

    def cosine(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rng = np.random.default_rng(0)

    for variant, hf_id in HF_SOURCES.items():
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        hf_model = HFModernBertModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        arch = ModernBertModel.config_from_hf(hf_model.config.to_dict())

        # Sequence longer than the local window (2*64+1=129) so the sliding-window
        # attention path is actually exercised, plus some right padding.
        ids = rng.integers(0, arch["vocab_size"], (2, 160)).astype("int64")
        mask = np.ones((2, 160), dtype="int64")
        mask[0, 150:] = 0
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }

        keras_model = ModernBertModel(**arch)
        transfer_modernbert_weights(keras_model, dict(hf_model.state_dict()))
        with torch.no_grad():
            hf_out = hf_model(**pt).last_hidden_state.detach().cpu().numpy()
        seq = keras_model(k_inputs, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        d_seq = float(np.abs(hf_out - seq).max())
        c_seq = cosine(hf_out, seq)
        print(f"  last_hidden_state max diff: {d_seq:.3e}  cosine: {c_seq:.6f}")
        if c_seq < 0.9999:
            raise ValueError(
                f"{variant}: ModernBertModel parity failed (cosine {c_seq:.4f})"
            )

        hf_mlm = ModernBertForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = ModernBertMaskedLM(**arch)
        transfer_modernbert_weights(keras_mlm, dict(hf_mlm.state_dict()))
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits.detach().cpu().numpy()
        k_logits = keras_mlm(k_inputs, training=False)
        k_logits = (
            k_logits.detach().cpu().numpy()
            if hasattr(k_logits, "detach")
            else np.asarray(k_logits)
        )
        d_mlm = float(np.abs(hf_logits - k_logits).max())
        c_mlm = cosine(hf_logits, k_logits)
        print(f"  mlm logits max diff: {d_mlm:.3e}  cosine: {c_mlm:.6f}")
        if c_mlm < 0.9999:
            raise ValueError(
                f"{variant}: ModernBertMaskedLM parity failed (cosine {c_mlm:.4f})"
            )
        out_path = f"{variant}.weights.h5"
        keras_mlm.save_weights(out_path)
        print(f"  Saved MLM superset -> {out_path}")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
