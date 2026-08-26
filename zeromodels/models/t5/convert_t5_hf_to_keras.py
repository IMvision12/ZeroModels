import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

# The T5 layers use flat leaf names (``enc_0_attn_q``, ``dec_3_cross_ln`` ...) so every
# weight's last-two path segments are unique -- convert by the LEAF (the penultimate path
# segment), which encodes stack + block + role. Leaf -> HF base key; the weight-type
# suffix (``.weight`` / ``.bias``) is appended in :func:`hf_name_for`.
FIXED_LEAVES = {
    "shared": "shared",
    "encoder_rel_bias": (
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias"
    ),
    "decoder_rel_bias": (
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias"
    ),
    "encoder_final_layer_norm": "encoder.final_layer_norm",
    "decoder_final_layer_norm": "decoder.final_layer_norm",
    "classifier_dense": "classification_head.dense",
    "classifier_out_proj": "classification_head.out_proj",
    "classifier": "classifier",
    "qa_outputs": "qa_outputs",
}

# task-head leaves have no counterpart in the T5Model backbone checkpoint (left random).
_HEAD_LEAVES = ("classifier_dense", "classifier_out_proj", "classifier", "qa_outputs")

# (stack, role) -> (HF stack, HF sublayer index, HF attention name). Encoder: self-attn=0,
# ff=1. Decoder: self-attn=0, cross-attn=1, ff=2.
_ROLE = {
    ("enc", "attn"): ("encoder", "0", "SelfAttention"),
    ("enc", "ff"): ("encoder", "1", "DenseReluDense"),
    ("dec", "self"): ("decoder", "0", "SelfAttention"),
    ("dec", "cross"): ("decoder", "1", "EncDecAttention"),
    ("dec", "ff"): ("decoder", "2", "DenseReluDense"),
}


def hf_base_for(leaf: str) -> Optional[str]:
    if leaf in FIXED_LEAVES:
        return FIXED_LEAVES[leaf]
    m = re.match(r"(enc|dec)_(\d+)_(attn|self|cross|ff)_(q|k|v|o|wi|wo|ln)$", leaf)
    if not m:
        return None
    stack, idx, role, part = m.groups()
    hf_stack, sub, name = _ROLE[(stack, role)]
    if part == "ln":
        return f"{hf_stack}.block.{idx}.layer.{sub}.layer_norm"
    return f"{hf_stack}.block.{idx}.layer.{sub}.{name}.{part}"


def hf_name_for(path: str) -> Optional[str]:
    parts = path.split("/")
    if len(parts) < 2:
        return None
    base = hf_base_for(parts[-2])
    if base is None:
        return None
    return base + (".bias" if parts[-1] == "bias" else ".weight")


def normalize_hf_key(key: str) -> str:
    # composed HF heads (T5ForSequenceClassification / T5ForTokenClassification) nest the
    # backbone under `transformer.`; drop it so backbone keys align.
    if key.startswith("transformer."):
        key = key[len("transformer.") :]
    return key


def transfer_t5_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.split("/")[-2] in _HEAD_LEAVES:
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


# Architecture per variant, for the standalone converter (models otherwise load by Hub
# repo id / kf_config). Only this converter builds an untrained model to transfer into.
T5_MODEL_CONFIG = {
    "t5_small": {
        "vocab_size": 32128,
        "embed_dim": 512,
        "key_value_dim": 64,
        "mlp_dim": 2048,
        "num_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
    },
    "t5_base": {
        "vocab_size": 32128,
        "embed_dim": 768,
        "key_value_dim": 64,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_decoder_layers": 12,
        "num_heads": 12,
    },
    "t5_large": {
        "vocab_size": 32128,
        "embed_dim": 1024,
        "key_value_dim": 64,
        "mlp_dim": 4096,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 16,
    },
}


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import T5EncoderModel as HFT5EncoderModel
    from transformers import T5ForConditionalGeneration as HFT5ConditionalGenerate

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.t5 import T5ConditionalGenerate, T5EncoderModel, T5Model

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_SOURCES = {
        "t5_small": "google-t5/t5-small",
        "t5_base": "google-t5/t5-base",
        "t5_large": "google-t5/t5-large",
    }

    rng = np.random.default_rng(0)

    for variant, arch in T5_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(3, arch["vocab_size"], (2, 12)).astype("int64")
        dids = rng.integers(3, arch["vocab_size"], (2, 7)).astype("int64")
        dids[:, 0] = 0
        mask = np.ones((2, 12), dtype="int64")
        mask[0, 9:] = 0
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "decoder_input_ids": dids.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
            "decoder_input_ids": torch.from_numpy(dids),
        }

        hf_gen = HFT5ConditionalGenerate.from_pretrained(hf_id, token=HF_TOKEN).eval()
        state = {k: v.float().cpu().numpy() for k, v in hf_gen.state_dict().items()}

        # Parity check on T5ConditionalGenerate (the generative view), then save the T5Model superset
        # (encoder + decoder + shared embedding = every pretrained weight; the tied lm_head
        # adds none). kf_config declares T5Model; each class loads its subset from this file.
        keras_gen = T5ConditionalGenerate(**arch)
        keras_gen(k_inputs)  # build
        transfer_t5_weights(keras_gen, state)

        with torch.no_grad():
            hf_out = hf_gen(**pt)
            hf_logits = hf_out.logits.numpy()
        full_logits = np.asarray(keras_gen(k_inputs)["logits"].detach().cpu())
        d_gen = float(np.abs(hf_logits - full_logits).max())
        print(f"  T5ConditionalGenerate logits max diff: {d_gen:.3e}")
        if d_gen > 2e-3:
            raise ValueError(f"{variant}: T5ConditionalGenerate parity failed")

        keras_full = T5Model(**arch)
        keras_full(k_inputs)
        copy_weights_by_path_suffix(keras_gen, keras_full)
        out_path = f"{variant}.weights.h5"
        keras_full.save_weights(out_path)
        print(f"  Saved single file -> {out_path}")

        # Verify the ONE file serves every view: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check the encoder end to end.
        ref = T5Model(**arch)
        ref(k_inputs)
        ref.load_weights(out_path)
        enc = T5EncoderModel(**arch)
        enc(k_inputs)
        copy_weights_by_path_suffix(ref, enc)
        with torch.no_grad():
            hf_enc = HFT5EncoderModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
            enc_ref = hf_enc(
                input_ids=pt["input_ids"], attention_mask=pt["attention_mask"]
            ).last_hidden_state.numpy()
        enc_out = np.asarray(enc(k_inputs)["last_hidden_state"].detach().cpu())
        d_enc = float(np.abs((enc_ref - enc_out) * mask[..., None]).max())
        bm = T5Model(**arch)
        bm(k_inputs)
        copy_weights_by_path_suffix(ref, bm)
        d_dec = float(
            np.abs(
                np.asarray(bm(k_inputs)["last_hidden_state"].detach().cpu())
                - np.asarray(keras_gen(k_inputs)["last_hidden_state"].detach().cpu())
            ).max()
        )
        print(f"  reload  T5EncoderModel: {d_enc:.3e}  T5Model dec: {d_dec:.3e}")
        if max(d_enc, d_dec) > 2e-3:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_gen, keras_gen, keras_full, ref, enc, bm, hf_enc
        keras.backend.clear_session()
        gc.collect()
