from typing import Dict

import numpy as np

from zeromodels.models.roberta.convert_roberta_hf_to_keras import (
    transfer_roberta_weights,
)


def transfer_xlm_roberta_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    transfer_roberta_weights(keras_model, hf_state_dict)


# Architecture per variant, moved here from xlm_roberta_config.py: the package config no longer
# carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer the HF weights into.
XLM_ROBERTA_MODEL_CONFIG = {
    "xlm_roberta_base": {
        "vocab_size": 250002,
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
    "xlm_roberta_large": {
        "vocab_size": 250002,
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
    from transformers import XLMRobertaForMaskedLM
    from transformers import XLMRobertaModel as HFXLMRobertaModel

    from zeromodels.conversion.weight_transfer_util import (
        copy_weights_by_path_suffix,
    )
    from zeromodels.models.xlm_roberta import XLMRobertaMaskedLM, XLMRobertaModel

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "xlm_roberta_base": "FacebookAI/xlm-roberta-base",
        "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
    }
    SHARD_THRESHOLD_GB = 1.9

    rng = np.random.default_rng(0)

    for variant, arch in XLM_ROBERTA_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
        pad = arch["pad_token_id"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(5, arch["vocab_size"], (2, 16)).astype("int64")
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

        hf_model = HFXLMRobertaModel.from_pretrained(
            hf_id, token=HF_TOKEN
        ).eval()  # + pooler
        hf_mlm = XLMRobertaForMaskedLM.from_pretrained(  # + MLM head
            hf_id, token=HF_TOKEN
        ).eval()

        # One superset checkpoint = encoder + pooler + MLM head. HF splits these across
        # XLMRobertaModel (pooler) and XLMRobertaForMaskedLM (MLM head), so merge both state
        # dicts. The tied MLM decoder kernel is stripped from safetensors -- drop it so the
        # converter reconstructs it from the word embeddings (mirrors the `hf:` path). Each
        # zeromodels class then loads its own subset out of this single file.
        merged = {**dict(hf_mlm.state_dict()), **dict(hf_model.state_dict())}
        merged.pop("lm_head.decoder.weight", None)
        keras_full = XLMRobertaMaskedLM(**arch, add_pooler=True)
        transfer_xlm_roberta_weights(keras_full, merged)

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

        total_gb = (
            sum(int(np.prod(w.shape)) for w in keras_full.weights) * 4 / (1024**3)
        )
        if total_gb > SHARD_THRESHOLD_GB:
            out_path = f"{variant}.weights.json"
            keras_full.save_weights(out_path, max_shard_size=1.7)
        else:
            out_path = f"{variant}.weights.h5"
            keras_full.save_weights(out_path)
        print(f"  Saved single file -> {out_path} (~{total_gb:.2f} GiB)")

        # Verify the ONE file serves every view: reload into a same-class reference, copy
        # each subset out by semantic path, and re-check parity end to end.
        ref = XLMRobertaMaskedLM(**arch, add_pooler=True)
        ref.load_weights(out_path)
        rm = XLMRobertaModel(**arch)
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
        mlm2 = XLMRobertaMaskedLM(**arch)
        copy_weights_by_path_suffix(ref, mlm2)
        d_mlm2 = float(
            np.abs(
                hf_logits.detach().cpu().numpy()
                - mlm2(k_inputs, training=False).detach().cpu().numpy()
            ).max()
        )
        print(
            f"  reload   XLMRobertaModel seq: {d_seq:.3e}  pooler: {d_rmpool:.3e}  |  "
            f"XLMRobertaMaskedLM mlm: {d_mlm2:.3e}"
        )
        if max(d_seq, d_rmpool, d_mlm2) > 1e-3:
            raise ValueError(f"{variant}: single-file reload parity failed")

        del hf_model, hf_mlm, keras_full, ref, rm, mlm2
        keras.backend.clear_session()
        gc.collect()
