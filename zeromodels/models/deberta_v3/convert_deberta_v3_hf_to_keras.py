from typing import Dict

import numpy as np

from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
    transfer_deberta_v2_weights,
)


def transfer_deberta_v3_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    transfer_deberta_v2_weights(keras_model, hf_state_dict)


# Architecture per variant, moved here from deberta_v3_config.py: the package config no longer
# carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer the HF weights into.
DEBERTA_V3_MODEL_CONFIG = {
    "deberta_v3_xsmall": {
        "vocab_size": 128100,
        "embed_dim": 384,
        "num_layers": 12,
        "num_heads": 6,
        "mlp_dim": 1536,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 0,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_v3_small": {
        "vocab_size": 128100,
        "embed_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 0,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_v3_base": {
        "vocab_size": 128100,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 0,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_v3_large": {
        "vocab_size": 128100,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 0,
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
    from transformers import DebertaV2Model as HFDebertaV2Model

    from zeromodels.models.deberta_v3 import DebertaV3Model

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_SOURCES = {
        "deberta_v3_xsmall": "microsoft/deberta-v3-xsmall",
        "deberta_v3_small": "microsoft/deberta-v3-small",
        "deberta_v3_base": "microsoft/deberta-v3-base",
        "deberta_v3_large": "microsoft/deberta-v3-large",
    }

    rng = np.random.default_rng(0)

    for variant, arch in DEBERTA_V3_MODEL_CONFIG.items():
        hf_id = HF_SOURCES[variant]
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

        hf_model = (
            HFDebertaV2Model.from_pretrained(hf_id, token=HF_TOKEN).float().eval()
        )
        keras_model = DebertaV3Model(**arch)
        transfer_deberta_v3_weights(keras_model, dict(hf_model.state_dict()))
        with torch.no_grad():
            ref = hf_model(**pt).last_hidden_state.numpy()
        seq = keras_model(k_inputs, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        valid = mask[..., None].astype(bool)
        d_seq = float(np.abs((ref - seq) * valid).max())
        print(f"  last_hidden_state max diff (non-pad): {d_seq:.3e}")
        if d_seq > 1e-3:
            raise ValueError(f"{variant}: DebertaV3Model parity failed")
        keras_model.save_weights(f"{variant}.weights.h5")
        print(f"  Saved -> {variant}.weights.h5")

        del hf_model, keras_model
        keras.backend.clear_session()
        gc.collect()
