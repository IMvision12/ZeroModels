import gc
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import transfer_nested_layer_weights

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight", "beta": "bias"}


def _attn(layer, state, hp, kp, kind):
    transfer_nested_layer_weights(
        layer,
        state,
        f"{hp}.{kind}",
        name_mapping={f"{kp}_{kind}_": "", "kernel": "weight"},
    )


def transfer_bart_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    state = {
        (k[len("model.") :] if k.startswith("model.") else k): np.asarray(v)
        for k, v in hf_state_dict.items()
    }
    km = keras_model

    shared_w = state.get("shared.weight")
    if shared_w is None:
        shared_w = state.get("encoder.embed_tokens.weight")
    if shared_w is None:
        shared_w = state.get("decoder.embed_tokens.weight")
    km.shared.embeddings.assign(shared_w)

    km.encoder_embed_positions.weight.assign(state["encoder.embed_positions.weight"])
    km.decoder_embed_positions.weight.assign(state["decoder.embed_positions.weight"])

    transfer_nested_layer_weights(
        km.encoder_layernorm_embedding,
        state,
        "encoder.layernorm_embedding",
        name_mapping=LN_MAP,
    )
    transfer_nested_layer_weights(
        km.decoder_layernorm_embedding,
        state,
        "decoder.layernorm_embedding",
        name_mapping=LN_MAP,
    )

    for i in tqdm(range(km.encoder_num_layers), desc="encoder"):
        b, hp, kp = km.encoder_blocks[i], f"encoder.layers.{i}", f"encoder_layers_{i}"
        _attn(b["self_attn"], state, hp, kp, "self_attn")
        transfer_nested_layer_weights(
            b["self_ln"], state, f"{hp}.self_attn_layer_norm", name_mapping=LN_MAP
        )
        transfer_nested_layer_weights(
            b["fc1"], state, f"{hp}.fc1", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            b["fc2"], state, f"{hp}.fc2", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            b["final_ln"], state, f"{hp}.final_layer_norm", name_mapping=LN_MAP
        )

    for i in tqdm(range(km.decoder_num_layers), desc="decoder"):
        b, hp, kp = km.decoder_blocks[i], f"decoder.layers.{i}", f"decoder_layers_{i}"
        _attn(b["self_attn"], state, hp, kp, "self_attn")
        transfer_nested_layer_weights(
            b["self_ln"], state, f"{hp}.self_attn_layer_norm", name_mapping=LN_MAP
        )
        _attn(b["cross_attn"], state, hp, kp, "encoder_attn")
        transfer_nested_layer_weights(
            b["cross_ln"], state, f"{hp}.encoder_attn_layer_norm", name_mapping=LN_MAP
        )
        transfer_nested_layer_weights(
            b["fc1"], state, f"{hp}.fc1", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            b["fc2"], state, f"{hp}.fc2", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            b["final_ln"], state, f"{hp}.final_layer_norm", name_mapping=LN_MAP
        )
    if getattr(km, "tied_head", None) is not None and "final_logits_bias" in state:
        km.tied_head.final_logits_bias.assign(state["final_logits_bias"])

    if hasattr(km, "classification_head_dense"):
        transfer_nested_layer_weights(
            km.classification_head_dense,
            state,
            "classification_head.dense",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            km.classification_head_out_proj,
            state,
            "classification_head.out_proj",
            name_mapping=DENSE_MAP,
        )

    if hasattr(km, "qa_outputs"):
        transfer_nested_layer_weights(
            km.qa_outputs, state, "qa_outputs", name_mapping=DENSE_MAP
        )


if __name__ == "__main__":
    import os

    import torch
    from keras import ops
    from transformers import (
        BartForConditionalGeneration,
        BartForSequenceClassification,
    )

    from zeromodels.models.bart import (
        BartConditionalGenerate,
        BartSequenceClassify,
    )

    OUT_DIR = os.environ.get("BART_OUT_DIR", "C:/Users/gites/Desktop/code/v1_weights")
    os.makedirs(OUT_DIR, exist_ok=True)

    CHECKPOINTS = [
        (
            "bart-base",
            "facebook/bart-base",
            BartForConditionalGeneration,
            BartConditionalGenerate,
            "gen",
        ),
        (
            "bart-large",
            "facebook/bart-large",
            BartForConditionalGeneration,
            BartConditionalGenerate,
            "gen",
        ),
        (
            "bart-large-cnn",
            "facebook/bart-large-cnn",
            BartForConditionalGeneration,
            BartConditionalGenerate,
            "gen",
        ),
        (
            "bart-large-xsum",
            "facebook/bart-large-xsum",
            BartForConditionalGeneration,
            BartConditionalGenerate,
            "gen",
        ),
        (
            "bart-large-mnli",
            "facebook/bart-large-mnli",
            BartForSequenceClassification,
            BartSequenceClassify,
            "cls",
        ),
    ]

    def cosine(a, b):
        a, b = a.ravel().astype("float64"), b.ravel().astype("float64")
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for variant, hf_name, hf_cls, km_cls, kind in CHECKPOINTS:
        print(f"\n{'=' * 60}\nConverting {hf_name}\n{'=' * 60}")
        hf_model = (
            hf_cls.from_pretrained(hf_name, torch_dtype=torch.float32).eval().float()
        )
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
        cfg = hf_model.config.to_dict()

        km = km_cls(**km_cls.config_from_hf(cfg))
        transfer_bart_weights(km, state)

        np.random.seed(0)
        vocab = cfg["vocab_size"]
        ids = np.random.randint(3, vocab, size=(1, 12)).astype("int32")
        ids[:, 0], ids[:, -1] = cfg["bos_token_id"], cfg["eos_token_id"]
        am = np.ones_like(ids)
        if kind == "gen":
            dec = np.concatenate(
                [[[cfg["decoder_start_token_id"]]], ids[:, :-1]], axis=1
            ).astype("int32")
            k_out = ops.convert_to_numpy(
                km({"input_ids": ids, "attention_mask": am, "decoder_input_ids": dec})[
                    "logits"
                ]
            )
            with torch.no_grad():
                h_out = hf_model(
                    input_ids=torch.from_numpy(ids),
                    attention_mask=torch.from_numpy(am),
                    decoder_input_ids=torch.from_numpy(dec),
                ).logits.numpy()
        else:  # cls
            k_out = ops.convert_to_numpy(km({"input_ids": ids, "attention_mask": am}))
            with torch.no_grad():
                h_out = hf_model(
                    input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(am)
                ).logits.numpy()

        diff = float(np.max(np.abs(k_out - h_out)))
        cos = cosine(k_out, h_out)
        print(f"  parity: max|diff|={diff:.3e}  cosine={cos:.8f}")

        out = os.path.join(OUT_DIR, f"{variant.replace('-', '_')}.weights.h5")
        km.save_weights(out)
        print(f"  saved -> {out}")
        assert diff < 1e-3, f"{variant}: parity too low (max|diff|={diff:.3e})"

        del hf_model, km, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
