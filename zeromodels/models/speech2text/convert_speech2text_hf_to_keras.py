import gc
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)

from .speech2text_layers import Speech2TextAttention

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight", "beta": "bias"}
EMBED_MAP = {"embeddings": "weight"}


def transfer_speech2text_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in hf_state_dict.items()
    }
    encoder = keras_model.encoder
    decoder = keras_model.decoder

    for i in range(keras_model.num_conv_layers):
        conv = encoder.get_layer(f"encoder_conv_layers_{i}")
        conv.kernel.assign(
            np.transpose(state[f"encoder.conv.conv_layers.{i}.weight"], (2, 1, 0))
        )
        transfer_weights("bias", conv.bias, state[f"encoder.conv.conv_layers.{i}.bias"])

    enc_attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, Speech2TextAttention)
    }
    for i in tqdm(
        range(keras_model.encoder_num_layers), desc="Transferring encoder layers"
    ):
        kp, hp = f"encoder_layers_{i}", f"encoder.layers.{i}"
        transfer_nested_layer_weights(
            enc_attns[f"{kp}_self_attn"],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{kp}_self_attn_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc1"), state, f"{hp}.fc1", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc2"), state, f"{hp}.fc2", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )
    transfer_nested_layer_weights(
        encoder.get_layer("encoder_layer_norm"),
        state,
        "encoder.layer_norm",
        name_mapping=LN_MAP,
    )

    transfer_nested_layer_weights(
        decoder.get_layer("decoder_embed_tokens"),
        state,
        "decoder.embed_tokens",
        name_mapping=EMBED_MAP,
    )

    dec_attns = {
        layer.name_prefix: layer
        for layer in decoder.layers
        if isinstance(layer, Speech2TextAttention)
    }
    for i in tqdm(
        range(keras_model.decoder_num_layers), desc="Transferring decoder layers"
    ):
        kp, hp = f"decoder_layers_{i}", f"decoder.layers.{i}"
        transfer_nested_layer_weights(
            dec_attns[f"{kp}_self_attn"],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{kp}_self_attn_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            dec_attns[f"{kp}_encoder_attn"],
            state,
            f"{hp}.encoder_attn",
            name_mapping={f"{kp}_encoder_attn_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_encoder_attn_layer_norm"),
            state,
            f"{hp}.encoder_attn_layer_norm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc1"), state, f"{hp}.fc1", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc2"), state, f"{hp}.fc2", name_mapping=DENSE_MAP
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )
    transfer_nested_layer_weights(
        decoder.get_layer("decoder_layer_norm"),
        state,
        "decoder.layer_norm",
        name_mapping=LN_MAP,
    )

    lm_head = decoder.get_layer("lm_head")
    lm_w = state.get("lm_head.weight", state.get("decoder.embed_tokens.weight"))
    lm_head.kernel.assign(np.transpose(np.asarray(lm_w)))


# Per-variant recipes (relocated from speech2text_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the kf_config backfill.
def _s2t(hidden, eh, dh, ef, df):
    return {
        "hidden_dim": hidden,
        "encoder_num_layers": 12,
        "decoder_num_layers": 6,
        "encoder_attention_heads": eh,
        "decoder_attention_heads": dh,
        "encoder_ffn_dim": ef,
        "decoder_ffn_dim": df,
        "vocab_size": 10000,
        "num_mel_bins": 80,
        "max_source_positions": 6000,
        "max_target_positions": 1024,
        "conv_channels": 1024,
        "conv_kernel_sizes": (5, 5),
        "num_conv_layers": 2,
        "scale_embedding": True,
        "activation_function": "relu",
    }


SPEECH2TEXT_RECIPES = {
    "s2t-small-librispeech-asr": _s2t(256, 4, 4, 2048, 2048),
    "s2t-medium-librispeech-asr": _s2t(512, 8, 8, 2048, 2048),
    "s2t-large-librispeech-asr": _s2t(1024, 16, 16, 4096, 4096),
}


if __name__ == "__main__":
    import torch
    from keras import ops
    from transformers import Speech2TextForConditionalGeneration

    from zeromodels.models.speech2text import Speech2TextModel

    HF_CHECKPOINT = {
        "s2t-small-librispeech-asr": "facebook/s2t-small-librispeech-asr",
        "s2t-medium-librispeech-asr": "facebook/s2t-medium-librispeech-asr",
        "s2t-large-librispeech-asr": "facebook/s2t-large-librispeech-asr",
    }

    for variant, hf_name in HF_CHECKPOINT.items():
        print(f"\n{'=' * 60}\nConverting {hf_name}\n{'=' * 60}")

        print("[1/4] Loading torch model")
        torch_model = (
            Speech2TextForConditionalGeneration.from_pretrained(
                hf_name, torch_dtype=torch.float32
            )
            .eval()
            .float()
        )
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        cfg = torch_model.config

        print(f"[2/4] Building Keras {variant}")
        model = Speech2TextModel(**SPEECH2TEXT_RECIPES[variant])

        print("[3/4] Transferring weights")
        transfer_speech2text_weights(model, state)

        print("[4/4] Verifying parity with HF")
        np.random.seed(0)
        test_feats = np.random.randn(1, 200, cfg.input_feat_per_channel).astype(
            np.float32
        )
        test_ids = np.array([[cfg.decoder_start_token_id, 100, 200]], dtype=np.int32)
        keras_logits = ops.convert_to_numpy(
            model({"input_features": test_feats, "decoder_input_ids": test_ids})[
                "logits"
            ]
        )
        with torch.no_grad():
            hf_logits = (
                torch_model(
                    input_features=torch.from_numpy(test_feats),
                    decoder_input_ids=torch.from_numpy(test_ids),
                )
                .logits.detach()
                .cpu()
                .numpy()
            )
        diff = float(np.max(np.abs(keras_logits - hf_logits)))
        print(f"  max abs logit diff: {diff:.6e}")

        out_path = f"{variant.replace('-', '_')}.weights.h5"
        model.save_weights(out_path)
        print(f"Saved -> {out_path}")
        assert diff < 5e-3, f"{variant}: logit diff too high: {diff:.6e}"

        del torch_model, model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
