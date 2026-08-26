import gc
import os
from typing import Dict

import numpy as np
from keras import ops
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.whisper import WhisperModel
from zeromodels.models.whisper.whisper_layers import WhisperAttention

HF_CHECKPOINT = {
    "whisper_tiny": "openai/whisper-tiny",
    "whisper_base": "openai/whisper-base",
    "whisper_small": "openai/whisper-small",
    "whisper_medium": "openai/whisper-medium",
    "whisper_large": "openai/whisper-large",
    "whisper_large_v2": "openai/whisper-large-v2",
    "whisper_large_v3": "openai/whisper-large-v3",
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo",
}


# Per-variant recipes (relocated from whisper_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the zm_config backfill.
def _w(hidden, el, eh, ef, dl, dh, df, vocab, mel):
    return {
        "hidden_dim": hidden,
        "encoder_num_layers": el,
        "encoder_attention_heads": eh,
        "encoder_ffn_dim": ef,
        "decoder_num_layers": dl,
        "decoder_attention_heads": dh,
        "decoder_ffn_dim": df,
        "vocab_size": vocab,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": mel,
    }


WHISPER_RECIPES = {
    "whisper_tiny": _w(384, 4, 6, 1536, 4, 6, 1536, 51865, 80),
    "whisper_base": _w(512, 6, 8, 2048, 6, 8, 2048, 51865, 80),
    "whisper_small": _w(768, 12, 12, 3072, 12, 12, 3072, 51865, 80),
    "whisper_medium": _w(1024, 24, 16, 4096, 24, 16, 4096, 51865, 80),
    "whisper_large": _w(1280, 32, 20, 5120, 32, 20, 5120, 51865, 80),
    "whisper_large_v2": _w(1280, 32, 20, 5120, 32, 20, 5120, 51865, 80),
    "whisper_large_v3": _w(1280, 32, 20, 5120, 32, 20, 5120, 51866, 128),
    "whisper_large_v3_turbo": _w(1280, 32, 20, 5120, 4, 20, 5120, 51866, 128),
}

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight", "beta": "bias"}
EMBED_MAP = {"embeddings": "weight"}


def transfer_whisper_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in hf_state_dict.items()
    }
    encoder = keras_model.encoder
    decoder = keras_model.decoder

    # ---- Encoder ----
    for i in (1, 2):
        conv = encoder.get_layer(f"encoder_conv{i}")
        conv.kernel.assign(np.transpose(state[f"encoder.conv{i}.weight"], (2, 1, 0)))
        transfer_weights("bias", conv.bias, state[f"encoder.conv{i}.bias"])

    encoder.get_layer("encoder_embed_positions").pos_embed.assign(
        state["encoder.embed_positions.weight"]
    )

    enc_attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, WhisperAttention)
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

    # ---- Decoder ----
    transfer_nested_layer_weights(
        decoder.get_layer("decoder_embed_tokens"),
        state,
        "decoder.embed_tokens",
        name_mapping=EMBED_MAP,
    )
    decoder.get_layer("decoder_embed_positions").pos_embed.assign(
        state["decoder.embed_positions.weight"]
    )

    dec_attns = {
        layer.name_prefix: layer
        for layer in decoder.layers
        if isinstance(layer, WhisperAttention)
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


def transfer_whisper_audio_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a ``WhisperForAudioClassification`` state dict (encoder + head)."""
    state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in hf_state_dict.items()
    }
    encoder = keras_model.encoder

    # ---- Encoder ----
    for i in (1, 2):
        conv = encoder.get_layer(f"encoder_conv{i}")
        conv.kernel.assign(np.transpose(state[f"encoder.conv{i}.weight"], (2, 1, 0)))
        transfer_weights("bias", conv.bias, state[f"encoder.conv{i}.bias"])

    encoder.get_layer("encoder_embed_positions").pos_embed.assign(
        state["encoder.embed_positions.weight"]
    )

    enc_attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, WhisperAttention)
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

    if keras_model.use_weighted_layer_sum:
        keras_model.get_layer("layer_weights").layer_weights.assign(
            state["layer_weights"]
        )

    projector = keras_model.get_layer("projector")
    projector.kernel.assign(np.transpose(state["projector.weight"]))
    transfer_weights("projector.bias", projector.bias, state["projector.bias"])

    classifier = keras_model.get_layer("classifier")
    classifier.kernel.assign(np.transpose(state["classifier.weight"]))
    transfer_weights("classifier.bias", classifier.bias, state["classifier.bias"])


if __name__ == "__main__":
    import torch
    from transformers import WhisperForConditionalGeneration

    SLUG = {
        "whisper_tiny": "tiny",
        "whisper_base": "base",
        "whisper_small": "small",
        "whisper_medium": "medium",
        "whisper_large": "large",
        "whisper_large_v2": "largev2",
        "whisper_large_v3": "largev3",
        "whisper_large_v3_turbo": "largev3turbo",
    }

    for variant, hf_name in HF_CHECKPOINT.items():
        print(f"\n{'=' * 60}")
        print(f"Converting {hf_name}")
        print(f"{'=' * 60}")

        base = f"whisper{SLUG[variant]}_openai"
        if os.path.exists(f"{base}.weights.h5") or os.path.exists(
            f"{base}.weights.json"
        ):
            print(f"  already converted, skipping ({base})")
            continue

        print(f"[1/4] Loading {hf_name}")
        torch_model = (
            WhisperForConditionalGeneration.from_pretrained(
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
        model = WhisperModel(**WHISPER_RECIPES[variant])

        print("[3/4] Transferring weights")
        transfer_whisper_weights(model, state)

        print("[4/4] Verifying parity with HF")
        np.random.seed(0)
        test_mel = np.random.randn(1, cfg.num_mel_bins, 3000).astype(np.float32)
        test_ids = np.array(
            [[cfg.decoder_start_token_id, cfg.decoder_start_token_id + 1]],
            dtype=np.int32,
        )
        keras_logits = ops.convert_to_numpy(
            model({"input_features": test_mel, "decoder_input_ids": test_ids})["logits"]
        )
        with torch.no_grad():
            hf_logits = (
                torch_model(
                    input_features=torch.from_numpy(test_mel),
                    decoder_input_ids=torch.from_numpy(test_ids),
                )
                .logits.detach()
                .cpu()
                .numpy()
            )
        diff = float(np.max(np.abs(keras_logits - hf_logits)))
        print(f"  max abs logit diff: {diff:.6e}")
        if diff > 1e-3:
            print(f"  WARNING: parity above 1e-3 (saw {diff:.6e})")

        total_params = sum(int(np.prod(w.shape)) for w in model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{base}.weights.json"
            model.save_weights(out_path, max_shard_size=1.7)
            print(f"Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{base}.weights.h5"
            model.save_weights(out_path)
            print(f"Saved -> {out_path} (~{total_gb:.2f} GB)")

        assert diff < 5e-3, f"{variant}: logit diff too high: {diff:.6e}"

        del torch_model, model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
