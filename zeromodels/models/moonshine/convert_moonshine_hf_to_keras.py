import gc
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)

from .moonshine_layers import MoonshineAttention

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight"}
EMBED_MAP = {"embeddings": "weight"}


def transfer_moonshine_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in hf_state_dict.items()
    }
    encoder = keras_model.encoder
    decoder = keras_model.decoder

    conv1 = encoder.get_layer("encoder_conv1")
    conv1.kernel.assign(np.transpose(state["encoder.conv1.weight"], (2, 1, 0)))
    for i in (2, 3):
        conv = encoder.get_layer(f"encoder_conv{i}")
        conv.kernel.assign(np.transpose(state[f"encoder.conv{i}.weight"], (2, 1, 0)))
        transfer_weights("bias", conv.bias, state[f"encoder.conv{i}.bias"])

    groupnorm = encoder.get_layer("encoder_groupnorm")
    groupnorm.gamma.assign(state["encoder.groupnorm.weight"])
    groupnorm.beta.assign(state["encoder.groupnorm.bias"])

    enc_attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, MoonshineAttention)
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
            encoder.get_layer(f"{kp}_input_layernorm"),
            state,
            f"{hp}.input_layernorm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.mlp.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.mlp.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_post_attention_layernorm"),
            state,
            f"{hp}.post_attention_layernorm",
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
        if isinstance(layer, MoonshineAttention)
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
            decoder.get_layer(f"{kp}_input_layernorm"),
            state,
            f"{hp}.input_layernorm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            dec_attns[f"{kp}_encoder_attn"],
            state,
            f"{hp}.encoder_attn",
            name_mapping={f"{kp}_encoder_attn_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_post_attention_layernorm"),
            state,
            f"{hp}.post_attention_layernorm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.mlp.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.mlp.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_final_layernorm"),
            state,
            f"{hp}.final_layernorm",
            name_mapping=LN_MAP,
        )
    transfer_nested_layer_weights(
        decoder.get_layer("decoder_layer_norm"),
        state,
        "decoder.norm",
        name_mapping=LN_MAP,
    )


# Per-variant recipes (relocated from moonshine_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the zm_config backfill.
MOONSHINE_RECIPES = {
    "moonshine_tiny": {
        "hidden_dim": 288,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1152,
        "decoder_ffn_dim": 1152,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.9,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
    "moonshine_base": {
        "hidden_dim": 416,
        "encoder_num_layers": 8,
        "decoder_num_layers": 8,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1664,
        "decoder_ffn_dim": 1664,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.62,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
}


if __name__ == "__main__":
    import torch
    from keras import ops
    from transformers import MoonshineForConditionalGeneration

    from zeromodels.models.moonshine import MoonshineModel

    HF_CHECKPOINT = {
        "moonshine_tiny": "UsefulSensors/moonshine-tiny",
        "moonshine_base": "UsefulSensors/moonshine-base",
    }

    for variant, hf_name in HF_CHECKPOINT.items():
        print(f"\n{'=' * 60}\nConverting {hf_name}\n{'=' * 60}")

        print("[1/4] Loading torch model")
        torch_model = (
            MoonshineForConditionalGeneration.from_pretrained(
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
        model = MoonshineModel(**MOONSHINE_RECIPES[variant])

        print("[3/4] Transferring weights")
        transfer_moonshine_weights(model, state)

        print("[4/4] Verifying parity with HF")
        np.random.seed(0)
        test_audio = np.random.randn(1, 16000).astype(np.float32)
        test_ids = np.array(
            [[cfg.decoder_start_token_id, cfg.decoder_start_token_id + 5]],
            dtype=np.int32,
        )
        keras_logits = ops.convert_to_numpy(
            model({"input_values": test_audio, "decoder_input_ids": test_ids})["logits"]
        )
        with torch.no_grad():
            hf_logits = (
                torch_model(
                    input_values=torch.from_numpy(test_audio),
                    decoder_input_ids=torch.from_numpy(test_ids),
                )
                .logits.detach()
                .cpu()
                .numpy()
            )
        diff = float(np.max(np.abs(keras_logits - hf_logits)))
        print(f"  max abs logit diff: {diff:.6e}")

        out_path = f"{variant}_usefulsensors.weights.h5"
        model.save_weights(out_path)
        print(f"Saved -> {out_path}")
        assert diff < 5e-3, f"{variant}: logit diff too high: {diff:.6e}"

        del torch_model, model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
