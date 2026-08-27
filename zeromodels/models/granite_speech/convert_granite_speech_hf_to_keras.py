import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

from .granite_speech_model import GraniteSpeechConditionalGenerate

DECODER_MAPPING = {
    "language_model.token_embedding.embeddings": "language_model.model.embed_tokens.weight",
    "language_model.final_norm.weight": "language_model.model.norm.weight",
    "language_model.decoder_layer_": "language_model.model.layers.",
    "attention.query.lora_a": "self_attn.q_proj.lora_A.weight",
    "attention.query.lora_b": "self_attn.q_proj.lora_B.weight",
    "attention.value.lora_a": "self_attn.v_proj.lora_A.weight",
    "attention.value.lora_b": "self_attn.v_proj.lora_B.weight",
    "attention.query.kernel": "self_attn.q_proj.weight",
    "attention.key.kernel": "self_attn.k_proj.weight",
    "attention.value.kernel": "self_attn.v_proj.weight",
    "attention.output_proj.kernel": "self_attn.o_proj.weight",
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
}

ENCODER_MAPPING = {
    "encoder.conformer_layer_": "encoder.layers.",
    "attn.to_q.kernel": "attn.to_q.weight",
    "attn.to_kv.kernel": "attn.to_kv.weight",
    "attn.rel_pos_emb": "attn.rel_pos_emb.weight",
    "conv.depth_kernel": "conv.depth_conv.conv.weight",
    "conv.batch_norm.moving_mean": "conv.batch_norm.running_mean",
    "conv.batch_norm.moving_variance": "conv.batch_norm.running_var",
}

PROJECTOR_MAPPING = {
    "projector.layernorm": "projector.qformer.layernorm",
    "projector.qformer_layer_": "projector.qformer.encoder.layer.",
    "crossattention.dense": "crossattention.output.dense",
    "crossattention.LayerNorm": "crossattention.output.LayerNorm",
    "attention.dense": "attention.output.dense",
    "attention.LayerNorm": "attention.output.LayerNorm",
    "intermediate_query": "intermediate_query.dense",
    "output_query_dense": "output_query.dense",
    "output_query_LayerNorm": "output_query.LayerNorm",
}

GENERIC_FIXUPS = {
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def transfer_granite_speech_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model.build_dummy()

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        keras_dotted = weight.path.replace("/", ".")
        if keras_dotted.startswith("audio_features."):
            keras_dotted = keras_dotted[len("audio_features.") :]
        if keras_dotted == "token_embedding.embeddings":
            name = "language_model.model.embed_tokens.weight"
        else:
            if keras_dotted.startswith("language_model."):
                table = DECODER_MAPPING
            elif keras_dotted.startswith("encoder."):
                table = ENCODER_MAPPING
            else:
                table = PROJECTOR_MAPPING
            name = keras_dotted
            for old, new in table.items():
                name = name.replace(old, new)
            for old, new in GENERIC_FIXUPS.items():
                name = name.replace(old, new)

        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]

        # transfer_weights handles the standard Dense / LayerNorm / BatchNorm
        # conversions; only these three need manual shaping (np.asarray for the
        # transposes, as the source may be a torch tensor).
        if weight.path.endswith("rel_pos_emb") or weight.path.endswith("query"):
            weight.assign(torch_weight)
            continue
        if "depth_kernel" in weight.path:
            weight.assign(np.asarray(torch_weight)[:, 0, :].T)
            continue
        if (
            "conv/up_conv" in weight.path or "conv/down_conv" in weight.path
        ) and weight.path.endswith("kernel"):
            weight.assign(np.asarray(torch_weight)[:, :, 0].T)
            continue

        transfer_weights(weight.path, weight, torch_weight)


# Per-variant recipes (relocated from granite_speech_config.py), as overrides of the
# GraniteSpeechConfig defaults (which describe granite_speech_3_3_2b). Models load
# from the Hub by repo id; these build the arch for conversion + drive the backfill.
GRANITE_SPEECH_RECIPES = {
    "granite_speech_3_3_2b": {},
    "granite_speech_3_3_8b": {
        "embed_dim": 4096,
        "mlp_dim": 12800,
        "attention_multiplier": 0.0078125,
        "logits_scaling": 16.0,
    },
    "granite_speech_4_1_2b": {
        "vocab_size": 100353,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_kv_heads": 4,
        "rope_theta": 10000.0,
        "attention_multiplier": 0.0078125,
        "tie_embeddings": False,
        "eos_token_id": 100257,
        "audio_token_id": 100352,
        "has_lora_adapter": False,
        "encoder_output_dim": 348,
    },
    "granite_4_0_1b_speech": {
        "vocab_size": 100353,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_kv_heads": 4,
        "rope_theta": 10000.0,
        "attention_multiplier": 0.0078125,
        "tie_embeddings": False,
        "eos_token_id": 100257,
        "audio_token_id": 100352,
        "has_lora_adapter": False,
        "encoder_output_dim": 348,
    },
}


if __name__ == "__main__":
    import gc
    import math
    import os

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from keras import ops
    from safetensors.torch import load_file

    VARIANT_TO_HF = {
        "granite_speech_3_3_2b": "ibm-granite/granite-speech-3.3-2b",
        "granite_speech_3_3_8b": "ibm-granite/granite-speech-3.3-8b",
        "granite_speech_4_1_2b": "ibm-granite/granite-speech-4.1-2b",
        "granite_4_0_1b_speech": "ibm-granite/granite-4.0-1b-speech",
    }
    VARIANT = os.environ.get("GRANITE_VARIANT", "granite_speech_3_3_2b")
    HF_ID = VARIANT_TO_HF[VARIANT]
    OUT = f"{VARIANT}.weights.json"

    files = list_repo_files(HF_ID)
    state = {}
    for shard in sorted(
        f for f in files if f.startswith("model") and f.endswith(".safetensors")
    ):
        for k, v in load_file(hf_hub_download(HF_ID, shard)).items():
            state[k] = v.to(torch.float32).cpu().numpy()
    if "adapter_model.safetensors" in files:
        for k, v in load_file(
            hf_hub_download(HF_ID, "adapter_model.safetensors")
        ).items():
            state[k.replace("base_model.model.", "")] = (
                v.to(torch.float32).cpu().numpy()
            )

    model = GraniteSpeechConditionalGenerate(**GRANITE_SPEECH_RECIPES[VARIANT])
    transfer_granite_speech_weights(model, state)

    frames = 4 * model.window_size
    nblocks = math.ceil(frames / model.window_size)
    n_audio = nblocks * (model.window_size // model.downsample_rate)
    out = model(
        {
            "input_ids": np.array(
                [[1] + [model.audio_token_id] * n_audio + [2]], dtype="int64"
            ),
            "input_features": np.zeros(
                (1, frames, model.encoder_input_dim), dtype="float32"
            ),
        }
    )
    print("  logits", tuple(ops.convert_to_numpy(out["logits"]).shape))

    model.save_weights(OUT, max_shard_size=2.0)
    del state
    gc.collect()
