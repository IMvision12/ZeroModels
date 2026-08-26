from zeromodels.models.granite_speech.convert_granite_speech_hf_to_keras import (
    transfer_granite_speech_weights,
)

from .granite_speech_plus_model import GraniteSpeechPlusConditionalGenerate

# Per-variant recipe (relocated from granite_speech_plus_config.py), as overrides of
# the GraniteSpeechConfig defaults. The Plus variant adds cat_hidden_layers.
GRANITE_SPEECH_PLUS_RECIPES = {
    "granite_speech_4_1_2b_plus": {
        "vocab_size": 100353,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_kv_heads": 4,
        "rope_theta": 10000.0,
        "attention_multiplier": 0.0078125,
        "eos_token_id": 100257,
        "audio_token_id": 100352,
        "has_lora_adapter": False,
        "encoder_output_dim": 348,
        "cat_hidden_layers": [3],
    },
}


if __name__ == "__main__":
    import gc
    import math

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from keras import ops
    from safetensors.torch import load_file

    VARIANT = "granite_speech_4_1_2b_plus"
    HF_ID = "ibm-granite/granite-speech-4.1-2b-plus"
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

    model = GraniteSpeechPlusConditionalGenerate(**GRANITE_SPEECH_PLUS_RECIPES[VARIANT])
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
