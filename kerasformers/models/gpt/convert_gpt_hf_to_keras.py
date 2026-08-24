import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "tokens_embed.embeddings": "tokens_embed.weight",
    "positions_embed.embeddings": "positions_embed.weight",
    "block_": "h.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

# c_attn/c_proj/c_fc are Conv1D: weight already (in, out) -> direct copy, no transpose.
_CONV1D = ("c_attn", "c_proj", "c_fc")


def hf_name_for(path):
    # Functional model weight paths are flat (no model-name root to strip).
    name = path.replace("/", ".")
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def transfer_gpt_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = hf_name_for(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        if weight.path.endswith("/kernel") and any(c in weight.path for c in _CONV1D):
            weight.assign(np.asarray(hf_state_dict[name]))
        else:
            transfer_weights(weight.path, weight, hf_state_dict[name])


# Per-variant recipes (relocated from gpt_config.py). Models load from the Hub by
# repo id; these build the arch for conversion.
GPT_VARIANTS = {
    "gpt": {
        "vocab_size": 40478,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
}


if __name__ == "__main__":
    import gc

    import keras
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import OpenAIGPTLMHeadModel

    from kerasformers.models.gpt import GptTextGenerate

    HF_SOURCES = {"gpt": "openai-community/openai-gpt"}
    rng = np.random.default_rng(0)

    for variant, arch in GPT_VARIANTS.items():
        hf_id = HF_SOURCES[variant]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        sd = load_file(hf_hub_download(hf_id, "model.safetensors"))
        model = GptTextGenerate(**arch)
        transfer_gpt_weights(model, sd)
        del sd

        ids = rng.integers(0, arch["vocab_size"], (1, 16)).astype("int64")
        k_logits = model({"input_ids": ids.astype("int32")})["logits"]
        k_logits = (
            k_logits.detach().cpu().numpy()
            if hasattr(k_logits, "detach")
            else np.asarray(k_logits)
        )
        hf = OpenAIGPTLMHeadModel.from_pretrained(hf_id).eval()
        with torch.no_grad():
            hf_logits = hf(torch.from_numpy(ids)).logits.numpy()
        d = float(np.abs(hf_logits - k_logits).max())
        print(f"  logits max diff: {d:.3e}")
        if d > 1e-3:
            raise ValueError(f"{variant}: GPT parity failed ({d:.3e})")

        out_path = f"{variant}.weights.h5"
        model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del hf, model
        keras.backend.clear_session()
        gc.collect()
