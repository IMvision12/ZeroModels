import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "wte.embeddings": "wte.weight",
    "wpe.embeddings": "wpe.weight",
    "ln_f.gamma": "ln_f.weight",
    "ln_f.beta": "ln_f.bias",
    "block_": "h.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

_CONV1D = ("c_attn", "c_proj", "c_fc")


def hf_name_for(path):
    # Functional model weight paths are flat ("wte/embeddings", "block_0/...");
    # there is no model-name root component to strip.
    name = path.replace("/", ".")
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def transfer_gpt2_weights(keras_model, hf_state_dict):
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


# Per-variant recipes (relocated from gpt2_config.py). Models load from the Hub by
# repo id; these build the arch for conversion.
GPT2_VARIANTS = {
    "gpt2": {
        "vocab_size": 50257,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_medium": {
        "vocab_size": 50257,
        "embed_dim": 1024,
        "mlp_dim": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_large": {
        "vocab_size": 50257,
        "embed_dim": 1280,
        "mlp_dim": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_xl": {
        "vocab_size": 50257,
        "embed_dim": 1600,
        "mlp_dim": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
}

# large / xl exceed GitHub's 2 GB release-asset cap, so they save as a sharded
# .weights.json index; the smaller two save as a single .weights.h5.
_SHARDED = {"gpt2_large", "gpt2_xl"}


if __name__ == "__main__":
    import gc

    import keras
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import GPT2LMHeadModel

    from zeromodels.models.gpt2 import GPT2TextGenerate

    HF_SOURCES = {
        "gpt2": "openai-community/gpt2",
        "gpt2_medium": "openai-community/gpt2-medium",
        "gpt2_large": "openai-community/gpt2-large",
        "gpt2_xl": "openai-community/gpt2-xl",
    }
    MAX_SHARD_GB = 1.7  # GitHub caps release assets at 2 GB; large/xl get sharded
    rng = np.random.default_rng(0)

    for variant, arch in GPT2_VARIANTS.items():
        hf_id = HF_SOURCES[variant]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        sd = load_file(hf_hub_download(hf_id, "model.safetensors"))
        model = GPT2TextGenerate(**arch)
        transfer_gpt2_weights(model, sd)
        del sd

        ids = rng.integers(0, arch["vocab_size"], (1, 16)).astype("int64")
        k_logits = model({"input_ids": ids.astype("int32")})["logits"]
        k_logits = (
            k_logits.detach().cpu().numpy()
            if hasattr(k_logits, "detach")
            else np.asarray(k_logits)
        )
        hf = GPT2LMHeadModel.from_pretrained(hf_id, attn_implementation="eager").eval()
        with torch.no_grad():
            hf_logits = hf(torch.from_numpy(ids)).logits.numpy()
        d = float(np.abs(hf_logits - k_logits).max())
        print(f"  logits max diff: {d:.3e}")
        if d > 1e-3:
            raise ValueError(f"{variant}: GPT2 parity failed ({d:.3e})")

        if variant in _SHARDED:
            out_path = f"{variant}.weights.json"
            model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        else:
            out_path = f"{variant}.weights.h5"
            model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del hf, model
        keras.backend.clear_session()
        gc.collect()
