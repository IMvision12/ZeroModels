import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/position_embeddings/embeddings": "embeddings.position_embeddings.weight",
    "embeddings/token_type_embeddings/embeddings": "embeddings.token_type_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "embed_project/kernel": "embeddings_project.weight",
    "embed_project/bias": "embeddings_project.bias",
    # sub-block name lookups (shared by the encoder-layer regexes below)
    "attention_output_dense": "attention.output.dense",
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_output_layernorm": "attention.output.LayerNorm",
    "output_layernorm": "output.LayerNorm",
    # generator (masked-LM) head
    "generator_dense/kernel": "generator_predictions.dense.weight",
    "generator_dense/bias": "generator_predictions.dense.bias",
    "generator_layernorm/gamma": "generator_predictions.LayerNorm.weight",
    "generator_layernorm/beta": "generator_predictions.LayerNorm.bias",
    # The LM head kernel is tied to the input word embeddings (safetensors strips the
    # tied copy), so map to the embedding table; transfer_weights transposes it in.
    "generator_lm_head/kernel": "embeddings.word_embeddings.weight",
    "generator_lm_head/bias": "generator_lm_head.bias",
    # sequence-classification head
    "classifier_dense/kernel": "classifier.dense.weight",
    "classifier_dense/bias": "classifier.dense.bias",
    "classifier_out_proj/kernel": "classifier.out_proj.weight",
    "classifier_out_proj/bias": "classifier.out_proj.bias",
    # token-classification / multiple-choice classifier
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    # question-answering span head
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
    # multiple-choice sequence summary
    "summary/kernel": "sequence_summary.summary.weight",
    "summary/bias": "sequence_summary.summary.bias",
}

_OPTIONAL_HEADS = (
    "classifier",
    "qa_outputs",
    "summary",
    "generator_dense",
    "generator_layernorm",
    "generator_lm_head",
)


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention_self/blocks_\d+_(query|key|value)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.self.{proj}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_dense|intermediate_dense|output_dense)/(kernel|bias)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_layernorm|output_layernorm)/(gamma|beta)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("electra."):
        key = key[len("electra.") :]
    return key


def transfer_electra_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_HEADS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import ElectraForMaskedLM
    from transformers import ElectraModel as HFElectraModel

    from zeromodels.models.electra import ElectraMaskedLM, ElectraModel

    HF_TOKEN = os.environ.get("HF_TOKEN")

    DISCRIMINATORS = {
        "electra_small_discriminator": "google/electra-small-discriminator",
        "electra_base_discriminator": "google/electra-base-discriminator",
        "electra_large_discriminator": "google/electra-large-discriminator",
    }
    GENERATORS = {
        "electra_small_generator": "google/electra-small-generator",
        "electra_base_generator": "google/electra-base-generator",
        "electra_large_generator": "google/electra-large-generator",
    }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)

    def inputs(vocab):
        ids = rng.integers(0, vocab, (2, 16)).astype("int64")
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        types = np.zeros((2, 16), dtype="int64")
        k = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "token_type_ids": types.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
            "token_type_ids": torch.from_numpy(types),
        }
        return k, pt

    for variant, hf_id in DISCRIMINATORS.items():
        print(f"\n{'=' * 60}\n{variant}  <-  {hf_id}\n{'=' * 60}")
        hf_model = HFElectraModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        arch = ElectraModel.config_from_hf(hf_model.config.to_dict())
        keras_model = ElectraModel(**arch)
        transfer_electra_weights(keras_model, dict(hf_model.state_dict()))
        k, pt = inputs(arch["vocab_size"])
        with torch.no_grad():
            hf_out = hf_model(**pt).last_hidden_state.detach().cpu().numpy()
        seq = keras_model(k, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        print(f"  last_hidden_state max diff: {np.abs(hf_out - seq).max():.3e}")
        del hf_model, keras_model
        keras.backend.clear_session()
        gc.collect()

    for variant, hf_id in GENERATORS.items():
        print(f"\n{'=' * 60}\n{variant}  <-  {hf_id}\n{'=' * 60}")
        hf_mlm = ElectraForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        arch = ElectraModel.config_from_hf(hf_mlm.config.to_dict())
        keras_mlm = ElectraMaskedLM(**arch)
        transfer_electra_weights(keras_mlm, dict(hf_mlm.state_dict()))
        k, pt = inputs(arch["vocab_size"])
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits.detach().cpu().numpy()
        out = keras_mlm(k, training=False)
        out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        print(f"  mlm logits max diff: {np.abs(hf_logits - out).max():.3e}")
        del hf_mlm, keras_mlm
        keras.backend.clear_session()
        gc.collect()
