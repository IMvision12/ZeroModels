import gc
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.mit import MiTImageClassify
from zeromodels.models.mit.mit_config import MIT_VARIANTS

# Architecture presets, moved here from mit_config.py: the package config no longer
# carries arch (models load by Hub repo id / zm_config). Only this converter builds an
# untrained model to transfer the SegFormer encoder weights into.
MIT_MODEL_CONFIG = {
    "mit_b0": {
        "embed_dim": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b1": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b2": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b3": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b4": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b5": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "block": "segformer.encoder.block",
    "patch.embed": "segformer.encoder.patch_embeddings",
    "layernorm": "layer_norm",
    "layer_norm.1": "layer_norm_1",
    "layer_norm.2": "layer_norm_2",
    "conv.proj": "proj",
    "dense.1": "dense1",
    "dense.2": "dense2",
    "dwconv": "dwconv.dwconv",
    "final": "segformer.encoder",
    "segformer.encoder.layer_norm_1": "segformer.encoder.layer_norm.1",
    "segformer.encoder.layer_norm_2": "segformer.encoder.layer_norm.2",
    "segformer.encoder.layer_norm_3": "segformer.encoder.layer_norm.3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "classifier",
}

_ATTN_REPLACEMENT: Dict[str, str] = {
    "block": "segformer.encoder.block",
    "attn.q": "attention.self.query",
    "attn.k": "attention.self.key",
    "attn.v": "attention.self.value",
    "attn.proj": "attention.output.dense",
    "attn.sr": "attention.self.sr",
    "attn.norm": "attention.self.layer_norm",
}


def transfer_mit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state_dict, _ATTN_REPLACEMENT
            )
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import transformers

    for variant, meta in MIT_VARIANTS.items():
        hf_id = meta["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(hf_id)
        keras_model = MiTImageClassify(
            **MIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_mit_weights(keras_model, state)

        hf_model = transformers.SegformerForImageClassification.from_pretrained(
            hf_id
        ).eval()
        results = verify_cls_model_equivalence(
            model_a=hf_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="hf_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, hf_model
        keras.backend.clear_session()
        gc.collect()
