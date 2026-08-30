import gc

import keras
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
from zeromodels.models.pvt_v2 import PvtV2ImageClassify
from zeromodels.models.pvt_v2.pvt_v2_config import PVT_V2_VARIANTS

PVT_V2_MODEL_CONFIG = {
    "pvt_v2_b0": {
        "hidden_sizes": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b1": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (2, 2, 2, 2),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b2": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (3, 4, 6, 3),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b2_linear": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (3, 4, 6, 3),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b3": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (3, 4, 18, 3),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b4": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (3, 8, 27, 3),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (8, 8, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pvt_v2_b5": {
        "hidden_sizes": (64, 128, 320, 512),
        "depths": (3, 6, 40, 3),
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
        "mlp_ratios": (4, 4, 4, 4),
        "linear_attention": False,
        "image_size": 224,
        "num_classes": 1000,
    },
}

WEIGHT_NAME_MAPPING = {
    "_": ".",
    "layers": "pvt_v2.encoder.layers",
    "patch.embed": "patch_embedding",
    "mlp.dwconv": "mlp.dwconv.dwconv",
    "layernorm.1": "layer_norm_1",
    "layernorm.2": "layer_norm_2",
    "layernorm": "layer_norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "predictions": "classifier",
}

ATTENTION_NAME_MAPPING = {
    "layers": "pvt_v2.encoder.layers",
    "attn.query": "attention.query",
    "attn.key": "attention.key",
    "attn.value": "attention.value",
    "attn.proj": "attention.proj",
    "attn.sr": "attention.spatial_reduction",
    "attn.norm": "attention.layer_norm",
}


def transfer_pvt_v2_weights(keras_model, state_dict):
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_name = keras_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)

        if "attention" in torch_name:
            transfer_attention_weights(
                keras_name, keras_weight, state_dict, ATTENTION_NAME_MAPPING
            )
            continue

        if torch_name not in state_dict:
            raise WeightMappingError(keras_name, torch_name)

        torch_weight = state_dict[torch_name]
        if not compare_keras_torch_names(
            keras_name, keras_weight, torch_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_name, keras_weight.shape, torch_name, torch_weight.shape
            )
        transfer_weights(keras_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import importlib.metadata as _meta

    _orig_version = _meta.version
    _meta.version = lambda name: (
        "0.23.0" if name == "tokenizers" else _orig_version(name)
    )
    import transformers

    for variant, meta in PVT_V2_VARIANTS.items():
        hf_id = meta["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(hf_id)
        keras_model = PvtV2ImageClassify(**PVT_V2_MODEL_CONFIG[meta["model"]])
        transfer_pvt_v2_weights(keras_model, state)

        hf_model = transformers.PvtV2ForImageClassification.from_pretrained(
            hf_id
        ).eval()
        results = verify_cls_model_equivalence(
            model_a=hf_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="hf_to_keras",
            run_performance=False,
            atol=1e-2,
            rtol=1e-2,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, hf_model
        keras.backend.clear_session()
        gc.collect()
