import gc
import json

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
    transfer_weights,
)
from zeromodels.models.regnet import RegNetImageClassify

# Hosted variant -> HF (transformers) repo id, for every standard RegNet X/Y FLOP
# variant (002=0.2 GF ... 320=32 GF). The arch is read from each repo's config.json
# via RegNetImageClassify.config_from_hf, so no per-variant arch table is needed
# here; ``hf:facebook/regnet-*`` also loads any of these on the fly. (The larger
# self-supervised "-seer" checkpoints are not listed but load the same way.)
REGNET_FLOPS = (
    "002",
    "004",
    "006",
    "008",
    "016",
    "032",
    "040",
    "064",
    "080",
    "120",
    "160",
    "320",
)
REGNET_VARIANTS = {
    f"regnet_{layer_type}_{flops}": f"facebook/regnet-{layer_type}-{flops}"
    for layer_type in ("x", "y")
    for flops in REGNET_FLOPS
}

# keras weight name ``{layer.name}_{weight.name}`` -> HF (torch) name. The keras
# layers are named as the HF module path with ``.`` replaced by ``_``, so ``_ ->
# .`` alone reconstructs the path; the rest renames Keras weight suffixes.
WEIGHT_NAME_MAPPING = {
    "_": ".",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def transfer_regnet_weights(keras_model, state_dict):
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_name = keras_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)

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
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    # Compare in true float32: cuDNN TF32 on a GPU inflates the conv/BN diff to
    # ~1e-2 (HF runs on CPU), which can spuriously trip the 1e-2 parity gate on the
    # SE ("y") variants. Disabling TF32 restores the real ~5e-6 conversion fidelity.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    for variant, hf_id in REGNET_VARIANTS.items():
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        with open(hf_hub_download(hf_id, "config.json"), encoding="utf-8") as f:
            hf_config = json.load(f)
        state = download_hf_state_dict(hf_id)
        keras_model = RegNetImageClassify(
            **RegNetImageClassify.config_from_hf(hf_config),
            include_normalization=False,
        )
        transfer_regnet_weights(keras_model, state)

        hf_model = transformers.RegNetForImageClassification.from_pretrained(
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
