import gc
import re
from typing import Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.mobilevit import MobileViTSemanticSegment

IR_SUBLAYER = {
    "conv_1": "expand_1x1",
    "dwconv": "conv_3x3",
    "conv_2": "reduce_1x1",
    "batchnorm_1": "expand_1x1.normalization",
    "batchnorm_2": "conv_3x3.normalization",
    "batchnorm_3": "reduce_1x1.normalization",
}

MV_SUBLAYER = {
    "conv_1": "conv_kxk",
    "conv_2": "conv_1x1",
    "conv_3": "conv_projection",
    "conv_4": "fusion",
    "batchnorm_1": "conv_kxk.normalization",
    "batchnorm_2": "conv_projection.normalization",
    "batchnorm_3": "fusion.normalization",
}

VAR_SUFFIX = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}

ASPP_DILATED_PREFIXES = {
    "seg_aspp_conv1_1x1": "segmentation_head.aspp.convs.0",
    "seg_aspp_conv2_3x3": "segmentation_head.aspp.convs.1",
    "seg_aspp_conv3_3x3": "segmentation_head.aspp.convs.2",
    "seg_aspp_conv4_3x3": "segmentation_head.aspp.convs.3",
    "seg_aspp_pool_1x1": "segmentation_head.aspp.convs.4.conv_1x1",
    "seg_aspp_project": "segmentation_head.aspp.project",
}


def map_keras_to_hf(name: str) -> str:
    if name.startswith("stem_conv"):
        return f"mobilevit.conv_stem.convolution.{VAR_SUFFIX[name[len('stem_conv') + 1 :]]}"
    if name.startswith("stem_batchnorm"):
        suf = name[len("stem_batchnorm") + 1 :]
        return f"mobilevit.conv_stem.normalization.{VAR_SUFFIX[suf]}"

    m = re.match(r"stages_(\d+)_(\d+)_ir_(\w+?)(?:_(\w+))?$", name)
    if m:
        stage = int(m.group(1))
        block = int(m.group(2))
        head = m.group(3)
        rest = m.group(4)
        token = head if rest is None else f"{head}_{rest}"
        var = ""
        for v in VAR_SUFFIX:
            if token.endswith(v):
                var = v
                token = (
                    token[: -len(v) - 1]
                    if token.endswith(f"_{v}")
                    else token[: -len(v)]
                )
                break
        sub = IR_SUBLAYER.get(token)
        if sub is None:
            raise WeightMappingError(name, f"unknown IR sublayer {token}")
        if stage < 2:
            prefix = f"mobilevit.encoder.layer.{stage}.layer.{block}.{sub}"
        else:
            prefix = f"mobilevit.encoder.layer.{stage}.downsampling_layer.{sub}"
        if sub.endswith(".normalization"):
            return f"{prefix}.{VAR_SUFFIX[var]}"
        return f"{prefix}.convolution.{VAR_SUFFIX[var]}"

    m = re.match(
        r"stages_(\d+)_1_transformer_(\d+)_(layernorm_1|layernorm_2|mlp_fc1|mlp_fc2)_(\w+)$",
        name,
    )
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        kind = m.group(3)
        var = m.group(4)
        hf_kind = {
            "layernorm_1": "layernorm_before",
            "layernorm_2": "layernorm_after",
            "mlp_fc1": "intermediate.dense",
            "mlp_fc2": "output.dense",
        }[kind]
        return (
            f"mobilevit.encoder.layer.{stage}.transformer.layer.{t_idx}."
            f"{hf_kind}.{VAR_SUFFIX[var]}"
        )

    m = re.match(r"stages_(\d+)_1_layernorm_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        var = m.group(2)
        return f"mobilevit.encoder.layer.{stage}.layernorm.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_mv_(\w+?)(?:_(\d+))?_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        head = m.group(2)
        idx = m.group(3)
        var = m.group(4)
        token = head if idx is None else f"{head}_{idx}"
        sub = MV_SUBLAYER.get(token)
        if sub is None:
            raise WeightMappingError(name, f"unknown MV sublayer {token}")
        prefix = f"mobilevit.encoder.layer.{stage}.{sub}"
        if sub.endswith(".normalization"):
            return f"{prefix}.{VAR_SUFFIX[var]}"
        return f"{prefix}.convolution.{VAR_SUFFIX[var]}"

    if name.startswith("final_conv"):
        return f"mobilevit.conv_1x1_exp.convolution.{VAR_SUFFIX[name[len('final_conv') + 1 :]]}"
    if name.startswith("final_batchnorm"):
        return f"mobilevit.conv_1x1_exp.normalization.{VAR_SUFFIX[name[len('final_batchnorm') + 1 :]]}"
    if name.startswith("predictions"):
        return f"classifier.{VAR_SUFFIX[name[len('predictions') + 1 :]]}"

    for keras_prefix, hf_prefix in ASPP_DILATED_PREFIXES.items():
        if name.startswith(keras_prefix + "_conv"):
            return (
                f"{hf_prefix}.convolution.{VAR_SUFFIX[name[len(keras_prefix) + 6 :]]}"
            )
        if name.startswith(keras_prefix + "_batchnorm"):
            return f"{hf_prefix}.normalization.{VAR_SUFFIX[name[len(keras_prefix) + 11 :]]}"

    if name.startswith("seg_classifier_conv_"):
        return f"segmentation_head.classifier.convolution.{VAR_SUFFIX[name[len('seg_classifier_conv_') :]]}"

    raise WeightMappingError(name, "no HF mapping rule matched")


def transfer_mobilevit_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring MobileViT weights"
    ):
        path = keras_weight.path

        m = re.match(
            r".+/stages_(\d+)_1_transformer_(\d+)_attn_(qkv|proj)/(kernel|bias)$",
            path,
        )
        if m:
            stage = int(m.group(1))
            t_idx = int(m.group(2))
            kind = m.group(3)
            var = m.group(4)
            torch_var = "weight" if var == "kernel" else "bias"
            base = f"mobilevit.encoder.layer.{stage}.transformer.layer.{t_idx}"
            if kind == "qkv":
                q = hf_state_dict[f"{base}.attention.attention.query.{torch_var}"]
                k = hf_state_dict[f"{base}.attention.attention.key.{torch_var}"]
                v = hf_state_dict[f"{base}.attention.attention.value.{torch_var}"]
                if hasattr(q, "numpy"):
                    q, k, v = q.numpy(), k.numpy(), v.numpy()
                torch_weight = np.concatenate([q, k, v], axis=0)
            else:
                torch_weight = hf_state_dict[
                    f"{base}.attention.output.dense.{torch_var}"
                ]
                if hasattr(torch_weight, "numpy"):
                    torch_weight = torch_weight.numpy()
            transfer_weights(keras_weight_name, keras_weight, torch_weight)
            continue

        hf_key = map_keras_to_hf(keras_weight_name)
        if hf_key not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, hf_key)

        torch_weight = hf_state_dict[hf_key]
        if hasattr(torch_weight, "numpy"):
            torch_weight = torch_weight.numpy()

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, hf_key, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                hf_key,
                torch_weight.shape,
            )
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


# MobileViT classification weights come from timm (see
# convert_mobilevit_timm_to_keras.py); only the DeepLabV3 segmentation
# checkpoints are converted from the source here.
MOBILEVIT_SEGMENT_VARIANTS: List[Tuple[str, str]] = [
    ("mobilevit_xxs_deeplabv3", "apple/deeplabv3-mobilevit-xx-small"),
    ("mobilevit_xs_deeplabv3", "apple/deeplabv3-mobilevit-x-small"),
    ("mobilevit_s_deeplabv3", "apple/deeplabv3-mobilevit-small"),
]


if __name__ == "__main__":
    import torch
    from transformers import MobileViTForSemanticSegmentation

    # Run the reference model on the same device Keras (torch backend) uses. On a
    # GPU, comparing GPU-Keras against the CPU reference lets cuDNN-vs-CPU float divergence
    # compound through MobileViT's deep conv stack and blows up the unnormalized
    # logits (O(1) diffs that look like a conversion bug but aren't).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant, hf_id in MOBILEVIT_SEGMENT_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting segment: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = MobileViTForSemanticSegmentation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        keras_model = MobileViTSemanticSegment.from_weights(f"hf:{hf_id}")
        transfer_mobilevit_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)
        with torch.no_grad():
            hf_out = (
                hf_model(pixel_values=torch.from_numpy(x).to(device))
                .logits.cpu()
                .numpy()
            )
        k_in = np.transpose(x, (0, 2, 3, 1))
        k_out = keras_model(k_in, training=False)
        if hasattr(k_out, "detach"):
            k_out = k_out.detach().cpu().numpy()
        else:
            k_out = np.asarray(k_out)
        k_out = np.transpose(k_out, (0, 3, 1, 2))
        diff = float(np.abs(k_out - hf_out).max())
        print(f"  max diff: {diff:.6e}")

        keras_model.save_weights(f"{variant}.weights.h5")
        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
