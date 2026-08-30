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
from zeromodels.models.mobilevitv2 import MobileViTV2SemanticSegment

IR_SUBLAYER = {
    "conv_1": "expand_1x1",
    "dwconv": "conv_3x3",
    "conv_2": "reduce_1x1",
    "batchnorm_1": "expand_1x1.normalization",
    "batchnorm_2": "conv_3x3.normalization",
    "batchnorm_3": "reduce_1x1.normalization",
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
        return f"mobilevitv2.conv_stem.convolution.{VAR_SUFFIX[name[len('stem_conv') + 1 :]]}"
    if name.startswith("stem_batchnorm"):
        return f"mobilevitv2.conv_stem.normalization.{VAR_SUFFIX[name[len('stem_batchnorm') + 1 :]]}"

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
            prefix = f"mobilevitv2.encoder.layer.{stage}.layer.{block}.{sub}"
        else:
            prefix = f"mobilevitv2.encoder.layer.{stage}.downsampling_layer.{sub}"
        if sub.endswith(".normalization"):
            return f"{prefix}.{VAR_SUFFIX[var]}"
        return f"{prefix}.convolution.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_mv2_dwconv_kernel$", name)
    if m:
        return f"mobilevitv2.encoder.layer.{m.group(1)}.conv_kxk.convolution.weight"

    m = re.match(r"stages_(\d+)_1_mv2_batchnorm_1_(\w+)$", name)
    if m:
        return f"mobilevitv2.encoder.layer.{m.group(1)}.conv_kxk.normalization.{VAR_SUFFIX[m.group(2)]}"

    m = re.match(r"stages_(\d+)_1_mv2_conv_1_(\w+)$", name)
    if m:
        return f"mobilevitv2.encoder.layer.{m.group(1)}.conv_1x1.convolution.{VAR_SUFFIX[m.group(2)]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_attn_conv_1_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.attention.qkv_proj.convolution.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_attn_conv_2_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.attention.out_proj.convolution.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_groupnorm_1_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.layernorm_before.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_groupnorm_2_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.layernorm_after.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_mlp_conv_1_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.ffn.conv1.convolution.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_transformer_(\d+)_mlp_conv_2_(\w+)$", name)
    if m:
        stage = int(m.group(1))
        t_idx = int(m.group(2))
        var = m.group(3)
        return f"mobilevitv2.encoder.layer.{stage}.transformer.layer.{t_idx}.ffn.conv2.convolution.{VAR_SUFFIX[var]}"

    m = re.match(r"stages_(\d+)_1_groupnorm_(\w+)$", name)
    if m:
        return (
            f"mobilevitv2.encoder.layer.{m.group(1)}.layernorm.{VAR_SUFFIX[m.group(2)]}"
        )

    m = re.match(r"stages_(\d+)_1_mv2_proj_conv_kernel$", name)
    if m:
        return (
            f"mobilevitv2.encoder.layer.{m.group(1)}.conv_projection.convolution.weight"
        )

    m = re.match(r"stages_(\d+)_1_mv2_proj_batchnorm_(\w+)$", name)
    if m:
        return f"mobilevitv2.encoder.layer.{m.group(1)}.conv_projection.normalization.{VAR_SUFFIX[m.group(2)]}"

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


def transfer_mobilevitv2_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring MobileViTV2 weights"
    ):
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


MOBILEVITV2_SEGMENT_VARIANTS: List[Tuple[str, str]] = [
    ("mobilevitv2_100_deeplabv3", "apple/mobilevitv2-1.0-voc-deeplabv3"),
    ("mobilevitv2_150_deeplabv3", "apple/mobilevitv2-1.5-voc-deeplabv3"),
]


if __name__ == "__main__":
    import torch
    from transformers import MobileViTV2ForSemanticSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant, hf_id in MOBILEVITV2_SEGMENT_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting segment: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = MobileViTV2ForSemanticSegmentation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        keras_model = MobileViTV2SemanticSegment.from_weights(f"hf:{hf_id}")
        transfer_mobilevitv2_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)
        with torch.no_grad():
            hf_out = (
                hf_model(pixel_values=torch.from_numpy(x).to(device))
                .logits.cpu()
                .numpy()
            )
        k_out = keras_model(np.transpose(x, (0, 2, 3, 1)), training=False)
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
