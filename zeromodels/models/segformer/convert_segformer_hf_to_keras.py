import gc
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
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.segformer import SegFormerSemanticSegment

weight_name_mapping: Dict[str, str] = {
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

attn_name_replace: Dict[str, str] = {
    "block": "segformer.encoder.block",
    "attn.q": "attention.self.query",
    "attn.k": "attention.self.key",
    "attn.v": "attention.self.value",
    "attn.proj": "attention.output.dense",
    "attn.sr": "attention.self.sr",
    "attn.norm": "attention.self.layer_norm",
}


def transfer_segformer_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    backbone_weights = list(split_model_weights(keras_model.backbone))
    trainable, non_trainable = backbone_weights

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable,
        desc="Transferring SegFormer backbone weights",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_part, torch_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(keras_part, torch_part)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name,
                keras_weight,
                hf_state_dict,
                attn_name_replace,
            )
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]
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

    for i in range(4):
        layer = keras_model.get_layer(f"head_linear_c{i + 1}")
        layer.weights[0].assign(
            hf_state_dict[f"decode_head.linear_c.{i}.proj.weight"].T
        )
        layer.weights[1].assign(hf_state_dict[f"decode_head.linear_c.{i}.proj.bias"])

    fusion_conv = keras_model.get_layer("head_fusion_conv")
    fusion_conv.weights[0].assign(
        np.transpose(hf_state_dict["decode_head.linear_fuse.weight"], (2, 3, 1, 0))
    )

    bn = keras_model.get_layer("head_fusion_bn")
    bn.weights[0].assign(hf_state_dict["decode_head.batch_norm.weight"])
    bn.weights[1].assign(hf_state_dict["decode_head.batch_norm.bias"])
    bn.weights[2].assign(hf_state_dict["decode_head.batch_norm.running_mean"])
    bn.weights[3].assign(hf_state_dict["decode_head.batch_norm.running_var"])

    classifier = keras_model.get_layer("head_classifier")
    classifier.weights[0].assign(
        np.transpose(hf_state_dict["decode_head.classifier.weight"], (2, 3, 1, 0))
    )
    classifier.weights[1].assign(hf_state_dict["decode_head.classifier.bias"])


SEGFORMER_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    (
        "segformer_b0_cityscapes_1024",
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b0_cityscapes_768", "nvidia/segformer-b0-finetuned-cityscapes-768-768"),
    ("segformer_b0_ade_512", "nvidia/segformer-b0-finetuned-ade-512-512"),
    (
        "segformer_b1_cityscapes_1024",
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b1_ade_512", "nvidia/segformer-b1-finetuned-ade-512-512"),
    (
        "segformer_b2_cityscapes_1024",
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b2_ade_512", "nvidia/segformer-b2-finetuned-ade-512-512"),
    (
        "segformer_b3_cityscapes_1024",
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b3_ade_512", "nvidia/segformer-b3-finetuned-ade-512-512"),
    (
        "segformer_b4_cityscapes_1024",
        "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b4_ade_512", "nvidia/segformer-b4-finetuned-ade-512-512"),
    (
        "segformer_b5_cityscapes_1024",
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    ),
    ("segformer_b5_ade_640", "nvidia/segformer-b5-finetuned-ade-640-640"),
]


if __name__ == "__main__":
    import torch
    from transformers import SegformerForSemanticSegmentation

    for variant, hf_id in SEGFORMER_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model: keras.Model = SegFormerSemanticSegment.from_weights(
            variant, load_weights=False
        )
        hf_model = SegformerForSemanticSegmentation.from_pretrained(hf_id).eval()
        hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        transfer_segformer_weights(keras_model, hf_state_dict)

        np.random.seed(42)
        image_size = keras_model.image_size
        test_input = np.random.rand(1, *image_size).astype(np.float32)
        hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
        with torch.no_grad():
            hf_output = hf_model(pixel_values=hf_input).logits.numpy()
        hf_output = np.transpose(hf_output, (0, 2, 3, 1))

        classifier_layer = keras_model.get_layer("head_classifier")
        sub_model = keras.Model(keras_model.input, classifier_layer.output)
        keras_output = np.array(sub_model.predict(test_input, verbose=0))

        max_diff = float(np.max(np.abs(hf_output - keras_output)))
        print(f"  Max logits diff: {max_diff:.6f}")
        if max_diff > 1e-3:
            raise ValueError(
                f"{variant}: max diff {max_diff:.6f} exceeds 1e-3 tolerance"
            )
        print("  Verification OK")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
