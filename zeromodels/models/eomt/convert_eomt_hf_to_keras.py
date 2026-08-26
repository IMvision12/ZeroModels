import gc
from typing import Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.eomt import EoMTUniversalSegment

weight_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_eomt_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    emb = keras_model.get_layer("embeddings")
    transfer_weights(
        "conv_kernel",
        emb.patch_embeddings.projection.weights[0],
        hf_state_dict["embeddings.patch_embeddings.projection.weight"],
    )
    emb.patch_embeddings.projection.weights[1].assign(
        hf_state_dict["embeddings.patch_embeddings.projection.bias"]
    )
    emb.cls_token.assign(hf_state_dict["embeddings.cls_token"])
    emb.register_tokens.assign(hf_state_dict["embeddings.register_tokens"])
    pos_emb = hf_state_dict["embeddings.position_embeddings.weight"]
    emb.position_embeddings.assign(np.expand_dims(pos_emb, axis=0))

    query_layer = keras_model.get_layer("query")
    query_layer.query_weight.assign(hf_state_dict["query.weight"])

    num_layers = keras_model.num_hidden_layers
    use_swiglu = keras_model.use_swiglu_ffn
    for i in tqdm(range(num_layers), desc="Transferring EoMT layers"):
        hf_prefix = f"layers.{i}"
        k_prefix = f"layers_{i}"

        norm1 = keras_model.get_layer(f"{k_prefix}_norm1")
        norm1.weights[0].assign(hf_state_dict[f"{hf_prefix}.norm1.weight"])
        norm1.weights[1].assign(hf_state_dict[f"{hf_prefix}.norm1.bias"])

        attn = keras_model.get_layer(f"{k_prefix}_attention")
        transfer_nested_layer_weights(
            attn,
            hf_state_dict,
            f"{hf_prefix}.attention",
            name_mapping=weight_name_mapping,
        )

        ls1 = keras_model.get_layer(f"{k_prefix}_layer_scale1")
        ls1.weights[0].assign(hf_state_dict[f"{hf_prefix}.layer_scale1.lambda1"])

        norm2 = keras_model.get_layer(f"{k_prefix}_norm2")
        norm2.weights[0].assign(hf_state_dict[f"{hf_prefix}.norm2.weight"])
        norm2.weights[1].assign(hf_state_dict[f"{hf_prefix}.norm2.bias"])

        if not use_swiglu:
            fc1 = keras_model.get_layer(f"{k_prefix}_mlp_fc1")
            transfer_weights(
                "kernel", fc1.weights[0], hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"]
            )
            fc1.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"])
            fc2 = keras_model.get_layer(f"{k_prefix}_mlp_fc2")
            transfer_weights(
                "kernel", fc2.weights[0], hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"]
            )
            fc2.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"])
        else:
            w_in = keras_model.get_layer(f"{k_prefix}_mlp_weights_in")
            transfer_weights(
                "kernel",
                w_in.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.weights_in.weight"],
            )
            w_in.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.weights_in.bias"])
            w_out = keras_model.get_layer(f"{k_prefix}_mlp_weights_out")
            transfer_weights(
                "kernel",
                w_out.weights[0],
                hf_state_dict[f"{hf_prefix}.mlp.weights_out.weight"],
            )
            w_out.weights[1].assign(hf_state_dict[f"{hf_prefix}.mlp.weights_out.bias"])

        ls2 = keras_model.get_layer(f"{k_prefix}_layer_scale2")
        ls2.weights[0].assign(hf_state_dict[f"{hf_prefix}.layer_scale2.lambda1"])

    layernorm = keras_model.get_layer("layernorm")
    layernorm.weights[0].assign(hf_state_dict["layernorm.weight"])
    layernorm.weights[1].assign(hf_state_dict["layernorm.bias"])

    class_pred = keras_model.get_layer("class_predictor")
    transfer_weights(
        "kernel", class_pred.weights[0], hf_state_dict["class_predictor.weight"]
    )
    class_pred.weights[1].assign(hf_state_dict["class_predictor.bias"])

    for fc_name in ["fc1", "fc2", "fc3"]:
        fc = keras_model.get_layer(f"mask_head_{fc_name}")
        transfer_weights(
            "kernel", fc.weights[0], hf_state_dict[f"mask_head.{fc_name}.weight"]
        )
        fc.weights[1].assign(hf_state_dict[f"mask_head.{fc_name}.bias"])

    for block_idx in range(keras_model.num_upscale_blocks):
        hf_block_prefix = f"upscale_block.block.{block_idx}"
        k_block_prefix = f"upscale_block_{block_idx}"

        conv1 = keras_model.get_layer(f"{k_block_prefix}_conv1")
        transfer_weights(
            "conv_kernel",
            conv1.weights[0],
            hf_state_dict[f"{hf_block_prefix}.conv1.weight"],
        )
        conv1.weights[1].assign(hf_state_dict[f"{hf_block_prefix}.conv1.bias"])

        conv2 = keras_model.get_layer(f"{k_block_prefix}_conv2")
        transfer_weights(
            "dwconv_kernel",
            conv2.weights[0],
            hf_state_dict[f"{hf_block_prefix}.conv2.weight"],
        )

        ln = keras_model.get_layer(f"{k_block_prefix}_layernorm")
        ln.weights[0].assign(hf_state_dict[f"{hf_block_prefix}.layernorm2d.weight"])
        ln.weights[1].assign(hf_state_dict[f"{hf_block_prefix}.layernorm2d.bias"])


EOMT_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    ("eomt_small_coco_panoptic_640", "tue-mps/coco_panoptic_eomt_small_640_2x"),
    ("eomt_base_coco_panoptic_640", "tue-mps/coco_panoptic_eomt_base_640_2x"),
    ("eomt_large_coco_panoptic_640", "tue-mps/coco_panoptic_eomt_large_640"),
    ("eomt_large_coco_instance_640", "tue-mps/coco_instance_eomt_large_640"),
    ("eomt_large_ade20k_semantic_512", "tue-mps/ade20k_semantic_eomt_large_512"),
]


if __name__ == "__main__":
    import torch
    from transformers import EomtForUniversalSegmentation

    for variant, hf_id in EOMT_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model: keras.Model = EoMTUniversalSegment.from_weights(
            variant, load_weights=False
        )
        # Run the reference model on the same device Keras (torch backend) uses,
        # so the equivalence check reflects the conversion, not GPU-vs-CPU
        # float accumulation, which inflates EoMT's raw mask logits on GPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model = EomtForUniversalSegmentation.from_pretrained(hf_id).eval().to(device)
        hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        transfer_eomt_weights(keras_model, hf_state_dict)

        np.random.seed(42)
        image_size = keras_model.image_size
        test_input = np.random.rand(1, *image_size).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        normalized_input = (test_input - mean) / std

        hf_input = torch.tensor(normalized_input).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            hf_output = hf_model(pixel_values=hf_input)
            hf_class_logits = hf_output.class_queries_logits.cpu().numpy()
            hf_mask_logits = hf_output.masks_queries_logits.cpu().numpy()

        keras_output = keras_model(normalized_input.astype(np.float32), training=False)
        keras_class_logits = keras.ops.convert_to_numpy(keras_output["class_logits"])
        keras_mask_logits = keras.ops.convert_to_numpy(keras_output["mask_logits"])

        class_diff = float(np.max(np.abs(hf_class_logits - keras_class_logits)))
        mask_diff = float(np.max(np.abs(hf_mask_logits - keras_mask_logits)))
        print(f"  Max class logits diff: {class_diff:.6f}")
        print(f"  Max mask logits diff:  {mask_diff:.6f}")
        # GPU conv/upsample float accumulation inflates the raw mask logits well
        # beyond CPU (the transfer itself matches to ~2e-4 on CPU), so allow a
        # looser mask tolerance on CUDA. The class path is robust on both.
        mask_tol = 3e-2 if device == "cuda" else 5e-3
        if class_diff > 1e-3 or mask_diff > mask_tol:
            raise ValueError(f"{variant}: class {class_diff:.6f}, mask {mask_diff:.6f}")
        print("  Verification OK")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
