from typing import Dict

import numpy as np

from zeromodels.conversion.weight_transfer_util import transfer_nested_layer_weights

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def transfer_levit_weights(keras_model, state_dict) -> None:
    sd = {k: to_numpy(v) for k, v in state_dict.items()}
    prefix = "levit." if any(k.startswith("levit.") for k in sd) else ""
    unmatched = []

    def transfer(layer_name, torch_prefix, has_biases=False):
        layer = keras_model.get_layer(layer_name)
        skip_paths = ["attention_biases"] if has_biases else []
        for weight, path in transfer_nested_layer_weights(
            layer, sd, torch_prefix, WEIGHT_NAME_MAPPING, skip_paths=skip_paths
        ):
            if "attention_biases" in path:
                weight.assign(sd[f"{torch_prefix}.attention_biases"])
            else:
                unmatched.append(path)

    def exists(layer_name):
        try:
            keras_model.get_layer(layer_name)
            return True
        except ValueError:
            return False

    for i in range(1, 5):
        embed = f"{prefix}patch_embeddings.embedding_layer_{i}"
        transfer(f"patch_conv_{i}", f"{embed}.convolution")
        transfer(f"patch_bn_{i}", f"{embed}.batch_norm")

    depths = keras_model.depths
    stages = f"{prefix}encoder.stages"
    for s in range(len(depths)):
        for d in range(depths[s]):
            transfer(
                f"stage{s}_attn{d}",
                f"{stages}.{s}.layers.{2 * d}.module",
                has_biases=True,
            )
            transfer(f"stage{s}_mlp{d}", f"{stages}.{s}.layers.{2 * d + 1}.module")
        if s < len(depths) - 1:
            li = 2 * depths[s]
            transfer(
                f"stage{s}_subsample", f"{stages}.{s}.layers.{li}", has_biases=True
            )
            transfer(f"stage{s}_sub_mlp", f"{stages}.{s}.layers.{li + 1}.module")

    for head in ("classifier", "classifier_distill"):
        if exists(f"{head}_linear"):
            transfer(f"{head}_bn", f"{head}.batch_norm")
            transfer(f"{head}_linear", f"{head}.linear")

    if unmatched:
        raise ValueError(f"unmatched LeViT weights: {unmatched[:5]}")
