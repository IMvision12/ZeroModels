"""
Weight Transfer Utility for Converting PyTorch Model Weights to Keras

This module provides utility functions for transferring weights between PyTorch and Keras
neural network layers, handling various layer types and weight transformations.

Key Features:
- Supports conversion of weights for different layer types:
  - Convolutional layers (1D and 2D)
  - Dense/Linear layers
  - RNN layers (LSTM and GRU)
  - Embedding layers
  - Layer Normalization
  - Attention mechanisms

Dependencies:
- numpy
- keras
- torch

Example:
    # Assuming you have PyTorch and Keras model weights
    transfer_weights(
        keras_name='conv1_weights',
        keras_weight=keras_model.layers[0].weights[0],
        torch_weight=torch_model.conv1.weight.numpy()
    )
"""

from __future__ import annotations

import contextlib
import inspect
import re
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import keras
import numpy as np

if TYPE_CHECKING:
    import torch

_skip_state: Dict[str, Any] = {"active": False, "skipped": []}


@contextlib.contextmanager
def skip_mismatched_weights(enabled: bool = True):
    """Context manager enabling skip-on-shape-mismatch for weight transfers.

    Args:
        enabled: When ``False`` this is a no-op (normal strict transfer).

    Yields:
        A list that is populated with the names of any weights skipped because
        their shapes did not match the source tensor.
    """
    if not enabled:
        yield []
        return
    prev_active, prev_skipped = _skip_state["active"], _skip_state["skipped"]
    _skip_state["active"], _skip_state["skipped"] = True, []
    try:
        yield _skip_state["skipped"]
    finally:
        _skip_state["active"], _skip_state["skipped"] = prev_active, prev_skipped


@contextlib.contextmanager
def zeros_init():
    """Force **trainable** weights to a zeros initializer for the scope.

    A strict pretrained load overwrites every trainable weight, so the default
    glorot / normal initializer is wasted RNG (nontrivial for a multi-billion-
    parameter model). This monkeypatches ``keras.layers.Layer.add_weight`` at the
    class level to swap ``initializer="zeros"`` for ``trainable=True`` weights
    only, and restores it in ``finally``. Non-trainable buffers (rope inv_freq,
    causal masks, batch-norm stats) keep their real initializer so computed
    constants are never clobbered.

    Use ONLY on a strict load (``skip_mismatch=False``): a trainable weight
    missing from the checkpoint would otherwise stay silently zeroed, so the
    transfer must raise on a missing weight instead. Already-built models issue
    no ``add_weight`` calls in transfer, so this is a harmless no-op for them.
    """
    original = keras.layers.Layer.add_weight
    sig = inspect.signature(original)

    def add_weight(self, *args, **kwargs):
        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        if bound.arguments.get("trainable", True):
            bound.arguments["initializer"] = "zeros"
            return original(*bound.args, **bound.kwargs)
        return original(self, *args, **kwargs)

    keras.layers.Layer.add_weight = add_weight
    try:
        yield
    finally:
        keras.layers.Layer.add_weight = original


def shape_count_mismatch(keras_weight: Any, torch_weight: Any) -> bool:
    """True if element counts differ (beyond the benign scalar/size-1 cases)."""
    keras_size = int(np.prod(keras_weight.shape))
    torch_shape = getattr(torch_weight, "shape", None) or np.shape(torch_weight)
    torch_size = int(np.prod(torch_shape))
    if keras_size == torch_size:
        return False
    if (
        (keras_size == 0 and torch_size == 1)
        or (keras_size == 1 and torch_size > 1)
        or (torch_size == 1 and keras_size > 1)
    ):
        return False
    return True


class WeightType(Enum):
    KERNEL = ("kernel", "weight")
    BIAS = ("bias", "bias")

    GAMMA = ("gamma", "weight")
    BETA = ("beta", "bias")
    MOVING_MEAN = ("moving_mean", "running_mean")
    MOVING_VARIANCE = ("moving_variance", "running_var")

    QUERY = ("query", "q_proj")
    KEY = ("key", "k_proj")
    VALUE = ("value", "v_proj")
    ATTENTION = ("attention", "attn")
    FFN = ("dense", "fc")
    OUTPUT = ("output", "out_proj")

    LAYER_NORM_GAMMA = ("layer_norm_gamma", "weight")
    LAYER_NORM_BETA = ("layer_norm_beta", "bias")

    EMBED_TOKEN = ("embed_token", "embed_tokens")
    EMBED_POS = ("embed_positions", "pos_embed")
    EMBED_PATCH = ("patch_embed", "patch_embed")

    CONV_KERNEL = ("conv_kernel", "conv.weight")
    CONV_BIAS = ("conv_bias", "conv.bias")

    POOL = ("pool", "pool")
    PROJ = ("projection", "proj")

    @classmethod
    def find_weight_type(cls, keras_name: str) -> Optional["WeightType"]:
        """Find the matching weight type for a given Keras weight name."""
        for weight_type in cls:
            if keras_name.endswith(weight_type.value[0]):
                return weight_type
        return None


class WeightMismatchError(Exception):
    """Custom exception for weight comparison mismatches."""

    pass


def validate_input_weights(
    keras_weight: Any, torch_weight: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    import torch

    # Ensure torch_weight is numpy array
    if isinstance(torch_weight, torch.Tensor):
        torch_weight = torch_weight.numpy()

    keras_shape = keras_weight.shape
    torch_shape = torch_weight.shape

    if not keras_shape or not torch_shape:
        raise ValueError(
            f"Empty shapes not allowed. Keras shape: {keras_shape}, "
            f"Torch shape: {torch_shape}"
        )

    return torch_weight, keras_shape, torch_shape


def transform_conv_weights(
    keras_name: str,
    torch_weight: np.ndarray,
) -> np.ndarray:
    if "conv1d" in keras_name.lower():  # [width, in_channels, out_channels]
        return np.transpose(torch_weight, [2, 1, 0])

    elif any(
        substring in keras_name.lower()
        for substring in [
            "depthwise",
            "dwconv2d",
            "dwconv",
        ]
    ):
        return np.transpose(torch_weight, [2, 3, 0, 1])

    elif any(
        substring in keras_name.lower()
        for substring in ["conv", "conv2d", "pointwise", "downsample", "sr"]
    ):
        # Standard 2D convolution
        return np.transpose(torch_weight, [2, 3, 1, 0])

    elif "grn" in keras_name:  # For ConvNextV2
        return np.expand_dims(
            np.expand_dims(np.expand_dims(torch_weight, axis=0), axis=0), axis=0
        )


def transform_dense_weights(
    keras_name: str, torch_weight: np.ndarray, keras_shape: Tuple[int, ...]
) -> np.ndarray:
    if "se" in keras_name and torch_weight.ndim == 4:  # SE block
        torch_weight = torch_weight.squeeze()

    if keras_shape[1] == torch_weight.shape[0]:
        return np.transpose(torch_weight)

    raise ValueError(
        f"Shape mismatch in Dense/SE layer {keras_name}. "
        f"Keras shape={keras_shape}, Torch shape={torch_weight.shape}"
    )


def transform_rnn_weights(
    keras_name: str, torch_weight: np.ndarray, rnn_type: str
) -> np.ndarray:
    if not ("kernel" in keras_name or "recurrent_kernel" in keras_name):
        return torch_weight

    if rnn_type == "lstm":
        split_size = torch_weight.shape[1] // 4
        return np.concatenate(
            [
                torch_weight[:, :split_size],  # input gate
                torch_weight[:, split_size : 2 * split_size],  # forget gate
                torch_weight[:, -split_size:],  # cell gate
                torch_weight[:, 2 * split_size : -split_size],  # output gate
            ],
            axis=1,
        )

    elif rnn_type == "gru":
        split_size = torch_weight.shape[1] // 3
        return np.concatenate(
            [
                torch_weight[:, split_size : 2 * split_size],  # reset gate
                torch_weight[:, :split_size],  # update gate
                torch_weight[:, 2 * split_size :],  # new gate
            ],
            axis=1,
        )

    raise ValueError(f"Unsupported RNN type: {rnn_type}")


def transfer_weights(
    keras_name: str, keras_weight: keras.Variable, torch_weight: np.ndarray
) -> None:
    """
    Transfer weights from PyTorch to Keras based on layer type.

    Handles weight transformation for various neural network layer types.

    Args:
        keras_name (str): Name of the Keras weight for type detection.
        keras_weight (keras.Variable): Keras weight variable to update.
        torch_weight (np.ndarray): PyTorch weight tensor to transfer.

    Raises:
        ValueError: If the layer type or weight shapes are unsupported.
    """
    if _skip_state["active"] and shape_count_mismatch(keras_weight, torch_weight):
        _skip_state["skipped"].append(keras_name)
        return

    torch_weight, keras_shape, torch_shape = validate_input_weights(
        keras_weight, torch_weight
    )

    if len(keras_shape) == 4:  # conv2d, depthwise conv
        transformed = transform_conv_weights(keras_name, torch_weight)

    elif len(keras_shape) == 2:
        if "embedding" in keras_name.lower():
            transformed = torch_weight
        else:
            transformed = transform_dense_weights(keras_name, torch_weight, keras_shape)

    elif len(keras_shape) == 1:
        if "layernorm" in keras_name.lower():
            if any(
                x in keras_name.lower() for x in ["gamma", "weight", "beta", "bias"]
            ):
                transformed = torch_weight
        elif "bias" in keras_name.lower() or "batchnorm" in keras_name.lower():
            transformed = torch_weight
        elif keras_shape == torch_shape:
            transformed = torch_weight
        else:
            raise ValueError(
                f"Shape mismatch in 1D weight {keras_name}. "
                f"Keras shape={keras_shape}, Torch shape={torch_shape}"
            )

    elif keras_shape == torch_shape:
        transformed = torch_weight

    elif len(keras_shape) == 0 and len(torch_shape) == 1:
        transformed = torch_weight[0]

    elif "lstm" in keras_name.lower():
        transformed = transform_rnn_weights(keras_name, torch_weight, "lstm")

    elif "gru" in keras_name.lower():
        transformed = transform_rnn_weights(keras_name, torch_weight, "gru")

    else:
        raise ValueError(
            f"Unsupported layer type or shape mismatch for {keras_name}. "
            f"Keras shape={keras_shape}, Torch shape={torch_shape}"
        )

    keras_weight.assign(transformed)


def transfer_attention_weights(
    keras_name: str,
    keras_weight: keras.Variable,
    torch_weights_dict: Dict[str, torch.Tensor],
    name_replacements: Dict[str, str] = None,
) -> None:
    """
    Transfer attention mechanism weights from PyTorch to Keras.

    Maps PyTorch attention layer weights to corresponding Keras weights.

    Args:
        keras_name (str): Name of the Keras weight.
        keras_weight (keras.Variable): Keras weight variable to update.
        torch_weight_name (str): Name of the corresponding PyTorch weight.
        torch_weights_dict (Dict[str, torch.Tensor]): Dictionary of PyTorch weights.
        name_replacements (Dict[str, str], optional): Dictionary of custom name replacements
            to apply after replacing "_" with ".". Keys are strings to replace, values are
            their replacements. Defaults to None.

    Raises:
        ValueError: If the PyTorch weight is missing or weight type is unexpected.
    """
    keras_layer_path = keras_weight.path
    layer_name = keras_layer_path.split("/")[-2].replace("_", ".")

    if name_replacements:
        for old_name, new_name in name_replacements.items():
            layer_name = layer_name.replace(old_name, new_name)

    if "kernel" in keras_name:
        torch_name = f"{layer_name}.weight"
    elif "bias" in keras_name:
        torch_name = f"{layer_name}.bias"
    elif "gamma" in keras_name:
        torch_name = f"{layer_name}.weight"
    elif "beta" in keras_name:
        torch_name = f"{layer_name}.bias"
    elif "moving_mean" in keras_name:
        torch_name = f"{layer_name}.running_mean"
    elif "moving_variance" in keras_name:
        torch_name = f"{layer_name}.running_var"
    else:
        raise ValueError(f"Unexpected weight type in attention layer: {keras_name}")

    try:
        torch_weights = torch_weights_dict[torch_name]
        transfer_weights(torch_name, keras_weight, torch_weights)
    except KeyError:
        raise ValueError(
            f"Missing PyTorch weight '{torch_name}' for Keras weight '{keras_name}'"
        )


def transfer_nested_layer_weights(
    keras_layer: keras.Layer,
    torch_weights_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    torch_prefix: str,
    name_mapping: Optional[Dict[str, str]] = None,
    skip_paths: Optional[list] = None,
) -> list:
    """
    Transfer weights for a nested Keras layer by mapping weight paths to
    PyTorch state dict keys.

    Converts Keras ``weight.path`` (e.g.
    ``decoder_layer_0/self_attn_q_proj/kernel``) into the corresponding
    PyTorch key (e.g.
    ``transformer.decoder.layers.0.self_attn.q_proj.weight``) using two
    steps:

    1. Strip the top-level layer name, replace ``/`` with ``.``.
    2. Apply *name_mapping* replacements sequentially (last segment maps
       like ``kernel`` → ``weight``).

    The resolved key is looked up in *torch_weights_dict* and handed to
    :func:`transfer_weights`, which takes care of shape transforms
    (conv transpose, dense transpose, etc.).

    Args:
        keras_layer: A Keras layer whose ``.weights`` will be iterated.
        torch_weights_dict: PyTorch state dict (values may be tensors or
            numpy arrays).
        torch_prefix: Prefix prepended to every resolved torch key
            (e.g. ``"transformer.decoder.layers.0"``).
        name_mapping: Optional ordered dict of ``{old: new}`` string
            replacements applied after the ``/`` → ``.`` conversion.
            Applied sequentially, so order matters.  Common entries::

                {"kernel": "weight", "gamma": "weight",
                 "beta": "bias", "moving_mean": "running_mean",
                 "moving_variance": "running_var"}

        skip_paths: Optional list of substrings; any weight whose
            ``path`` contains one of these strings is skipped and
            returned for manual handling.

    Returns:
        A list of ``(keras_weight, weight_path)`` tuples that were
        skipped (matched *skip_paths* or not found in
        *torch_weights_dict*), so the caller can handle them manually.
    """
    if name_mapping is None:
        name_mapping = {
            "kernel": "weight",
            "gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var",
        }

    if skip_paths is None:
        skip_paths = []

    layer_name = keras_layer.name
    skipped: list = []

    for w in keras_layer.weights:
        path = w.path

        if any(s in path for s in skip_paths):
            skipped.append((w, path))
            continue

        suffix = path[len(layer_name) :].lstrip("/")
        torch_suffix = suffix.replace("/", ".")

        for old, new in name_mapping.items():
            torch_suffix = torch_suffix.replace(old, new)
        torch_key = f"{torch_prefix}.{torch_suffix}" if torch_prefix else torch_suffix

        if torch_key not in torch_weights_dict:
            skipped.append((w, path))
            continue

        torch_weight = torch_weights_dict[torch_key]
        transfer_weights(path, w, torch_weight)

    return skipped


def compare_keras_torch_names(
    keras_name: str,
    keras_weights: Union[keras.Variable, np.ndarray],
    torch_name: str,
    torch_weights: Union[torch.Tensor, np.ndarray],
    verbose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    check_values: bool = False,
) -> bool:
    """
    Enhanced comparison of Keras and PyTorch weights with comprehensive error reporting.

    Args:
        keras_name: Name of the Keras weights
        keras_weights: Keras weights as Variable or numpy array
        torch_name: Name of the PyTorch weights
        torch_weights: PyTorch weights as Tensor or numpy array
        verbose: Whether to print mismatch details (default: True)
        rtol: Relative tolerance for value comparison
        atol: Absolute tolerance for value comparison
        check_values: Whether to check actual weight values (default: False)

    Returns:
        Boolean indicating if weights match

    Raises:
        WeightMismatchError: When weights don't match and detailed error information
    """
    import torch

    def _format_mismatch(error_type: str, details: str) -> str:
        return (
            f"Weight Mismatch Detected:\n"
            f"  Keras name: {keras_name}\n"
            f"  Torch name: {torch_name}\n"
            f"  Keras shape: {keras_weights_np.shape}\n"
            f"  Torch shape: {torch_weights_np.shape}\n"
            f"  Type: {error_type}\n"
            f"  Details: {details}\n"
            f"{'-' * 50}"
        )

    def _handle_mismatch(error_type: str, details: str) -> bool:
        message = _format_mismatch(error_type, details)
        if verbose:
            print(message)
        return False

    keras_weights_np = (
        keras_weights.numpy() if hasattr(keras_weights, "numpy") else keras_weights
    )
    torch_weights_np = (
        torch_weights.detach().cpu().numpy()
        if isinstance(torch_weights, torch.Tensor)
        else torch_weights
    )

    keras_size = np.prod(keras_weights_np.shape)
    torch_size = np.prod(torch_weights_np.shape)

    if keras_size != torch_size:
        if (
            (keras_size == 0 and torch_size == 1)
            or (keras_size == 1 and torch_size > 1)
            or (torch_size == 1 and keras_size > 1)
        ):
            return True

        if _skip_state["active"]:
            return True

        return _handle_mismatch(
            "shape",
            f"Element count mismatch: Keras={keras_size} ({keras_weights_np.shape}), "
            f"Torch={torch_size} ({torch_weights_np.shape})",
        )

    weight_type = WeightType.find_weight_type(keras_name)
    if weight_type:
        keras_suffix, torch_suffix = weight_type.value
        if not torch_name.endswith(torch_suffix):
            return _handle_mismatch(
                "type",
                f"Expected Torch suffix '{torch_suffix}' for Keras '{keras_suffix}'",
            )

    if check_values:
        try:
            if not np.allclose(
                keras_weights_np, torch_weights_np, rtol=rtol, atol=atol
            ):
                max_diff = np.max(np.abs(keras_weights_np - torch_weights_np))
                return _handle_mismatch(
                    "values", f"Weight values differ (max diff: {max_diff:.6f})"
                )
        except Exception as e:
            return _handle_mismatch(
                "comparison", f"Error during value comparison: {str(e)}"
            )

    return True


def copy_weights_by_path_suffix(src, dst):
    """Copy matching weights from ``src`` into ``dst`` by stable path suffix.

    Drops auto-counter wrappers like ``clip_attention_X/`` so weights can be
    shared across classes built with different layer-instantiation orders
    (where Keras' auto-counter would otherwise misalign). Only weights with
    matching last-two path segments *and* identical shape are assigned;
    everything else is left untouched.

    Used to warm-start a task-specific subclass (e.g.
    ``CLIPImageClassify``) from a base model checkpoint
    (``CLIPModel.from_weights(variant)``).

    Returns:
        list[str]: paths of ``dst`` weights left untouched (no suffix match or
        shape mismatch), e.g. a task head absent from the source checkpoint.
    """

    def suffix(w):
        parts = w.path.split("/")[-2:]
        parts[-1] = re.sub(r"_\d+$", "", parts[-1])
        return "/".join(parts)

    src_counts = Counter(suffix(w) for w in src.weights)

    def key(w):
        return w.path if src_counts[suffix(w)] > 1 else suffix(w)

    src_map = {key(w): w for w in src.weights}
    skipped = []
    for dst_w in dst.weights:
        src_w = src_map.get(key(dst_w))
        if src_w is not None and tuple(src_w.shape) == tuple(dst_w.shape):
            dst_w.assign(src_w)
        else:
            skipped.append(dst_w.path)
    return skipped
