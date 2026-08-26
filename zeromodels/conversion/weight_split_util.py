from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import keras

if TYPE_CHECKING:
    import torch

# Constants for weight classification
IRRELEVANT_KEYWORDS = {
    "num_batches_tracked",
    "weight_u",
    "weight_v",
    "bias_u",
    "bias_v",
    "auxiliary",
    "temp",
}

NON_TRAINABLE_KEYWORDS = {
    "running_mean",
    "running_var",
    "bias",
    "bn",
    "norm",
    "moving_mean",
    "moving_variance",
}


def calculate_weight_stats(state_dict: OrderedDict) -> Dict[str, Any]:
    """
    Calculate statistics about model weights.

    Args:
        state_dict: The model's state dictionary

    Returns:
        Dictionary containing parameter statistics
    """
    stats = {
        "total_params": 0,
        "trainable_params": 0,
        "non_trainable_params": 0,
        "irrelevant_params": 0,
        "memory_mb": 0.0,
    }

    for param_name, param_value in state_dict.items():
        num_params = param_value.nelement()
        stats["total_params"] += num_params

        if any(keyword in param_name for keyword in IRRELEVANT_KEYWORDS):
            stats["irrelevant_params"] += num_params
        elif any(keyword in param_name for keyword in NON_TRAINABLE_KEYWORDS):
            stats["non_trainable_params"] += num_params
        else:
            stats["trainable_params"] += num_params

        stats["memory_mb"] += (
            param_value.nelement() * param_value.element_size() / (1024 * 1024)
        )

    return stats


def split_state_dict(
    state_dict: OrderedDict,
    return_stats: bool = False,
    custom_irrelevant: Optional[Set[str]] = None,
    custom_non_trainable: Optional[Set[str]] = None,
) -> Union[
    Tuple[OrderedDict, OrderedDict, OrderedDict],
    Tuple[OrderedDict, OrderedDict, OrderedDict, Dict[str, Any]],
]:
    """
    Split PyTorch state dictionary into different parameter types.

    Args:
        state_dict: The model's state dictionary
        return_stats: Whether to return weight statistics
        custom_irrelevant: Additional keywords for irrelevant parameters
        custom_non_trainable: Additional keywords for non-trainable parameters

    Returns:
        Tuple of (trainable_params, non_trainable_params, irrelevant_params)
        and optionally weight statistics
    """
    if not state_dict:
        raise ValueError("Empty state dictionary provided")

    irrelevant_keys = IRRELEVANT_KEYWORDS.union(custom_irrelevant or set())
    non_trainable_keys = NON_TRAINABLE_KEYWORDS.union(custom_non_trainable or set())

    trainable_params = OrderedDict()
    non_trainable_params = OrderedDict()
    irrelevant_params = OrderedDict()

    for param_name, param_value in state_dict.items():
        if any(keyword in param_name for keyword in irrelevant_keys):
            irrelevant_params[param_name] = param_value
        elif any(keyword in param_name for keyword in non_trainable_keys):
            non_trainable_params[param_name] = param_value
        else:
            trainable_params[param_name] = param_value

    if return_stats:
        stats = calculate_weight_stats(state_dict)
        return trainable_params, non_trainable_params, irrelevant_params, stats

    return trainable_params, non_trainable_params, irrelevant_params


def separate_keras_weights(
    keras_model: keras.Model, return_metadata: bool = False
) -> Union[
    Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]]],
    Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]], Dict[str, Any]],
]:
    """
    Separate Keras model weights into trainable and non-trainable parts.

    Args:
        keras_model: The Keras model to process
        return_metadata: Whether to return model metadata

    Returns:
        Tuple of (trainable_weights, non_trainable_weights) and
        optionally metadata dictionary
    """
    if not keras_model.layers:
        raise ValueError("Model has no layers")

    trainable_weights = []
    non_trainable_weights = []

    for layer in keras_model.layers:
        for weight in layer.trainable_weights:
            trainable_weights.append((weight, f"{layer.name}_{weight.name}"))

        for weight in layer.non_trainable_weights:
            non_trainable_weights.append((weight, f"{layer.name}_{weight.name}"))

    if return_metadata:
        metadata = {
            "model_name": keras_model.name,
            "total_params": keras_model.count_params(),
            "trainable_params": sum(w.numpy().size for w, _ in trainable_weights),
            "non_trainable_params": sum(
                w.numpy().size for w, _ in non_trainable_weights
            ),
            "layer_count": len(keras_model.layers),
        }
        return trainable_weights, non_trainable_weights, metadata

    return trainable_weights, non_trainable_weights


def split_model_weights(
    model: Union[torch.nn.Module, keras.Model],
    return_stats: bool = False,
    custom_keywords: Optional[Dict[str, Set[str]]] = None,
) -> Union[
    Tuple[OrderedDict, OrderedDict, OrderedDict],
    Tuple[OrderedDict, OrderedDict, OrderedDict, Dict[str, Any]],
    Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]]],
    Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]], Dict[str, Any]],
]:
    """
    Universal model weight splitter for both PyTorch and Keras models.

    Args:
        model: PyTorch or Keras model
        return_stats: Whether to return statistics/metadata
        custom_keywords: Optional dictionary with custom keywords for classification

    Returns:
        Various tuple combinations based on model type and return_stats

    Raises:
        ValueError: If model type is unsupported or model is invalid
        RuntimeError: If weight extraction fails
    """
    from torch import nn

    try:
        # Check Keras first: on torch backend, Keras models inherit nn.Module
        if isinstance(model, keras.Model):
            return separate_keras_weights(model, return_stats)

        elif isinstance(model, nn.Module):
            if not hasattr(model, "state_dict"):
                raise ValueError("Invalid PyTorch model: missing state_dict")

            custom_irrelevant = (
                custom_keywords.get("irrelevant") if custom_keywords else None
            )
            custom_non_trainable = (
                custom_keywords.get("non_trainable") if custom_keywords else None
            )

            return split_state_dict(
                model.state_dict(),
                return_stats,
                custom_irrelevant,
                custom_non_trainable,
            )

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    except Exception as e:
        raise RuntimeError(f"Failed to split model weights: {str(e)}") from e
