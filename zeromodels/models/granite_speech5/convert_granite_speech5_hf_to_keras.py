import numpy as np

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "/": ".",
    "layers_": "layers.",
    ".kernel": ".weight",
    ".gamma": ".weight",
    ".beta": ".bias",
    ".moving_mean": ".running_mean",
    ".moving_variance": ".running_var",
    ".rel_pos_emb": ".rel_pos_emb.weight",
    ".depthwise_conv": ".depthwise_conv.weight",
}


def transfer_granite_speech5_weights(keras_model, state_dict):
    for keras_weight in keras_model.weights:
        torch_name = keras_weight.path
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)
        torch_name = f"encoder.{torch_name}"

        if torch_name not in state_dict:
            raise WeightMappingError(keras_weight.path, torch_name)
        torch_weight = state_dict[torch_name]

        if keras_weight.path.endswith("depthwise_conv"):
            keras_weight.assign(np.transpose(np.asarray(torch_weight)[:, 0, :], (1, 0)))
        elif keras_weight.path.endswith("rel_pos_emb"):
            keras_weight.assign(torch_weight)
        else:
            transfer_weights(keras_weight.path, keras_weight, torch_weight)
