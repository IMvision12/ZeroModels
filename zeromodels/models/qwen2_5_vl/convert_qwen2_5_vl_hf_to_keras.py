from zeromodels.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    transfer_qwen2_vl_weights,
)


def transfer_qwen2_5_vl_weights(keras_model, hf_state_dict):
    transfer_qwen2_vl_weights(keras_model, hf_state_dict)
