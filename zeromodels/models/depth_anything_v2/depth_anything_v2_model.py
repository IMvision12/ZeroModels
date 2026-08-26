import keras

from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.depth_anything_v1.depth_anything_v1_model import (
    DepthAnythingV1DepthEstimation,
    DepthAnythingV1Model,
)

from .depth_anything_v2_config import DepthAnythingV2Config


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV2Model(DepthAnythingV1Model):
    """Depth Anything V2 backbone + DPT neck (no depth-prediction head).

    V2 reuses V1's architecture end-to-end: only training data and
    weights differ (synthetic data + larger-capacity teacher model).
    This class inherits from :class:`DepthAnythingV1Model` and just
    swaps in the V2 weights config.

    Reference:
        - `Depth Anything V2 <https://arxiv.org/abs/2406.09414>`_
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "depth_anything"

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = DepthAnythingV2DepthEstimation.from_hf(
                hf_id, skip_mismatch=skip_mismatch
            )
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model

    def __init__(self, name="DepthAnythingV2Model", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV2DepthEstimation(DepthAnythingV1DepthEstimation):
    """Depth Anything V2 full monocular depth estimator.

    Includes both the relative-depth checkpoints
    (``depth_anything_v2_{small,base,large}``) and the metric fine-tunes
    (``depth_anything_v2_metric_{indoor,outdoor}_{small,base,large}``). The metric
    variants set ``depth_estimation_type="metric"`` and a non-trivial
    ``max_depth`` (``20.0`` for indoor, ``80.0`` for outdoor).

    Inherits the architecture from :class:`DepthAnythingV1DepthEstimation` and
    just swaps the weights config.

    Reference:
        - `Depth Anything V2 <https://arxiv.org/abs/2406.09414>`_
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DepthAnythingV2Config
    HF_MODEL_TYPE = "depth_anything"

    def __init__(self, name="DepthAnythingV2DepthEstimation", **kwargs):
        super().__init__(name=name, **kwargs)
