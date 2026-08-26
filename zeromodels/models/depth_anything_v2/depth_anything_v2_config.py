from zeromodels.models.depth_anything_v1.depth_anything_v1_config import (
    DepthAnythingV1Config,
)


class DepthAnythingV2Config(DepthAnythingV1Config):
    r"""Configuration for [`DepthAnythingV2DepthEstimation`], the Depth Anything V2
    monocular depth estimator.

    V2 reuses the V1 architecture end-to-end (only the training data and weights
    differ), so this config inherits every field from [`DepthAnythingV1Config`].
    The defaults describe the V2 Small relative-depth variant; the metric
    fine-tunes override `depth_estimation_type="metric"` and `max_depth` (`20.0`
    indoor, `80.0` outdoor). Fields serialize flat to a repo's `zm_config.json`.

    Examples:

    ```python
    >>> from zeromodels.models.depth_anything_v2 import (
    ...     DepthAnythingV2Config, DepthAnythingV2DepthEstimation)

    >>> configuration = DepthAnythingV2Config()
    >>> model = DepthAnythingV2DepthEstimation(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "depth_anything"
