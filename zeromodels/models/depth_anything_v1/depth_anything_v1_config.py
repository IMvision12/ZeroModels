from zeromodels.base import BaseConfig


class DepthAnythingV1Config(BaseConfig):
    r"""Configuration for [`DepthAnythingV1DepthEstimation`], the Depth Anything V1
    monocular depth estimator (DINOv2 ViT backbone + DPT neck + depth head).

    The defaults describe the Depth Anything V1 Small variant; other variants
    override the backbone dimensions and neck widths. Fields mirror the model
    constructor and serialize flat to a repo's `zm_config.json`.

    Args:
        backbone_dim (`int`, *optional*, defaults to 384):
            Hidden dimension of the DINOv2 ViT backbone.
        backbone_depth (`int`, *optional*, defaults to 12):
            Number of transformer layers in the backbone.
        backbone_num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads in the backbone.
        out_indices (`tuple`, *optional*, defaults to `(9, 10, 11, 12)`):
            Backbone layer indices whose hidden states feed the DPT neck.
        neck_hidden_sizes (`tuple`, *optional*, defaults to `(48, 96, 192, 384)`):
            Channel widths of the four DPT reassemble stages.
        fusion_hidden_size (`int`, *optional*, defaults to 64):
            Channel width of the DPT fusion blocks and depth head.
        reassemble_factors (`tuple`, *optional*, defaults to `(4, 2, 1, 0.5)`):
            Spatial resampling factors of the four DPT reassemble stages.
        depth_estimation_type (`str`, *optional*, defaults to `"relative"`):
            Head mode: `"relative"` (final ReLU, disparity-style) for the base
            release, or `"metric"` (final `sigmoid * max_depth`) for metric
            fine-tunes.
        max_depth (`float`, *optional*, defaults to 1.0):
            Maximum depth in meters, used only when `depth_estimation_type` is
            `"metric"`.
        image_size (`int`, *optional*, defaults to 518):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.depth_anything_v1 import (
    ...     DepthAnythingV1Config, DepthAnythingV1DepthEstimation)

    >>> configuration = DepthAnythingV1Config()
    >>> model = DepthAnythingV1DepthEstimation(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "depth_anything"

    backbone_dim: int = 384
    backbone_depth: int = 12
    backbone_num_heads: int = 6
    out_indices: tuple = (9, 10, 11, 12)
    neck_hidden_sizes: tuple = (48, 96, 192, 384)
    fusion_hidden_size: int = 64
    reassemble_factors: tuple = (4, 2, 1, 0.5)
    depth_estimation_type: str = "relative"
    max_depth: float = 1.0
    image_size: int = 518
