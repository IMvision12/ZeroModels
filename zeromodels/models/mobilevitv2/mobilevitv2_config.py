from zeromodels.base import BaseConfig


class MobileViTV2Config(BaseConfig):
    r"""Configuration for MobileViTV2: [`MobileViTV2Model`],
    [`MobileViTV2ImageClassify`] and [`MobileViTV2SemanticSegment`].

    MobileViTV2 replaces MobileViT's multi-head attention with separable self-attention
    (linear in token count) and scales the whole network by a single width multiplier.
    A single config serves the whole family: the backbone/classifier read `multiplier`
    / `image_size` / `num_classes`, and the DeepLabV3 segmentation head additionally
    reads the `output_stride` / `atrous_rates` / `aspp_*` fields (the classifier ignores
    them). One `kf_config.json` sits on each variant's repo and fields serialize flat.

    Args:
        multiplier (`float`, *optional*, defaults to 1.0):
            Width multiplier scaling all channel counts.
        image_size (`int`, *optional*, defaults to 256):
            Square input resolution the weights were trained at (512 for the
            segmentation checkpoints).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of output classes (backbone ignores it; 21 for the PASCAL VOC
            segmentation checkpoints).
        output_stride (`int`, *optional*, defaults to 32):
            Segmentation only. Ratio of input to backbone-output spatial resolution;
            the last stage uses atrous convolutions to hold this stride (16 for the
            DeepLabV3 heads). Ignored by the classifier, which always uses 32.
        atrous_rates (`tuple`, *optional*, defaults to `(6, 12, 18)`):
            Segmentation only. Dilation rates of the ASPP parallel branches.
        aspp_out_channels (`int`, *optional*, defaults to 512):
            Segmentation only. Channel count of each ASPP branch and the fused
            projection.
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            Segmentation only. Dropout probability before the final classifier conv.

    Examples:

    ```python
    >>> from zeromodels.models.mobilevitv2 import (
    ...     MobileViTV2Config,
    ...     MobileViTV2ImageClassify,
    ... )

    >>> configuration = MobileViTV2Config()
    >>> model = MobileViTV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilevitv2"

    multiplier: float = 1.0
    image_size: int = 256
    num_classes: int = 1000
    # DeepLabV3 segmentation head (ignored by the backbone / classifier).
    output_stride: int = 32
    atrous_rates: tuple = (6, 12, 18)
    aspp_out_channels: int = 512
    aspp_dropout_prob: float = 0.1
