from zeromodels.base import BaseConfig


class MobileNetV4Config(BaseConfig):
    r"""Configuration for [`MobileNetV4Model`] / [`MobileNetV4ImageClassify`].

    MobileNetV4 (MNv4) refines the MobileNet line with the Universal Inverted
    Bottleneck (UIB) block, which generalizes the inverted residual with optional
    starting and mid depthwise convolutions (yielding the IB / ConvNeXt / ExtraDW /
    FFN shapes), plus a Mobile Multi-Query Attention (Mobile MQA) block in the hybrid
    variants. One `kf_config.json` (declaring the canonical
    [`MobileNetV4ImageClassify`]) sits on each variant's repo, and both the backbone
    and classifier load from it. Fields mirror the model constructor and serialize
    flat.

    Args:
        config (`str`, *optional*, defaults to `"conv_small"`):
            Variant key selecting the block schedule, stem width, activation, and
            layer-scale. One of `"conv_small"`, `"conv_medium"`, `"conv_large"`,
            `"hybrid_medium"`, `"hybrid_large"`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.mobilenetv4 import (
    ...     MobileNetV4Config,
    ...     MobileNetV4ImageClassify,
    ... )

    >>> configuration = MobileNetV4Config()
    >>> model = MobileNetV4ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilenetv4"

    config: str = "conv_small"
    image_size: int = 224
    num_classes: int = 1000
