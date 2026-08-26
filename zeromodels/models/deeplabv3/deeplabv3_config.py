from zeromodels.base import BaseConfig


class DeepLabV3Config(BaseConfig):
    r"""Configuration for [`DeepLabV3SemanticSegment`], the DeepLabV3 segmenter.

    Instantiating it with the defaults yields a configuration close to the
    torchvision deeplabv3_resnet50 (COCO/VOC) style. Fields mirror the model
    constructor and serialize flat to a repo's `zm_config.json`.

    Args:
        backbone_variant (`str`, *optional*, defaults to `"ResNet50"`):
            Dilated ResNet backbone to use, one of `"ResNet50"` or
            `"ResNet101"`.
        num_classes (`int`, *optional*, defaults to 21):
            Number of segmentation classes (Pascal VOC: 20 + 1 background).
        image_size (`int`, *optional*, defaults to 520):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.deeplabv3 import (
    ...     DeepLabV3Config,
    ...     DeepLabV3SemanticSegment,
    ... )

    >>> # Initializing a zeromodels/deeplabv3_resnet50_coco_voc style configuration
    >>> configuration = DeepLabV3Config()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = DeepLabV3SemanticSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deeplabv3"

    backbone_variant: str = "ResNet50"
    num_classes: int = 21
    image_size: int = 520
