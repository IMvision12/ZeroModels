from zeromodels.base import BaseConfig


class ConditionalDetrConfig(BaseConfig):
    r"""Configuration for [`ConditionalDetrModel`] / [`ConditionalDETRDetect`].

    Instantiating it with the defaults yields a configuration close to the
    microsoft/conditional-detr-resnet-50 checkpoint. Fields mirror the model
    constructor and serialize to a repo's `zm_config.json`.

    Args:
        backbone_variant (`str`, *optional*, defaults to `"ResNet50"`):
            ResNet backbone, one of `"ResNet50"` or `"ResNet101"`.
        hidden_dim (`int`, *optional*, defaults to 256):
            Width of the transformer encoder and decoder.
        num_heads (`int`, *optional*, defaults to 8):
            Attention heads in every encoder / decoder attention layer.
        num_encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feed-forward width inside each encoder / decoder layer.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability in the transformer layers.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. the maximum detections per image.
        num_classes (`int`, *optional*, defaults to 91):
            Number of classes. Conditional DETR is trained with a sigmoid focal
            loss, so there is no extra "no-object" class (COCO: 91).
        image_size (`int`, *optional*, defaults to `None`):
            Fixed square input resolution, or `None` for a dynamic input that
            accepts the aspect-preserving resize of the image processor.

    Examples:

    ```python
    >>> from zeromodels.models.conditional_detr import (
    ...     ConditionalDetrConfig,
    ...     ConditionalDETRDetect,
    ... )

    >>> configuration = ConditionalDetrConfig()
    >>> model = ConditionalDETRDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "conditional_detr"

    backbone_variant: str = "ResNet50"
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout_rate: float = 0.1
    num_queries: int = 300
    num_classes: int = 91
    image_size: int = None
