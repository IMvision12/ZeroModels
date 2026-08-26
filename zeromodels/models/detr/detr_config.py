from zeromodels.base import BaseConfig


class DetrConfig(BaseConfig):
    r"""Configuration for [`DETRDetect`], the DETR object detector.

    Instantiating it with the defaults yields a configuration close to the DETR
    facebook/detr-resnet-50 style. Fields mirror the model constructor and
    serialize flat to a repo's `zm_config.json`.

    Args:
        backbone_variant (`str`, *optional*, defaults to `"ResNet50"`):
            ResNet backbone to use, one of `"ResNet50"` or `"ResNet101"`.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the transformer encoder and decoder layers.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the transformer.
        num_encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Dimension of the feed-forward ("intermediate") layer in the transformer.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability in the transformer layers.
        num_queries (`int`, *optional*, defaults to 100):
            Number of object queries, i.e. detection slots. The maximal number of
            objects [`DETRDetect`] can detect in a single image. For COCO, 100.
        num_classes (`int`, *optional*, defaults to 92):
            Number of object classes, including the no-object class (COCO: 91 + 1).
        image_size (`int`, *optional*, defaults to 800):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.detr import DetrConfig, DETRDetect

    >>> # Initializing a DETR zeromodels/detr-resnet-50 style configuration
    >>> configuration = DetrConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = DETRDetect(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "detr"

    backbone_variant: str = "ResNet50"
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout_rate: float = 0.1
    num_queries: int = 100
    num_classes: int = 92
    image_size: int = 800


class DetrSegmentConfig(BaseConfig):
    r"""Configuration for [`DETRPanopticSegment`], the DETR panoptic segmenter.

    Same transformer as [`DetrConfig`] with a mask head on top; the defaults
    match the facebook/detr-resnet-50-panoptic style.

    Args:
        backbone_variant (`str`, *optional*, defaults to `"ResNet50"`):
            ResNet backbone to use, one of `"ResNet50"` or `"ResNet101"`.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the transformer encoder and decoder layers.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the transformer.
        num_encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Dimension of the feed-forward ("intermediate") layer in the transformer.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability in the transformer layers.
        num_queries (`int`, *optional*, defaults to 100):
            Number of object queries, i.e. detection slots.
        num_classes (`int`, *optional*, defaults to 250):
            Number of segmentation classes (COCO panoptic uses 250, or 251 with
            the no-object class).
        image_size (`int`, *optional*, defaults to 800):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.detr import DetrSegmentConfig, DETRPanopticSegment

    >>> configuration = DetrSegmentConfig()
    >>> model = DETRPanopticSegment(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "detr"

    backbone_variant: str = "ResNet50"
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout_rate: float = 0.1
    num_queries: int = 100
    num_classes: int = 250
    image_size: int = 800
