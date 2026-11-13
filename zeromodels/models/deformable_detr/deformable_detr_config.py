from zeromodels.base import BaseConfig


class DeformableDetrConfig(BaseConfig):
    r"""Configuration for [`DeformableDetrModel`] / [`DeformableDETRDetect`].

    Instantiating it with the defaults yields a configuration close to the
    SenseTime/deformable-detr checkpoint. Fields mirror the model constructor and
    serialize to a repo's `zm_config.json`.

    Args:
        backbone_variant (`str`, *optional*, defaults to `"ResNet50"`):
            ResNet backbone, one of `"ResNet50"` or `"ResNet101"`.
        d_model (`int`, *optional*, defaults to 256):
            Width of the transformer (and of every projected feature level).
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of deformable encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward width in the encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward width in the decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Attention heads in the encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Attention heads in the decoder.
        num_feature_levels (`int`, *optional*, defaults to 4):
            Number of feature levels: the ResNet stages at strides 8 / 16 / 32
            plus extra stride-2 convolutions (4), or only the stride-32 stage (1).
        encoder_n_points (`int`, *optional*, defaults to 4):
            Sampling points per head per level in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            Sampling points per head per level in the decoder.
        num_queries (`int`, *optional*, defaults to 300):
            Object queries (the region proposals kept when `two_stage`).
        with_box_refine (`bool`, *optional*, defaults to `False`):
            Iterative bounding-box refinement: each decoder layer refines the
            reference boxes of the previous one with its own box head.
        two_stage (`bool`, *optional*, defaults to `False`):
            Two-stage variant: the encoder output proposes the top `num_queries`
            regions, which initialize the decoder queries. Requires
            `with_box_refine`.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability in the transformer layers.
        num_classes (`int`, *optional*, defaults to 91):
            Number of classes (sigmoid scores, no "no-object" class; COCO: 91).
        image_size (`int`, *optional*, defaults to `None`):
            Fixed square input resolution, or `None` for a dynamic input that
            accepts the aspect-preserving resize of the image processor.

    Examples:

    ```python
    >>> from zeromodels.models.deformable_detr import (
    ...     DeformableDetrConfig,
    ...     DeformableDETRDetect,
    ... )

    >>> configuration = DeformableDetrConfig(with_box_refine=True)
    >>> model = DeformableDETRDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deformable_detr"

    backbone_variant: str = "ResNet50"
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    num_feature_levels: int = 4
    encoder_n_points: int = 4
    decoder_n_points: int = 4
    num_queries: int = 300
    with_box_refine: bool = False
    two_stage: bool = False
    dropout_rate: float = 0.1
    num_classes: int = 91
    image_size: int = None
