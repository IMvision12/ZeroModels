from zeromodels.base import BaseConfig


class MaskFormerConfig(BaseConfig):
    r"""Configuration for [`MaskFormerUniversalSegment`], the MaskFormer segmenter.

    Instantiating it with the defaults yields a configuration close to the
    maskformer-swin-tiny-ade style. Fields mirror the model constructor and
    serialize flat to a repo's `kf_config.json`.

    Args:
        backbone_embed_dim (`int`, *optional*, defaults to 96):
            Stage-0 Swin embedding dimension.
        backbone_depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Swin blocks per stage (length-4).
        backbone_num_heads (`tuple`, *optional*, defaults to `(3, 6, 12, 24)`):
            Swin attention heads per stage (length-4).
        backbone_window_size (`int`, *optional*, defaults to 7):
            Swin window edge length.
        fpn_feature_size (`int`, *optional*, defaults to 256):
            FPN channel count in the pixel decoder.
        mask_feature_size (`int`, *optional*, defaults to 256):
            Mask-feature dimension produced by the pixel decoder.
        decoder_d_model (`int`, *optional*, defaults to 256):
            Transformer-decoder model dimension.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of transformer-decoder layers.
        decoder_heads (`int`, *optional*, defaults to 8):
            Number of decoder attention heads.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Decoder feed-forward dimension.
        num_queries (`int`, *optional*, defaults to 100):
            Number of object queries.
        num_classes (`int`, *optional*, defaults to 150):
            Number of semantic classes (excluding the no-object class).
        image_size (`int`, *optional*, defaults to 512):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.maskformer import (
    ...     MaskFormerConfig,
    ...     MaskFormerUniversalSegment,
    ... )

    >>> # Initializing a zeromodels/maskformer-swin-tiny-ade style configuration
    >>> configuration = MaskFormerConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = MaskFormerUniversalSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "maskformer"

    backbone_embed_dim: int = 96
    backbone_depths: tuple = (2, 2, 6, 2)
    backbone_num_heads: tuple = (3, 6, 12, 24)
    backbone_window_size: int = 7
    fpn_feature_size: int = 256
    mask_feature_size: int = 256
    decoder_d_model: int = 256
    decoder_num_layers: int = 6
    decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    num_queries: int = 100
    num_classes: int = 150
    image_size: int = 512
