from zeromodels.base import BaseConfig


class Mask2FormerConfig(BaseConfig):
    r"""Configuration for [`Mask2FormerUniversalSegment`], the Mask2Former segmenter.

    Instantiating it with the defaults yields a configuration close to the
    mask2former-swin-tiny-coco-instance style. Fields mirror the model
    constructor and serialize flat to a repo's `zm_config.json`.

    Args:
        backbone_embed_dim (`int`, *optional*, defaults to 96):
            Stage-0 Swin embedding dimension.
        backbone_depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Swin blocks per stage (length-4).
        backbone_num_heads (`tuple`, *optional*, defaults to `(3, 6, 12, 24)`):
            Swin attention heads per stage (length-4).
        backbone_window_size (`int`, *optional*, defaults to 12):
            Swin window edge length.
        hidden_dim (`int`, *optional*, defaults to 256):
            Transformer / pixel-decoder model dimension.
        mask_feature_size (`int`, *optional*, defaults to 256):
            Mask-feature dimension produced by the pixel decoder.
        encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of MSDeformAttn pixel-decoder encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Encoder feed-forward dimension.
        decoder_num_layers (`int`, *optional*, defaults to 9):
            Number of masked-attention transformer-decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Decoder feed-forward dimension.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_queries (`int`, *optional*, defaults to 100):
            Number of object queries.
        num_classes (`int`, *optional*, defaults to 80):
            Number of semantic classes (excluding the no-object class).
        image_size (`int`, *optional*, defaults to 384):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.mask2former import (
    ...     Mask2FormerConfig,
    ...     Mask2FormerUniversalSegment,
    ... )

    >>> # Initializing a zeromodels/mask2former-swin-tiny-coco-instance style config
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = Mask2FormerUniversalSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mask2former"

    backbone_embed_dim: int = 96
    backbone_depths: tuple = (2, 2, 6, 2)
    backbone_num_heads: tuple = (3, 6, 12, 24)
    backbone_window_size: int = 12
    hidden_dim: int = 256
    mask_feature_size: int = 256
    encoder_num_layers: int = 6
    encoder_ffn_dim: int = 1024
    decoder_num_layers: int = 9
    decoder_ffn_dim: int = 2048
    num_heads: int = 8
    num_queries: int = 100
    num_classes: int = 80
    image_size: int = 384
