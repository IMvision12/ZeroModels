from zeromodels.base import BaseConfig


class OneFormerConfig(BaseConfig):
    r"""Configuration for [`OneFormerUniversalSegment`], the OneFormer segmenter.

    Instantiating it with the defaults yields a configuration close to the
    oneformer_ade20k_swin_tiny style. Fields mirror the model constructor and
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
        query_dec_layers (`int`, *optional*, defaults to 2):
            Number of query-transformer layers that initialize the queries.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_queries (`int`, *optional*, defaults to 150):
            Number of object queries.
        num_classes (`int`, *optional*, defaults to 150):
            Number of semantic classes (excluding the no-object class).
        task_seq_len (`int`, *optional*, defaults to 77):
            Padded task-prompt length in tokens consumed by the task MLP.
        image_size (`int`, *optional*, defaults to 512):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.oneformer import (
    ...     OneFormerConfig,
    ...     OneFormerUniversalSegment,
    ... )

    >>> # Initializing a zeromodels/oneformer_ade20k_swin_tiny style configuration
    >>> configuration = OneFormerConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = OneFormerUniversalSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "oneformer"

    backbone_embed_dim: int = 96
    backbone_depths: tuple = (2, 2, 6, 2)
    backbone_num_heads: tuple = (3, 6, 12, 24)
    backbone_window_size: int = 7
    hidden_dim: int = 256
    mask_feature_size: int = 256
    encoder_num_layers: int = 6
    encoder_ffn_dim: int = 1024
    decoder_num_layers: int = 9
    decoder_ffn_dim: int = 2048
    query_dec_layers: int = 2
    num_heads: int = 8
    num_queries: int = 150
    num_classes: int = 150
    task_seq_len: int = 77
    image_size: int = 512
