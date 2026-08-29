from zeromodels.base import BaseConfig


class BeitConfig(BaseConfig):
    r"""Configuration for the BEiT models ([`BeitModel`], [`BeitImageClassify`],
    [`BeitSemanticSegment`]).

    BEiT is a ViT-family vision transformer: a convolutional patch stem, a learned
    CLS token, and a stack of pre-norm transformer blocks. It differs from vanilla
    ViT in three ways that all hosted checkpoints use: a per-layer 2D relative
    position bias in attention (no absolute position embeddings), a learnable layer
    scale (``lambda_1`` / ``lambda_2``) on each residual branch, and mean pooling of
    the patch tokens (followed by a LayerNorm) for classification. The
    ``BeitSemanticSegment`` head adds an FPN neck plus a UPerNet decode head on four
    intermediate feature maps. Defaults describe ``microsoft/beit-base-patch16-224``.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads per block.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the MLP ("intermediate") layer.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.
        patch_size (`int`, *optional*, defaults to 16):
            Patch (conv-stem) size.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Initial value of the per-channel layer-scale weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Epsilon of every LayerNorm.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of head outputs (classification classes, or segmentation labels).
        out_indices (`tuple`, *optional*, defaults to `(3, 5, 7, 11)`):
            1-based encoder layers whose outputs feed the segmentation FPN neck.
            Used only by [`BeitSemanticSegment`].
        pool_scales (`tuple`, *optional*, defaults to `(1, 2, 3, 6)`):
            Pyramid-pooling scales of the UPerNet decode head. Used only by
            [`BeitSemanticSegment`].

    Examples:

    ```python
    >>> from zeromodels.models.beit import BeitConfig, BeitImageClassify

    >>> configuration = BeitConfig()
    >>> model = BeitImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "beit"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    layer_scale_init_value: float = 0.1
    layer_norm_eps: float = 1e-12
    num_classes: int = 1000
    out_indices: tuple = (3, 5, 7, 11)
    pool_scales: tuple = (1, 2, 3, 6)
