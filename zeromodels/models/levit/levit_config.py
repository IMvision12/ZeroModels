from zeromodels.base import BaseConfig


class LevitConfig(BaseConfig):
    r"""Configuration for the LeViT models ([`LevitModel`], [`LevitImageClassify`]).

    LeViT is a hybrid conv/transformer image classifier: a four-layer convolutional
    patch stem (each ``Conv2d`` stride 2, so the input is downsampled 16x) feeds a
    stack of attention stages. Each attention adds a learnable 2D relative-position
    bias to its scores, every linear layer is fused with a BatchNorm
    (``MLPLayerWithBN``), and an ``AttentionSubsample`` halves the resolution between
    stages. The classifier mean-pools the final tokens and applies a BatchNorm +
    Dense head; the distilled variants add a second head whose logits are averaged
    with the first. Defaults describe ``facebook/levit-128S``.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the patch-stem convolutions.
        stride (`int`, *optional*, defaults to 2):
            Stride of the patch-stem convolutions.
        padding (`int`, *optional*, defaults to 1):
            Zero-padding of the patch-stem convolutions.
        patch_size (`int`, *optional*, defaults to 16):
            Total downsampling of the patch stem (``image_size // patch_size`` is the
            stage-0 token grid).
        hidden_sizes (`tuple`, *optional*, defaults to `(128, 256, 384)`):
            Token dimension of each of the three stages.
        num_attention_heads (`tuple`, *optional*, defaults to `(4, 8, 12)`):
            Attention heads per stage.
        depths (`tuple`, *optional*, defaults to `(4, 4, 4)`):
            Number of attention blocks per stage.
        key_dim (`tuple`, *optional*, defaults to `(16, 16, 16)`):
            Per-head query/key dimension in each stage.
        mlp_ratio (`tuple`, *optional*, defaults to `(2, 2, 2)`):
            MLP expansion ratio per stage.
        attention_ratio (`tuple`, *optional*, defaults to `(2, 2, 2)`):
            Ratio of the attention value dimension to the key dimension.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier outputs.
        use_distillation (`bool`, *optional*, defaults to True):
            Add a second (distillation) head and average the two heads' logits (the
            released ``facebook/levit-*`` recipe). Ignored by [`LevitModel`].

    Examples:

    ```python
    >>> from zeromodels.models.levit import LevitConfig, LevitImageClassify

    >>> configuration = LevitConfig()
    >>> model = LevitImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "levit"

    image_size: int = 224
    num_channels: int = 3
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    patch_size: int = 16
    hidden_sizes: tuple = (128, 256, 384)
    num_attention_heads: tuple = (4, 8, 12)
    depths: tuple = (4, 4, 4)
    key_dim: tuple = (16, 16, 16)
    mlp_ratio: tuple = (2, 2, 2)
    attention_ratio: tuple = (2, 2, 2)
    num_classes: int = 1000
    use_distillation: bool = True
