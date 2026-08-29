from zeromodels.base import BaseConfig


class RegNetConfig(BaseConfig):
    r"""Configuration for [`RegNetModel`] / [`RegNetImageClassify`].

    RegNet (Designing Network Design Spaces) is a quantized-linear ConvNet: a
    3x3 stride-2 stem followed by four stages of residual blocks whose width and
    depth follow a simple parametric rule. Each block is a 1x1 -> 3x3 grouped ->
    1x1 bottleneck (the ``"y"`` variant adds a Squeeze-and-Excitation module),
    with the 3x3 convolution split into ``out_channels // groups_width`` groups.
    The defaults describe ``regnet-y-040``; the hosted variants override
    ``hidden_sizes`` / ``depths`` / ``groups_width``. One ``zm_config.json``
    (declaring the canonical [`RegNetImageClassify`]) sits on each variant's repo,
    and both the backbone and the classifier load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        embedding_size (`int`, *optional*, defaults to 32):
            Output width of the 3x3 stride-2 stem.
        hidden_sizes (`tuple`, *optional*, defaults to `(128, 192, 512, 1088)`):
            Output width per stage.
        depths (`tuple`, *optional*, defaults to `(2, 6, 12, 2)`):
            Number of residual blocks per stage.
        groups_width (`int`, *optional*, defaults to 64):
            Channels per group of the 3x3 grouped convolution (the group count of
            a block is ``out_channels // groups_width``).
        layer_type (`str`, *optional*, defaults to `"y"`):
            Block variant: `"y"` adds a Squeeze-and-Excitation module, `"x"` does
            not.
        downsample_in_first_stage (`bool`, *optional*, defaults to `True`):
            Whether the first stage downsamples (stride 2). RegNet has no
            pooling stem, so this is `True` for the standard checkpoints.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (used by
            [`RegNetImageClassify`]; the backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.regnet import RegNetConfig, RegNetImageClassify

    >>> configuration = RegNetConfig()
    >>> model = RegNetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "regnet"

    embedding_size: int = 32
    hidden_sizes: tuple = (128, 192, 512, 1088)
    depths: tuple = (2, 6, 12, 2)
    groups_width: int = 64
    layer_type: str = "y"
    downsample_in_first_stage: bool = True
    num_classes: int = 1000
