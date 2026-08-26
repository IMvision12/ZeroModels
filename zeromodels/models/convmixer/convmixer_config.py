from zeromodels.base import BaseConfig


class ConvMixerConfig(BaseConfig):
    r"""Configuration for [`ConvMixerModel`] / [`ConvMixerImageClassify`].

    ConvMixer patchifies the input, then applies a stack of depthwise + pointwise
    convolution mixer blocks at a single resolution. One `zm_config.json` (declaring
    the canonical [`ConvMixerImageClassify`]) sits on each variant's repo, and both the
    backbone and classifier load from it. Fields mirror the model constructor and
    serialize flat.

    Args:
        embed_dim (`int`, *optional*, defaults to 768):
            Channel width used throughout the network.
        depth (`int`, *optional*, defaults to 32):
            Number of mixer blocks.
        kernel_size (`int`, *optional*, defaults to 7):
            Depthwise convolution kernel size.
        patch_size (`int`, *optional*, defaults to 7):
            Patchify stem stride / kernel size.
        activation (`str`, *optional*, defaults to `"gelu"`):
            Activation used inside each block.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.convmixer import ConvMixerConfig, ConvMixerImageClassify

    >>> configuration = ConvMixerConfig()
    >>> model = ConvMixerImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "convmixer"

    embed_dim: int = 768
    depth: int = 32
    kernel_size: int = 7
    patch_size: int = 7
    activation: str = "gelu"
    image_size: int = 224
    num_classes: int = 1000
