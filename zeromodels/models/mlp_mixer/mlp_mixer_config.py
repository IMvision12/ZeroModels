from zeromodels.base import BaseConfig


class MLPMixerConfig(BaseConfig):
    r"""Configuration for [`MLPMixerModel`] / [`MLPMixerImageClassify`].

    MLP-Mixer replaces attention and convolution entirely with two MLPs per block: a
    token-mixing MLP across patches and a channel-mixing MLP across features. One
    `zm_config.json` (declaring the canonical [`MLPMixerImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        depths (`int`, *optional*, defaults to 12):
            Number of mixer blocks.
        embed_dim (`int`, *optional*, defaults to 768):
            Patch embedding (hidden) size.
        mlp_ratio (`tuple`, *optional*, defaults to `(0.5, 4.0)`):
            Token-mixing and channel-mixing MLP expansion ratios.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.mlp_mixer import MLPMixerConfig, MLPMixerImageClassify

    >>> configuration = MLPMixerConfig()
    >>> model = MLPMixerImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mlp_mixer"

    patch_size: int = 16
    depths: int = 12
    embed_dim: int = 768
    mlp_ratio: tuple = (0.5, 4.0)
    image_size: int = 224
    num_classes: int = 1000
