from zeromodels.base import BaseConfig


class ResMLPConfig(BaseConfig):
    r"""Configuration for [`ResMLPModel`] / [`ResMLPImageClassify`].

    ResMLP is an all-MLP image classifier that swaps LayerNorm for a learned affine
    (Aff) transform and mixes patches with a single linear layer plus residual
    connections. One `kf_config.json` (declaring the canonical [`ResMLPImageClassify`])
    sits on each variant's repo, and both the backbone and classifier load from it.
    Fields mirror the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        embed_dim (`int`, *optional*, defaults to 384):
            Patch embedding (hidden) size.
        depth (`int`, *optional*, defaults to 12):
            Number of ResMLP blocks.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Channel-MLP expansion ratio.
        layer_scale_init (`float`, *optional*, defaults to 1e-4):
            Initial value for the per-channel LayerScale.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.resmlp import ResMLPConfig, ResMLPImageClassify

    >>> configuration = ResMLPConfig()
    >>> model = ResMLPImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "resmlp"

    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    mlp_ratio: int = 4
    layer_scale_init: float = 1e-4
    image_size: int = 224
    num_classes: int = 1000
