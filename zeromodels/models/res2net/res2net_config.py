from zeromodels.base import BaseConfig


class Res2NetConfig(BaseConfig):
    r"""Configuration for [`Res2NetModel`] / [`Res2NetImageClassify`].

    Res2Net adds multi-scale residual connections inside the bottleneck via a
    `scale` split. The hosted variants override `depth` / `base_width` / `scale` /
    `cardinality`. One `kf_config.json` (declaring the canonical
    [`Res2NetImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        depth (`tuple`, *optional*, defaults to `(3, 4, 6, 3)`):
            Number of Res2Net blocks per stage.
        base_width (`int`, *optional*, defaults to 26):
            Base channel width of each scale split.
        scale (`int`, *optional*, defaults to 4):
            Number of feature-map scales inside the bottleneck.
        cardinality (`int`, *optional*, defaults to 1):
            Number of grouped-conv groups (>1 for the res2next variant).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.res2net import Res2NetConfig, Res2NetImageClassify

    >>> configuration = Res2NetConfig()
    >>> model = Res2NetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "res2net"

    depth: tuple = (3, 4, 6, 3)
    base_width: int = 26
    scale: int = 4
    cardinality: int = 1
    num_classes: int = 1000
