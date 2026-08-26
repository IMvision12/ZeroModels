from zeromodels.base import BaseConfig


class CaiTConfig(BaseConfig):
    r"""Configuration for [`CaiTModel`] / [`CaiTImageClassify`].

    CaiT (Class-Attention in Image Transformers) deepens ViT with LayerScale and
    separates patch self-attention from a small stack of class-attention layers that
    update only the class token. One `zm_config.json` (declaring the canonical
    [`CaiTImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        embed_dim (`int`, *optional*, defaults to 192):
            Transformer hidden size.
        depth (`int`, *optional*, defaults to 24):
            Number of patch self-attention blocks.
        num_heads (`int`, *optional*, defaults to 4):
            Number of attention heads.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.cait import CaiTConfig, CaiTImageClassify

    >>> configuration = CaiTConfig()
    >>> model = CaiTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "cait"

    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 24
    num_heads: int = 4
    image_size: int = 224
    num_classes: int = 1000
