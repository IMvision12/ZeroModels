from zeromodels.base import BaseConfig


class MiTConfig(BaseConfig):
    r"""Configuration for [`MiTModel`] / [`MiTImageClassify`].

    MiT (Mix Vision Transformer) is the hierarchical, convolution-augmented transformer
    encoder introduced with SegFormer, producing multi-scale features via overlapping
    patch merging and efficient (sequence-reduced) attention. One `kf_config.json`
    (declaring the canonical [`MiTImageClassify`]) sits on each variant's repo, and both
    the backbone and classifier load from it. Fields mirror the model constructor and
    serialize flat.

    Args:
        embed_dim (`tuple`, *optional*, defaults to `(32, 64, 160, 256)`):
            Channel width per stage.
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Number of transformer blocks per stage.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.mit import MiTConfig, MiTImageClassify

    >>> configuration = MiTConfig()
    >>> model = MiTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mit"

    embed_dim: tuple = (32, 64, 160, 256)
    depths: tuple = (2, 2, 2, 2)
    image_size: int = 224
    num_classes: int = 1000


# Hosted variants -> (model arch key). Weights load by Hub repo id (kf_config.json);
# the github release urls have been removed. MiT ships no timm ids (SegFormer encoder).
MIT_VARIANTS = {
    "mit_b0_in1k": {
        "model": "mit_b0",
    },
    "mit_b1_in1k": {
        "model": "mit_b1",
    },
    "mit_b2_in1k": {
        "model": "mit_b2",
    },
    "mit_b3_in1k": {
        "model": "mit_b3",
    },
    "mit_b4_in1k": {
        "model": "mit_b4",
    },
    "mit_b5_in1k": {
        "model": "mit_b5",
    },
}
