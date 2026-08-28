from zeromodels.base import BaseConfig


class PvtConfig(BaseConfig):
    r"""Configuration for [`PvtModel`] / [`PvtImageClassify`].

    PVT (Pyramid Vision Transformer v1) is a hierarchical transformer: four stages, each a
    non-overlapping convolutional patch embedding with a learned position embedding,
    spatial-reduction attention (the key/value tokens are reduced by a strided conv), and a
    standard two-Dense feed-forward network. The last stage prepends a class token, and the
    classifier reads it. Variable input resolution is handled by bilinearly interpolating
    each stage's position embedding at weight-load time. One `zm_config.json` (declaring the
    canonical [`PvtImageClassify`]) sits on each variant's repo; both the backbone and the
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        hidden_sizes (`tuple`, *optional*, defaults to `(64, 128, 320, 512)`):
            Channel width per stage.
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Number of transformer blocks per stage.
        num_attention_heads (`tuple`, *optional*, defaults to `(1, 2, 5, 8)`):
            Attention heads per stage.
        sr_ratios (`tuple`, *optional*, defaults to `(8, 4, 2, 1)`):
            Spatial-reduction ratio of the key/value tokens per stage.
        mlp_ratios (`tuple`, *optional*, defaults to `(8, 8, 4, 4)`):
            Feed-forward hidden expansion per stage.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.pvt import PvtConfig, PvtImageClassify

    >>> configuration = PvtConfig()
    >>> model = PvtImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "pvt"

    hidden_sizes: tuple = (64, 128, 320, 512)
    depths: tuple = (2, 2, 2, 2)
    num_attention_heads: tuple = (1, 2, 5, 8)
    sr_ratios: tuple = (8, 4, 2, 1)
    mlp_ratios: tuple = (8, 8, 4, 4)
    image_size: int = 224
    num_classes: int = 1000


# Hosted variants -> arch preset. Weights load by Hub repo id (zm_config.json).
PVT_VARIANTS = {
    "pvt_tiny": {"model": "pvt_tiny", "hf_id": "Zetatech/pvt-tiny-224"},
    "pvt_small": {"model": "pvt_small", "hf_id": "Zetatech/pvt-small-224"},
    "pvt_medium": {"model": "pvt_medium", "hf_id": "Zetatech/pvt-medium-224"},
    "pvt_large": {"model": "pvt_large", "hf_id": "Zetatech/pvt-large-224"},
}
