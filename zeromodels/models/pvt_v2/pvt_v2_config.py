from zeromodels.base import BaseConfig


class PvtV2Config(BaseConfig):
    r"""Configuration for [`PvtV2Model`] / [`PvtV2ImageClassify`].

    PVTv2 (Pyramid Vision Transformer v2) is a hierarchical, convolution-augmented
    transformer: four stages, each with an OVERLAPPING convolutional patch embedding,
    spatial-reduction attention (optionally the linear variant with 7x7 adaptive
    pooling), and a convolutional feed-forward network (a 3x3 depthwise conv between the
    two Dense layers). It uses no learned position embeddings, so variable input
    resolution works out of the box. One `zm_config.json` (declaring the canonical
    [`PvtV2ImageClassify`]) sits on each variant's repo; both the backbone and the
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        hidden_sizes (`tuple`, *optional*, defaults to `(32, 64, 160, 256)`):
            Channel width per stage.
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Number of transformer blocks per stage.
        num_attention_heads (`tuple`, *optional*, defaults to `(1, 2, 5, 8)`):
            Attention heads per stage.
        sr_ratios (`tuple`, *optional*, defaults to `(8, 4, 2, 1)`):
            Spatial-reduction ratio of the key/value tokens per stage.
        mlp_ratios (`tuple`, *optional*, defaults to `(8, 8, 4, 4)`):
            Feed-forward hidden expansion per stage.
        linear_attention (`bool`, *optional*, defaults to `False`):
            Use the linear-attention variant (7x7 adaptive pooling + 1x1 conv + GELU,
            plus a ReLU after the first FFN Dense).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.pvt_v2 import PvtV2Config, PvtV2ImageClassify

    >>> configuration = PvtV2Config()
    >>> model = PvtV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "pvt_v2"

    hidden_sizes: tuple = (32, 64, 160, 256)
    depths: tuple = (2, 2, 2, 2)
    num_attention_heads: tuple = (1, 2, 5, 8)
    sr_ratios: tuple = (8, 4, 2, 1)
    mlp_ratios: tuple = (8, 8, 4, 4)
    linear_attention: bool = False
    image_size: int = 224
    num_classes: int = 1000


# Hosted variants -> arch preset. Weights load by Hub repo id (zm_config.json).
PVT_V2_VARIANTS = {
    "pvt_v2_b0": {"model": "pvt_v2_b0", "hf_id": "OpenGVLab/pvt_v2_b0"},
    "pvt_v2_b1": {"model": "pvt_v2_b1", "hf_id": "OpenGVLab/pvt_v2_b1"},
    "pvt_v2_b2": {"model": "pvt_v2_b2", "hf_id": "OpenGVLab/pvt_v2_b2"},
    "pvt_v2_b2_linear": {
        "model": "pvt_v2_b2_linear",
        "hf_id": "OpenGVLab/pvt_v2_b2_linear",
    },
    "pvt_v2_b3": {"model": "pvt_v2_b3", "hf_id": "OpenGVLab/pvt_v2_b3"},
    "pvt_v2_b4": {"model": "pvt_v2_b4", "hf_id": "OpenGVLab/pvt_v2_b4"},
    "pvt_v2_b5": {"model": "pvt_v2_b5", "hf_id": "OpenGVLab/pvt_v2_b5"},
}
