from zeromodels.base import BaseConfig


class EfficientDetConfig(BaseConfig):
    r"""Configuration for EfficientDet (Google AutoML) object detection.

    EfficientDet = an EfficientNet backbone -> a weighted bidirectional feature
    pyramid (BiFPN, repeated ``fpn_cell_repeats`` times) -> shared class + box
    prediction heads run over pyramid levels ``min_level``..``max_level``. Defaults
    describe **EfficientDet-D0**; the D1-D7 variants override a handful of fields.
    Pretrained variants load their architecture from the hosted ``zm_config.json`` via
    ``from_weights``; the per-variant overrides used only when converting Google's
    config-less ``.h5`` live in ``convert_efficientdet_h5_to_keras.EFFICIENTDET_RECIPES``.

    Args:
        backbone_name (`str`, *optional*, defaults to `"efficientnet_b0"`):
            zeromodels EfficientNet backbone variant (the ``tf_efficientnet`` flavor
            matches Google's backbone). D0->b0, D1->b1, ... D6/D7->b6.
        image_size (`int`, *optional*, defaults to `512`):
            Square input resolution.
        num_classes (`int`, *optional*, defaults to `90`):
            COCO class count (background excluded).
        min_level / max_level (`int`, *optional*, defaults to `3` / `7`):
            Pyramid levels the BiFPN + heads run over.
        num_scales (`int`, *optional*, defaults to `3`):
            Anchor scales per level; ``num_anchors = num_scales * len(aspect_ratios)``.
        aspect_ratios (`tuple`, *optional*, defaults to `(1.0, 2.0, 0.5)`):
            Anchor aspect ratios.
        anchor_scale (`float`, *optional*, defaults to `4.0`):
            Base anchor size in units of the feature stride.
        fpn_num_filters (`int`, *optional*, defaults to `64`):
            BiFPN (and head) channel width.
        fpn_cell_repeats (`int`, *optional*, defaults to `3`):
            Number of stacked BiFPN cells.
        box_class_repeats (`int`, *optional*, defaults to `3`):
            Shared conv layers in each of the class / box heads.
        act_type (`str`, *optional*, defaults to `"swish"`):
            Activation used throughout the BiFPN and heads.
        separable_conv (`bool`, *optional*, defaults to `True`):
            Use depthwise-separable 3x3 convs (as EfficientDet does).
        apply_bn_for_resampling (`bool`, *optional*, defaults to `True`):
            BatchNorm after the 1x1 channel-match in resampling.
        conv_after_downsample (`bool`, *optional*, defaults to `False`):
            Apply the 1x1 channel-match after (vs before) the downsample pool.
        conv_bn_act_pattern (`bool`, *optional*, defaults to `False`):
            `False` => act -> conv -> BN in a BiFPN node; `True` => conv -> BN -> act.
        fpn_weight_method (`str`, *optional*, defaults to `"fastattn"`):
            BiFPN edge fusion: `"fastattn"` (fast normalized), `"attn"` (softmax),
            or `"sum"`.
        survival_prob (`float`, *optional*):
            Drop-connect keep prob in the heads (inference ignores it).

    Example:

    ```python
    >>> from zeromodels.models.efficientdet import EfficientDetConfig

    >>> config = EfficientDetConfig(backbone_name="efficientnet_b2", image_size=768)
    ```"""

    model_type = "efficientdet"

    backbone_name: str = "efficientnet_b0"
    image_size: int = 512
    num_classes: int = 90
    min_level: int = 3
    max_level: int = 7
    num_scales: int = 3
    aspect_ratios: tuple = (1.0, 2.0, 0.5)
    anchor_scale: float = 4.0
    fpn_num_filters: int = 64
    fpn_cell_repeats: int = 3
    box_class_repeats: int = 3
    act_type: str = "swish"
    separable_conv: bool = True
    apply_bn_for_resampling: bool = True
    conv_after_downsample: bool = False
    conv_bn_act_pattern: bool = False
    fpn_weight_method: str = "fastattn"
    survival_prob: float = None
