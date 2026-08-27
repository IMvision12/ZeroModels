from zeromodels.base import BaseConfig


class EfficientDetConfig(BaseConfig):
    r"""Configuration for EfficientDet (Google AutoML) object detection.

    EfficientDet = an EfficientNet backbone -> a weighted bidirectional feature
    pyramid (BiFPN, repeated ``fpn_cell_repeats`` times) -> shared class + box
    prediction heads run over pyramid levels ``min_level``..``max_level``. Defaults
    describe **EfficientDet-D0**; the D1-D7 variants override a handful of fields
    (see :data:`EFFICIENTDET_RECIPES`).

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
    >>> from zeromodels.models.efficientdet import EfficientDetConfig, EFFICIENTDET_RECIPES

    >>> config = EfficientDetConfig(**EFFICIENTDET_RECIPES["efficientdet_d2"])
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


# Per-variant overrides of the D0 defaults (Google's efficientdet-d0..d7 recipes).
EFFICIENTDET_RECIPES = {
    "efficientdet_d0": {
        "backbone_name": "efficientnet_b0",
        "image_size": 512,
        "fpn_num_filters": 64,
        "fpn_cell_repeats": 3,
        "box_class_repeats": 3,
    },
    "efficientdet_d1": {
        "backbone_name": "efficientnet_b1",
        "image_size": 640,
        "fpn_num_filters": 88,
        "fpn_cell_repeats": 4,
        "box_class_repeats": 3,
    },
    "efficientdet_d2": {
        "backbone_name": "efficientnet_b2",
        "image_size": 768,
        "fpn_num_filters": 112,
        "fpn_cell_repeats": 5,
        "box_class_repeats": 3,
    },
    "efficientdet_d3": {
        "backbone_name": "efficientnet_b3",
        "image_size": 896,
        "fpn_num_filters": 160,
        "fpn_cell_repeats": 6,
        "box_class_repeats": 4,
    },
    "efficientdet_d4": {
        "backbone_name": "efficientnet_b4",
        "image_size": 1024,
        "fpn_num_filters": 224,
        "fpn_cell_repeats": 7,
        "box_class_repeats": 4,
    },
    "efficientdet_d5": {
        "backbone_name": "efficientnet_b5",
        "image_size": 1280,
        "fpn_num_filters": 288,
        "fpn_cell_repeats": 7,
        "box_class_repeats": 4,
    },
    "efficientdet_d6": {
        "backbone_name": "efficientnet_b6",
        "image_size": 1280,
        "fpn_num_filters": 384,
        "fpn_cell_repeats": 8,
        "box_class_repeats": 5,
        # D6/D7 fuse BiFPN inputs with an unweighted sum (no per-edge weights).
        "fpn_weight_method": "sum",
    },
    "efficientdet_d7": {
        "backbone_name": "efficientnet_b6",
        "image_size": 1536,
        "fpn_num_filters": 384,
        "fpn_cell_repeats": 8,
        "box_class_repeats": 5,
        "anchor_scale": 5.0,
        "fpn_weight_method": "sum",
    },
}


def bifpn_nodes(min_level, max_level):
    """Google's dynamic BiFPN node topology for ``min_level``..``max_level``.

    Each node lists the offsets (into the growing feature list) it fuses; a
    top-down path (P{max-1}'..P{min}') then a bottom-up path (P{min+1}''..P{max}'').
    """
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    counter = num_levels
    nodes = []
    for level in range(max_level - 1, min_level - 1, -1):  # top-down
        nodes.append(
            {
                "feat_level": level,
                "inputs_offsets": [node_ids[level][-1], node_ids[level + 1][-1]],
            }
        )
        node_ids[level].append(counter)
        counter += 1
    for level in range(min_level + 1, max_level + 1):  # bottom-up
        nodes.append(
            {
                "feat_level": level,
                "inputs_offsets": node_ids[level] + [node_ids[level - 1][-1]],
            }
        )
        node_ids[level].append(counter)
        counter += 1
    return nodes
