import keras
from keras import layers

from zeromodels.base import BaseModel
from zeromodels.models.efficientnet.efficientnet_model import (
    efficientnet_backbone_feature,
)

from .efficientdet_config import EfficientDetConfig
from .efficientdet_layers import (
    EfficientDetResample,
    FPNCells,
    PredictionHead,
    class_predict_bias,
)

# EfficientNet (width, depth, dropout) coefficients per EfficientDet backbone.
BACKBONE_COEFFS = {
    "efficientnet_b0": (1.0, 1.0, 0.2),
    "efficientnet_b1": (1.0, 1.1, 0.2),
    "efficientnet_b2": (1.1, 1.2, 0.3),
    "efficientnet_b3": (1.2, 1.4, 0.3),
    "efficientnet_b4": (1.4, 1.8, 0.4),
    "efficientnet_b5": (1.6, 2.2, 0.4),
    "efficientnet_b6": (1.8, 2.6, 0.5),
}

CONFIG_ATTRS = (
    "backbone_name",
    "image_size",
    "num_classes",
    "min_level",
    "max_level",
    "num_scales",
    "aspect_ratios",
    "anchor_scale",
    "fpn_num_filters",
    "fpn_cell_repeats",
    "box_class_repeats",
    "act_type",
    "separable_conv",
    "apply_bn_for_resampling",
    "conv_after_downsample",
    "conv_bn_act_pattern",
    "fpn_weight_method",
    "survival_prob",
)


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientDetModel(BaseModel):
    """EfficientDet backbone + BiFPN + heads (raw outputs, Google AutoML).

    An EfficientNet backbone (multi-level features P3-P5) is extended with strided
    resamples to P6-P7, fused by a repeated weighted BiFPN, and read out by shared
    class + box heads. Takes an ``images`` input ``(B, image_size, image_size, 3)``
    and returns ``{"class_outputs", "box_outputs"}``: lists of per-level tensors
    ``(B, H_l, W_l, num_anchors * num_classes)`` and ``(B, H_l, W_l, num_anchors * 4)``
    for levels ``min_level``..``max_level``. Decode with anchors + NMS.

    References:
    - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

    Args:
        See :class:`EfficientDetConfig`. Defaults describe EfficientDet-D0.
        name: String, model name. Defaults to `"EfficientDet"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "efficientdet"
    config_class = EfficientDetConfig

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        image_size=512,
        num_classes=90,
        min_level=3,
        max_level=7,
        num_scales=3,
        aspect_ratios=(1.0, 2.0, 0.5),
        anchor_scale=4.0,
        fpn_num_filters=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        act_type="swish",
        separable_conv=True,
        apply_bn_for_resampling=True,
        conv_after_downsample=False,
        conv_bn_act_pattern=False,
        fpn_weight_method="fastattn",
        survival_prob=None,
        name="EfficientDetModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes_"):
            kwargs.pop(k, None)

        num_anchors = num_scales * len(aspect_ratios)
        width, depth, dropout = BACKBONE_COEFFS[backbone_name]
        level_sizes = [
            (image_size // (2**level), image_size // (2**level))
            for level in range(min_level, max_level + 1)
        ]

        image = layers.Input(shape=(image_size, image_size, 3), name="images")

        # Backbone P3/P4/P5 (strides 8/16/32 -> stages[2:5]).
        stages = efficientnet_backbone_feature(
            image,
            width_coefficient=width,
            depth_coefficient=depth,
            dropout_rate=dropout,
            data_format="channels_last",
            channels_axis=-1,
            return_stages=True,
        )
        feats = [stages[2], stages[3], stages[4]]

        # Coarser levels P6..max_level by strided resampling of the last feature.
        for level in range(6, max_level + 1):
            th, tw = level_sizes[level - min_level]
            feats.append(
                EfficientDetResample(
                    fpn_num_filters,
                    apply_bn=apply_bn_for_resampling,
                    conv_after_downsample=conv_after_downsample,
                    name=f"resample_p{level}",
                )(feats[-1], target_height=th, target_width=tw)
            )

        fpn_feats = FPNCells(
            min_level,
            max_level,
            fpn_num_filters,
            fpn_cell_repeats,
            fpn_weight_method,
            act_type,
            apply_bn_for_resampling,
            conv_after_downsample,
            conv_bn_act_pattern,
            name="fpn_cells",
        )(feats, level_sizes=level_sizes)

        num_levels = max_level - min_level + 1
        class_outputs = PredictionHead(
            num_anchors * num_classes,
            fpn_num_filters,
            box_class_repeats,
            num_levels,
            act_type,
            "class",
            min_level=min_level,
            predict_bias_init=class_predict_bias(),
            name="class_net",
        )(fpn_feats)
        box_outputs = PredictionHead(
            num_anchors * 4,
            fpn_num_filters,
            box_class_repeats,
            num_levels,
            act_type,
            "box",
            min_level=min_level,
            predict_bias_init=0.0,
            name="box_net",
        )(fpn_feats)

        super().__init__(
            inputs=image,
            outputs={"class_outputs": class_outputs, "box_outputs": box_outputs},
            name=name,
            **kwargs,
        )
        for attr in CONFIG_ATTRS:
            setattr(self, attr, locals()[attr])

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in CONFIG_ATTRS})
        config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
