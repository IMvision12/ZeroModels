import math

import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .mobilenetv3_config import MobileNetV3Config

# The backbone (MobileNetV3Model) and classifier (MobileNetV3ImageClassify) share the
# variant's repo, whose zm_config.json declares MobileNetV3ImageClassify.
MOBILENETV3_HUB_SIBLINGS = frozenset({"MobileNetV3Model", "MobileNetV3ImageClassify"})


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Args:
        v: Value to be adjusted.
        divisor: Multiple to which ``v`` should be rounded.
        min_value: Minimum allowed value (defaults to ``divisor``).
        round_limit: Lower-bound ratio that triggers bumping the result up by
            one ``divisor`` when rounding-down went too far.

    Returns:
        Adjusted value divisible by ``divisor`` and at least ``min_value``.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    x,
    expansion_ratio,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    block_id,
    data_format,
    channels_axis,
    noskip=False,
    se_round_divisor=8,
    se_activation="relu",
    bn_epsilon=1e-5,
):
    """MobileNetV3-style inverted residual block with optional Squeeze-and-Excitation.

    Args:
        x: Input feature tensor.
        expansion_ratio: Expansion factor applied to the input channel count.
        filters: Output channel count after the projection conv.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        se_ratio: Squeeze-and-Excitation ratio (or ``None``/0 to disable).
        activation: Activation name used after the expand and depthwise convs.
        block_id: Block index used to construct unique layer names.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        noskip: If True, never add the residual connection even when shapes
            allow it. Defaults to False.
        se_round_divisor: Divisor used to round the SE reduce channel count
            via :func:`make_divisible`. If ``None``, the SE reduce channel
            count is computed as ``int(expanded_filters * se_ratio)`` with no
            rounding (matches the timm ``rw`` variant). Defaults to ``8``.
        se_activation: Activation used inside the SE block (between the two
            1x1 SE convs). Defaults to ``"relu"`` (matches timm's standard
            mobilenetv3 which forces ReLU via ``force_act_layer=nn.ReLU``).
            The timm ``rw`` variant omits ``force_act_layer`` so the SE
            activation follows the block's own activation: callers should
            pass ``activation`` (the block's act) in that case.

    Returns:
        Output feature tensor with ``filters`` channels.
    """
    shortcut = x
    prefix = f"ir_block_{block_id}"
    input_filters = x.shape[channels_axis]
    expanded_filters = make_divisible(input_filters * expansion_ratio)

    if expansion_ratio != 1:
        x = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=0.999,
            name=f"{prefix}_batchnorm_1",
        )(x)
        x = layers.Activation(activation, name=f"{prefix}_activation_1")(x)

    if stride > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(
            data_format=data_format, padding=((pad, pad), (pad, pad))
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.999,
        name=f"{prefix}_batchnorm_2",
    )(x)
    x = layers.Activation(activation, name=f"{prefix}_activation_2")(x)

    if se_ratio:
        if se_round_divisor is None:
            se_filters = int(expanded_filters * se_ratio)
        else:
            se_filters = make_divisible(
                expanded_filters * se_ratio, divisor=se_round_divisor
            )
        x_se = layers.GlobalAveragePooling2D(
            keepdims=True, data_format=data_format, name=f"{prefix}_se_pool"
        )(x)
        x_se = layers.Conv2D(
            se_filters,
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_1",
        )(x_se)
        x_se = layers.Activation(se_activation, name=f"{prefix}_se_activation_1")(x_se)
        x_se = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_2",
        )(x_se)
        x_se = layers.Activation("hard_sigmoid", name=f"{prefix}_se_activation_2")(x_se)
        x = layers.Multiply(name=f"{prefix}_se_multiply")([x, x_se])

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.999,
        name=f"{prefix}_batchnorm_3",
    )(x)

    if not noskip and stride == 1 and input_filters == filters:
        x = layers.Add(name=f"{prefix}_add")([shortcut, x])
    return x


# Each stage is a list of sub-groups; each sub-group is
# (expansion_ratio, filters, kernel_size, stride, se_ratio, activation, repeat_count).
# Layout mirrors timm's ``arch_def`` (groups within a stage) so that
# ``block_count_multiplier`` is distributed across sub-groups via
# :func:`scale_stage_depth` (matching timm's ``_scale_stage_depth``).
# Only the first block of a stage carries its stride; all subsequent blocks
# in the stage are forced to stride=1.
_SMALL_STAGE_GROUPS = [
    [(1, 16, 3, 2, 0.25, "relu", 1)],
    [(72.0 / 16, 24, 3, 2, None, "relu", 1), (88.0 / 24, 24, 3, 1, None, "relu", 1)],
    [(4, 40, 5, 2, 0.25, "hard_swish", 1), (6, 40, 5, 1, 0.25, "hard_swish", 2)],
    [(3, 48, 5, 1, 0.25, "hard_swish", 2)],
    [(6, 96, 5, 2, 0.25, "hard_swish", 1), (6, 96, 5, 1, 0.25, "hard_swish", 2)],
]

_LARGE_STAGE_GROUPS = [
    [(1, 16, 3, 1, None, "relu", 1)],
    [(4, 24, 3, 2, None, "relu", 1), (3, 24, 3, 1, None, "relu", 1)],
    [(3, 40, 5, 2, 0.25, "relu", 3)],
    [
        (6, 80, 3, 2, None, "hard_swish", 1),
        (2.5, 80, 3, 1, None, "hard_swish", 1),
        (2.3, 80, 3, 1, None, "hard_swish", 2),
    ],
    [(6, 112, 3, 1, 0.25, "hard_swish", 2)],
    [(6, 160, 5, 2, 0.25, "hard_swish", 3)],
]


def scale_stage_depth(repeats, depth_multiplier):
    """Replicate timm's ``_scale_stage_depth``.

    Scale the total repeat count of a stage by ``depth_multiplier`` (ceil)
    and distribute the scaled total back across the stage's sub-groups in
    reverse order, so the first sub-group is least likely to grow. Returns
    a per-sub-group list of repeats.
    """
    num_repeat = sum(repeats)
    num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round(r / num_repeat * num_repeat_scaled))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    return repeats_scaled[::-1]


def mobilenetv3_backbone_feature(
    inputs,
    *,
    config,
    width_multiplier,
    depth_multiplier,
    minimal,
    data_format,
    channels_axis,
    return_stages=False,
    block_count_multiplier=1.0,
    head_count_multiplier=1,
    first_block_noskip=False,
    se_round_divisor=8,
    se_use_block_act=False,
    bn_epsilon=1e-5,
):
    """MobileNetV3 stem + inverted-residual stages + final 1x1 conv head.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        config: Variant key, ``"large"`` or ``"small"``, selecting the stage table.
        width_multiplier: Multiplier applied to per-stage channel counts.
        depth_multiplier: Multiplier applied to per-block expansion ratios.
        minimal: If True, force kernel size 3, ReLU activations, and disable SE
            for every IR block (minimal variant).
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-final-conv); otherwise return the
            post-final-conv tensor.
        block_count_multiplier: Multiplier applied to the per-stage repeat
            counts. The total repeat count for each stage is scaled (ceil)
            and then redistributed across the stage's sub-groups via
            :func:`scale_stage_depth` (timm's ``_scale_stage_depth``). Used
            by timm's ``large_150d`` variant. Defaults to ``1.0``.
        head_count_multiplier: Number of times the final 1x1 conv head
            (conv + BN + activation) is repeated. Used by timm's ``large_150d``
            variant whose head has two consecutive 1x1 convs. Defaults to ``1``.
        first_block_noskip: If True, disable the residual add in the very first
            IR block of the network (block_id 0). Used by timm's ``rw`` variant
            whose first depthwise-separable block carries ``noskip``.
            Defaults to ``False``.
        se_round_divisor: Divisor for SE reduce-channel rounding. ``None``
            disables rounding (``int(c * r)`` is used directly). Defaults to ``8``.
        se_use_block_act: If True, the SE inner activation follows the block's
            own activation (matches timm's ``rw`` variant which omits
            ``force_act_layer``). If False, the SE inner activation is forced
            to ReLU (matches timm's standard mobilenetv3 ``se_layer`` partial).
            Defaults to ``False``.
        bn_epsilon: Epsilon used for every BatchNormalization layer in the
            network. The timm ``mobilenetv3_rw`` registration calls
            ``kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT=1e-3)`` so rw uses
            1e-3 instead of the PyTorch default 1e-5. Defaults to ``1e-5``.

    Returns:
        Final 4D feature tensor after the final 1x1 conv (post BN + activation),
        or a list of per-stage feature tensors when ``return_stages`` is True.
    """
    stage_groups = _LARGE_STAGE_GROUPS if config == "large" else _SMALL_STAGE_GROUPS

    # Match timm: stem channels are kept fixed at 16 when width_multiplier < 0.75
    # (see ``fix_stem`` in timm.models.mobilenetv3._gen_mobilenet_v3).
    if width_multiplier < 0.75:
        stem_channels = 16
    else:
        stem_channels = make_divisible(16 * width_multiplier)

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), data_format=data_format, name="stem_padding"
    )(inputs)
    x = layers.Conv2D(
        stem_channels,
        kernel_size=3,
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.999,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation(
        "hard_swish" if not minimal else "relu", name="stem_activation"
    )(x)

    stage_outputs = []
    block_id = 0
    for groups in stage_groups:
        repeats = [g[6] for g in groups]
        scaled_repeats = scale_stage_depth(repeats, block_count_multiplier)

        stage_block_configs = []
        for group, rep in zip(groups, scaled_repeats):
            expansion_ratio, filters, kernel_size, stride, se_ratio, activation, _ = (
                group
            )
            for _ in range(rep):
                stage_block_configs.append(
                    (
                        expansion_ratio,
                        filters,
                        kernel_size,
                        stride,
                        se_ratio,
                        activation,
                    )
                )

        for blk_idx, layer_config in enumerate(stage_block_configs):
            (
                expansion_ratio,
                filters,
                kernel_size,
                stride,
                se_ratio,
                activation,
            ) = layer_config
            # Match timm: only the first block of each stage carries its
            # stride; every subsequent block in the stage is forced to 1.
            if blk_idx >= 1:
                stride = 1
            if minimal:
                kernel_size = 3
                activation = "relu"
                se_ratio = None

            if return_stages and stride == 2:
                stage_outputs.append(x)

            noskip = first_block_noskip and block_id == 0
            x = inverted_residual_block(
                x,
                expansion_ratio=expansion_ratio * depth_multiplier,
                filters=make_divisible(filters * width_multiplier),
                kernel_size=kernel_size,
                stride=stride,
                se_ratio=se_ratio,
                activation=activation,
                block_id=block_id,
                data_format=data_format,
                channels_axis=channels_axis,
                noskip=noskip,
                se_round_divisor=se_round_divisor,
                se_activation=activation if se_use_block_act else "relu",
                bn_epsilon=bn_epsilon,
            )
            block_id += 1

    if return_stages:
        stage_outputs.append(x)
        return stage_outputs

    final_conv_channels = make_divisible(x.shape[channels_axis] * 6)
    for head_idx in range(head_count_multiplier):
        suffix = "" if head_count_multiplier == 1 else f"_{head_idx}"
        x = layers.Conv2D(
            final_conv_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"final_conv{suffix}",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=0.999,
            name=f"final_batchnorm{suffix}",
        )(x)
        x = layers.Activation(
            "hard_swish" if not minimal else "relu", name=f"final_activation{suffix}"
        )(x)

    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV3Model(BaseModel):
    """Instantiates the MobileNetV3 backbone.

    MobileNetV3 is a NAS-optimized refinement of MobileNetV2 that mixes
    in Squeeze-and-Excitation modules on selected blocks, replaces ReLU
    with h-swish (hard swish) in the later stages, and applies
    architecture-specific tweaks to produce two main variants
    (``"large"`` and ``"small"``) tuned for different latency budgets.
    The network is composed of a 3x3 conv stem, a sequence of NAS-tuned
    inverted-residual blocks (with optional SE), and a 1x1 final conv
    whose output channel count is six times the last block's channel
    width (post BN + activation).

    Output is the last layer output before the classifier head: the
    post-final-conv 4D feature map of shape ``(B, H, W, C)``.
    :class:`MobileNetV3ImageClassify` composes this model and adds a
    GlobalAveragePooling2D + Dense + activation + Dropout + Dense
    classifier head on top.

    References:
    - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

    Args:
        width_multiplier: Float, multiplier applied to per-stage channel
            counts. Defaults to `1.0`.
        depth_multiplier: Float, multiplier applied to per-block
            expansion ratios. Defaults to `1.0`.
        config: String, variant key selecting the block table. One of
            ``"large"`` or ``"small"``. Defaults to `"large"`.
        minimal: Boolean, if True force kernel size 3, ReLU activations,
            and disable SE for every IR block (minimal variant for
            hardware that lacks h-swish / SE support). Defaults to
            `False`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            feature maps grouped by stride boundary (pre-final-conv).
            Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"MobileNetV3Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV3Config
    HUB_REPO_SIBLINGS = MOBILENETV3_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MobileNetV3ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MobileNetV3ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv3_timm_to_keras import (
            stage_counts,
            transfer_mobilenetv3_weights,
        )

        scounts = stage_counts(keras_model.arch, keras_model.block_count_multiplier)
        transfer_mobilenetv3_weights(
            keras_model, state_dict, scounts, keras_model.head_count_multiplier
        )

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        config="large",
        minimal=False,
        block_count_multiplier=1.0,
        head_count_multiplier=1,
        first_block_noskip=False,
        se_round_divisor=8,
        se_use_block_act=False,
        bn_epsilon=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="MobileNetV3Model",
        **kwargs,
    ):
        for k in (
            "num_classes",
            "classifier_activation",
            "dropout_rate",
            "head_use_bias",
            "timm_id",
        ):
            kwargs.pop(k, None)

        if config not in ("large", "small"):
            raise ValueError(
                f"Invalid config. Expected 'large' or 'small', got {config!r}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = mobilenetv3_backbone_feature(
            x,
            config=config,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            minimal=minimal,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
            block_count_multiplier=block_count_multiplier,
            head_count_multiplier=head_count_multiplier,
            first_block_noskip=first_block_noskip,
            se_round_divisor=se_round_divisor,
            se_use_block_act=se_use_block_act,
            bn_epsilon=bn_epsilon,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.arch = config
        self.minimal = minimal
        self.block_count_multiplier = block_count_multiplier
        self.head_count_multiplier = head_count_multiplier
        self.first_block_noskip = first_block_noskip
        self.se_round_divisor = se_round_divisor
        self.se_use_block_act = se_use_block_act
        self.bn_epsilon = bn_epsilon
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "config": self.arch,
                "minimal": self.minimal,
                "block_count_multiplier": self.block_count_multiplier,
                "head_count_multiplier": self.head_count_multiplier,
                "first_block_noskip": self.first_block_noskip,
                "se_round_divisor": self.se_round_divisor,
                "se_use_block_act": self.se_use_block_act,
                "bn_epsilon": self.bn_epsilon,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV3ImageClassify(BaseModel):
    """Instantiates the MobileNetV3 classifier.

    This classifier wraps a :class:`MobileNetV3Model` backbone and
    attaches a GlobalAveragePooling2D + Dense + activation + Dropout +
    Dense classifier head to produce ``num_classes`` class logits. The
    intermediate Dense projects to ``head_channels`` (1280 for the
    ``"large"`` variant, 1024 for ``"small"``) and is followed by the
    backbone's head activation (h-swish, or ReLU when ``minimal=True``).
    All architectural parameters are forwarded to the underlying
    :class:`MobileNetV3Model`; ``num_classes``, ``classifier_activation``
    and ``dropout_rate`` are head-specific.

    References:
    - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

    Args:
        width_multiplier: Float, multiplier applied to per-stage channel
            counts. Defaults to `1.0`.
        depth_multiplier: Float, multiplier applied to per-block
            expansion ratios. Defaults to `1.0`.
        config: String, variant key selecting the block table and the
            head Dense width (1280 for ``"large"``, 1024 for
            ``"small"``). Defaults to `"large"`.
        minimal: Boolean, if True force kernel size 3, ReLU activations,
            and disable SE for every IR block (also switches the head
            activation from h-swish to ReLU). Defaults to `False`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        dropout_rate: Float, dropout rate applied between the head
            activation and the final Dense classifier (skipped when
            ``<= 0``). Defaults to `0.2`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to
            `"MobileNetV3ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV3Config
    HUB_REPO_SIBLINGS = MOBILENETV3_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv3_timm_to_keras import (
            stage_counts,
            transfer_mobilenetv3_weights,
        )

        scounts = stage_counts(keras_model.arch, keras_model.block_count_multiplier)
        transfer_mobilenetv3_weights(
            keras_model, state_dict, scounts, keras_model.head_count_multiplier
        )

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        config="large",
        minimal=False,
        block_count_multiplier=1.0,
        head_count_multiplier=1,
        first_block_noskip=False,
        se_round_divisor=8,
        se_use_block_act=False,
        bn_epsilon=1e-5,
        head_use_bias=True,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        dropout_rate=0.2,
        name="MobileNetV3ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if config not in ("large", "small"):
            raise ValueError(
                f"Invalid config. Expected 'large' or 'small', got {config!r}"
            )

        data_format = keras.config.image_data_format()

        backbone = MobileNetV3Model(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            config=config,
            minimal=minimal,
            block_count_multiplier=block_count_multiplier,
            head_count_multiplier=head_count_multiplier,
            first_block_noskip=first_block_noskip,
            se_round_divisor=se_round_divisor,
            se_use_block_act=se_use_block_act,
            bn_epsilon=bn_epsilon,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        head_channels = 1024 if config == "small" else 1280
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.Dense(
            head_channels,
            use_bias=head_use_bias,
            name="head_conv",
        )(x)
        x = layers.Activation(
            "hard_swish" if not minimal else "relu", name="head_activation"
        )(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="head_dropout")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.arch = config
        self.minimal = minimal
        self.block_count_multiplier = block_count_multiplier
        self.head_count_multiplier = head_count_multiplier
        self.first_block_noskip = first_block_noskip
        self.se_round_divisor = se_round_divisor
        self.se_use_block_act = se_use_block_act
        self.bn_epsilon = bn_epsilon
        self.head_use_bias = head_use_bias
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "config": self.arch,
                "minimal": self.minimal,
                "block_count_multiplier": self.block_count_multiplier,
                "head_count_multiplier": self.head_count_multiplier,
                "first_block_noskip": self.first_block_noskip,
                "se_round_divisor": self.se_round_divisor,
                "se_use_block_act": self.se_use_block_act,
                "bn_epsilon": self.bn_epsilon,
                "head_use_bias": self.head_use_bias,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "dropout_rate": self.dropout_rate,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
