import copy
import math

import keras
from keras import initializers, layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .efficientnetv2_config import EfficientNetV2Config

# The backbone (EfficientNetV2Model) and classifier (EfficientNetV2ImageClassify)
# share the variant's repo, whose zm_config.json declares EfficientNetV2ImageClassify.
EFFICIENTNETV2_HUB_SIBLINGS = frozenset(
    {"EfficientNetV2Model", "EfficientNetV2ImageClassify"}
)

EFFICIENTNETV2_BLOCK_CONFIG = {
    "EfficientNetV2S": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 48,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 6,
            "input_filters": 64,
            "output_filters": 128,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 9,
            "input_filters": 128,
            "output_filters": 160,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 15,
            "input_filters": 160,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2M": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2L": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2XL": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 16,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 24,
            "input_filters": 192,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 32,
            "input_filters": 256,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 512,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    # Shared block config for all B variants (B0, B1, B2, B3)
    "EfficientNetV2B": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}


def round_filters(filters, width_coefficient, min_depth=8, depth_divisor=8):
    """Round filter count by ``width_coefficient`` and snap to a multiple of ``depth_divisor``.

    Args:
        filters: Base filter count to scale.
        width_coefficient: Multiplier applied to ``filters`` before rounding.
        min_depth: Minimum allowed channel count (falls back to ``depth_divisor``).
        depth_divisor: Multiple to which the rounded count is snapped.

    Returns:
        Adjusted integer filter count satisfying the divisibility constraint.
    """
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round-up repeat count by ``depth_coefficient``.

    Args:
        repeats: Base number of block repeats.
        depth_coefficient: Depth multiplier applied to ``repeats``.

    Returns:
        Integer repeat count after ceiling.
    """
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(
    inputs,
    input_filters,
    output_filters,
    channels_axis,
    data_format,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """Mobile Inverted Residual Block (MBConv) with optional SE and stochastic depth.

    Args:
        inputs: Input feature tensor.
        input_filters: Number of input channels.
        output_filters: Number of output channels.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        expand_ratio: Expansion factor for the inverted residual.
        kernel_size: Depthwise convolution kernel size.
        strides: Depthwise convolution stride.
        se_ratio: Squeeze-and-Excitation ratio; SE is skipped when ``<= 0``.
        survival_probability: Stochastic-depth drop rate applied on the residual branch.
        block_idx: Stage index used to construct unique layer names.
        layer_idx: Within-stage block index used to construct unique layer names.

    Returns:
        Output feature tensor with ``output_filters`` channels.
    """
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=block_name + "MBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            name=block_name + "batchnorm1",
        )(x)
        x = layers.Activation("swish", name=block_name + "act1")(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "MBdwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    x = layers.Activation("swish", name=block_name + "act2")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(
            data_format=data_format, name=block_name + "se_avgpool"
        )(x)
        if channels_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "MBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm3"
    )(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


def fusedmb_conv_block(
    inputs,
    input_filters,
    output_filters,
    channels_axis,
    data_format,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """Fused MBConv block: replaces the expand-then-depthwise pair with a single ``kxk`` conv.

    Args:
        inputs: Input feature tensor.
        input_filters: Number of input channels.
        output_filters: Number of output channels.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        expand_ratio: Expansion factor for the inverted residual.
        kernel_size: Convolution kernel size for the fused expansion conv.
        strides: Convolution stride for the fused expansion conv.
        se_ratio: Squeeze-and-Excitation ratio; SE is skipped when ``<= 0``.
        survival_probability: Stochastic-depth drop rate applied on the residual branch.
        block_idx: Stage index used to construct unique layer names.
        layer_idx: Within-stage block index used to construct unique layer names.

    Returns:
        Output feature tensor with ``output_filters`` channels.
    """
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=block_name + "FMBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, name=block_name + "batchnorm1"
        )(x)
        x = layers.Activation(activation="swish", name=block_name + "act1")(x)
    else:
        x = inputs

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(
            data_format=data_format, name=block_name + "se_avgpool"
        )(x)
        if channels_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)

        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1 if expand_ratio != 1 else strides,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "FMBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    if expand_ratio == 1:
        x = layers.Activation(activation="swish", name=block_name + "act2")(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


def efficientnetv2_backbone_feature(
    inputs,
    *,
    width_coefficient,
    depth_coefficient,
    block_arch,
    head_filters,
    data_format,
    channels_axis,
    return_stages=False,
):
    """EfficientNetV2 stem + Fused/MBConv stages + head conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        width_coefficient: Filter-count multiplier.
        depth_coefficient: Depth multiplier applied to block repeats.
        block_arch: Key into ``EFFICIENTNETV2_BLOCK_CONFIG`` selecting the variant
            (e.g. ``"EfficientNetV2S"``, ``"EfficientNetV2M"``).
        head_filters: Output channel count of the final 1x1 head conv.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-head-conv); otherwise return the post-head-conv
            tensor.

    Returns:
        Final 4D feature tensor after the head 1x1 conv (post BN + swish), or a
        list of per-stage feature tensors when ``return_stages`` is True.
    """
    block_config = copy.deepcopy(EFFICIENTNETV2_BLOCK_CONFIG[block_arch])

    stem_filters = round_filters(
        filters=block_config[0]["input_filters"],
        width_coefficient=width_coefficient,
    )
    x = layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="conv_stem",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        name="batchnorm1",
    )(x)
    x = layers.Activation("swish", name="act1")(x)

    b = 0
    blocks = float(sum(args["num_repeat"] for args in block_config))

    stages = []
    for i, args in enumerate(block_config):
        assert args["num_repeat"] > 0

        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
        )

        group_stride = args["strides"]
        if return_stages and group_stride == 2:
            stages.append(x)

        block = {0: mb_conv_block, 1: fusedmb_conv_block}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
        )
        for j in range(repeats):
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = block(
                x,
                survival_probability=0.2 * b / blocks,
                block_idx=i,
                layer_idx=j,
                data_format=data_format,
                channels_axis=channels_axis,
                **args,
            )
            b += 1

    if return_stages:
        stages.append(x)
        return stages

    x = layers.Conv2D(
        filters=head_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        use_bias=False,
        padding="same",
        data_format=data_format,
        name="conv_head",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        name="batchnorm2",
    )(x)
    x = layers.Activation(activation="swish", name="act2")(x)

    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientNetV2Model(BaseModel):
    """Instantiates the EfficientNetV2 backbone.

    EfficientNetV2 is the faster-training successor to EfficientNet. It
    swaps the first few low-resolution MBConv stages for Fused-MBConv
    blocks (which collapse the expand + depthwise pair into a single
    3x3 conv) to improve early-stage throughput, introduces
    progressive image-size scaling during training, and adds improved
    regularization. The published variants S, M, L, and XL cover a wide
    capacity range, with the XL variant being pretrained on ImageNet21k
    before ImageNet1k fine-tuning.

    Output is the last layer output before the classifier head: the
    post-head-conv 4D feature map of shape ``(B, H, W, C)``.
    :class:`EfficientNetV2ImageClassify` composes this model and adds a
    GlobalAveragePooling2D + (optional) Dropout + Dense head on top.

    References:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

    Args:
        width_coefficient: Float, filter-count multiplier applied to
            every stage's channel widths. Defaults to `1.0`.
        depth_coefficient: Float, depth multiplier applied to per-stage
            block repeats. Defaults to `1.0`.
        default_size: Integer, the original training resolution for the
            selected variant (kept for reference / config). Defaults to
            `300`.
        block_arch: String, key into the internal block-config table
            selecting the variant. One of ``"EfficientNetV2S"``,
            ``"EfficientNetV2M"``, ``"EfficientNetV2L"``,
            ``"EfficientNetV2XL"``, or ``"EfficientNetV2B"``.
            Defaults to `"EfficientNetV2S"`.
        head_filters: Integer, output channel count of the final 1x1
            head conv. Defaults to `1280`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `300`.
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
            feature maps grouped by stride boundary (pre-head-conv).
            Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"EfficientNetV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientNetV2Config
    HUB_REPO_SIBLINGS = EFFICIENTNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with EfficientNetV2ImageClassify (which
        # the zm_config declares); build from zm_config, then copy backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = EfficientNetV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientnetv2_timm_to_keras import (
            transfer_efficientnetv2_weights,
        )

        transfer_efficientnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=300,
        block_arch="EfficientNetV2S",
        head_filters=1280,
        image_size=300,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="EfficientNetV2Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

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
        x = efficientnetv2_backbone_feature(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            block_arch=block_arch,
            head_filters=head_filters,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.block_arch = block_arch
        self.head_filters = head_filters
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "block_arch": self.block_arch,
                "head_filters": self.head_filters,
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
class EfficientNetV2ImageClassify(BaseModel):
    """Instantiates the EfficientNetV2 classifier.

    This classifier wraps a :class:`EfficientNetV2Model` backbone and
    attaches a GlobalAveragePooling2D + (optional) Dropout + Dense head
    to produce ``num_classes`` class logits. All architectural parameters
    are forwarded to the underlying :class:`EfficientNetV2Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

    Args:
        width_coefficient: Float, filter-count multiplier applied to
            every stage's channel widths. Defaults to `1.0`.
        depth_coefficient: Float, depth multiplier applied to per-stage
            block repeats. Defaults to `1.0`.
        default_size: Integer, the original training resolution for the
            selected variant (kept for reference / config). Defaults to
            `300`.
        block_arch: String, key into the internal block-config table
            selecting the variant. One of ``"EfficientNetV2S"``,
            ``"EfficientNetV2M"``, ``"EfficientNetV2L"``,
            ``"EfficientNetV2XL"``, or ``"EfficientNetV2B"``.
            Defaults to `"EfficientNetV2S"`.
        head_filters: Integer, output channel count of the final 1x1
            head conv inside the backbone. Defaults to `1280`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `300`.
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
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to
            `"EfficientNetV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientNetV2Config
    HUB_REPO_SIBLINGS = EFFICIENTNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientnetv2_timm_to_keras import (
            transfer_efficientnetv2_weights,
        )

        transfer_efficientnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=300,
        block_arch="EfficientNetV2S",
        head_filters=1280,
        image_size=300,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        dropout_rate=0.2,
        name="EfficientNetV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = EfficientNetV2Model(
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            default_size=default_size,
            block_arch=block_arch,
            head_filters=head_filters,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=initializers.Constant(0.0),
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.block_arch = block_arch
        self.head_filters = head_filters
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
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "block_arch": self.block_arch,
                "head_filters": self.head_filters,
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
