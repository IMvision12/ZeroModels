import copy
import math

import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .efficientnet_config import EfficientNetConfig

# The backbone (EfficientNetModel) and classifier (EfficientNetImageClassify) share
# the variant's weights repo, whose kf_config.json declares EfficientNetImageClassify.
EFFICIENTNET_HUB_SIBLINGS = frozenset(
    {"EfficientNetModel", "EfficientNetImageClassify"}
)

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


def round_filters(filters, width_coefficient, divisor=8):
    """Round filter count by ``width_coefficient`` and snap to a multiple of ``divisor``.

    Args:
        filters: Base filter count to scale.
        width_coefficient: Multiplier applied to ``filters`` before rounding.
        divisor: Multiple to which the rounded count is snapped.

    Returns:
        Adjusted integer filter count satisfying the divisibility constraint.
    """
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
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


def efficientnet_block(
    inputs,
    channels_axis,
    data_format,
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """MBConv block with optional Squeeze-and-Excitation and residual skip.

    Args:
        inputs: Input feature tensor.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string (``"channels_last"`` or ``"channels_first"``).
        drop_rate: Dropout rate applied to the residual branch when the skip is active.
        name: Prefix used to name the layers inside the block.
        filters_in: Number of input channels.
        filters_out: Number of output channels.
        kernel_size: Depthwise convolution kernel size.
        strides: Depthwise convolution stride.
        expand_ratio: Expansion factor for the inverted residual.
        se_ratio: Squeeze-and-Excitation ratio; SE is skipped when ``<= 0``.
        id_skip: Whether to add the identity skip connection (when shapes match).

    Returns:
        Output feature tensor with ``filters_out`` channels.
    """
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "conv2d_1",
        )(inputs)
        x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_1")(x)
        x = layers.Activation("swish")(x)
    else:
        x = inputs

    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            data_format=data_format,
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "dwconv2d",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_2")(x)
    x = layers.Activation("swish")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(data_format=data_format)(x)
        se_shape = (filters, 1, 1) if channels_axis == 1 else (1, 1, filters)
        se = layers.Reshape(se_shape)(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "se_conv_expand",
        )(se)
        x = layers.multiply([x, se])

    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "conv2d_2",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_3")(x)

    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs])
    return x


def efficientnet_backbone_feature(
    inputs,
    *,
    width_coefficient,
    depth_coefficient,
    dropout_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """EfficientNet stem + 7 MBConv stages + head conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        width_coefficient: Filter-count multiplier.
        depth_coefficient: Depth multiplier applied to block repeats.
        dropout_rate: Stochastic-depth drop rate ramp applied across blocks.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-head-conv); otherwise return the post-head-conv
            tensor.

    Returns:
        Final 4D feature tensor after the head 1x1 conv (post BN + swish), or a
        list of per-stage feature tensors when ``return_stages`` is True.
    """
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(inputs, 3), data_format=data_format
    )(inputs)
    x = layers.Conv2D(
        round_filters(32, width_coefficient=width_coefficient),
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name="conv_stem",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_1")(x)
    x = layers.Activation("swish")(x)

    b = 0
    blocks = float(
        sum(
            round_repeats(args["repeats"], depth_coefficient=depth_coefficient)
            for args in DEFAULT_BLOCKS_ARGS
        )
    )

    stages = []
    for i, block_args in enumerate(DEFAULT_BLOCKS_ARGS):
        args = copy.deepcopy(block_args)
        args["filters_in"] = round_filters(
            args["filters_in"], width_coefficient=width_coefficient
        )
        args["filters_out"] = round_filters(
            args["filters_out"], width_coefficient=width_coefficient
        )
        repeats = round_repeats(args["repeats"], depth_coefficient=depth_coefficient)
        group_stride = args["strides"]
        del args["repeats"]

        # When this group downsamples (stride=2), the prior accumulated feature
        # belongs to the previous stage.
        if return_stages and group_stride == 2:
            stages.append(x)

        for j in range(repeats):
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = efficientnet_block(
                x,
                channels_axis,
                data_format,
                dropout_rate * b / blocks,
                name=f"blocks_{i}_{j}_",
                **args,
            )
            b += 1

    if return_stages:
        stages.append(x)
        return stages

    x = layers.Conv2D(
        round_filters(1280, width_coefficient=width_coefficient),
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name="conv_head",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_2")(x)
    x = layers.Activation("swish")(x)

    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class EfficientNetModel(BaseModel):
    """Instantiates the EfficientNet backbone.

    EfficientNet is a compound-scaled MBConv-based CNN that uniformly
    scales depth, width, and resolution by a single coefficient. The
    network consists of a 3x3 conv stem, seven stages of mobile inverted
    bottleneck (MBConv) blocks with Squeeze-and-Excitation, and a 1x1
    head conv (post BN + swish). Variants B0-B8 plus L2 span the full
    range from tiny mobile-scale models to massive high-capacity ones.

    Output is the last layer output before the classifier head: the
    post-head-conv 4D feature map of shape ``(B, H, W, C)``.
    :class:`EfficientNetImageClassify` composes this model and adds a
    GlobalAveragePooling2D + (optional) Dropout + Dense head on top.

    References:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

    Args:
        width_coefficient: Float, filter-count multiplier applied to every
            stage's channel widths. Defaults to `1.0`.
        depth_coefficient: Float, depth multiplier applied to per-stage
            block repeats. Defaults to `1.0`.
        dropout_rate: Float, stochastic-depth drop rate ramped linearly
            across the MBConv blocks. Defaults to `0.2`.
        default_size: Integer, the original training resolution for the
            selected variant (kept for reference / config). Defaults to
            `224`.
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
            use. Must be one of: `'imagenet'` (default), `'inception'`,
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
            Defaults to `"EfficientNetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientNetConfig
    HUB_REPO_SIBLINGS = EFFICIENTNET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with EfficientNetImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = EfficientNetImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientnet_timm_to_keras import transfer_efficientnet_weights

        transfer_efficientnet_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.2,
        default_size=224,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="EfficientNetModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation"):
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
        x = efficientnet_backbone_feature(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            dropout_rate=dropout_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
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
                "dropout_rate": self.dropout_rate,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class EfficientNetImageClassify(BaseModel):
    """Instantiates the EfficientNet classifier.

    This classifier wraps a :class:`EfficientNetModel` backbone and
    attaches a GlobalAveragePooling2D + (optional) Dropout + Dense head
    to produce ``num_classes`` class logits. All architectural parameters
    are forwarded to the underlying :class:`EfficientNetModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

    Args:
        width_coefficient: Float, filter-count multiplier applied to every
            stage's channel widths. Defaults to `1.0`.
        depth_coefficient: Float, depth multiplier applied to per-stage
            block repeats. Defaults to `1.0`.
        dropout_rate: Float, stochastic-depth drop rate ramped across the
            MBConv blocks, also used for the head Dropout. Defaults to
            `0.2`.
        default_size: Integer, the original training resolution for the
            selected variant (kept for reference / config). Defaults to
            `224`.
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
            use. Must be one of: `'imagenet'` (default), `'inception'`,
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
            `"EfficientNetImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientNetConfig
    HUB_REPO_SIBLINGS = EFFICIENTNET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientnet_timm_to_keras import transfer_efficientnet_weights

        transfer_efficientnet_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.2,
        default_size=224,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="EfficientNetImageClassify",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()

        backbone = EfficientNetModel(
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            dropout_rate=dropout_rate,
            default_size=default_size,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="dropout")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "dropout_rate": self.dropout_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
