import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .mobilenetv2_config import MobileNetV2Config

# The backbone (MobileNetV2Model) and classifier (MobileNetV2ImageClassify) share the
# variant's repo, whose zm_config.json declares MobileNetV2ImageClassify.
MOBILENETV2_HUB_SIBLINGS = frozenset({"MobileNetV2Model", "MobileNetV2ImageClassify"})


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """
    Adjusts the given value `v` to be divisible by `divisor`,
        ensuring it meets the specified constraints.

    Args:
        v (int or float): The value to be adjusted.
        divisor (int, optional): The divisor to which `v` should be rounded. Default is 8.
        min_value (int, optional): The minimum allowed value. If None, it defaults to `divisor`.
        round_limit (float, optional): The threshold to increase `new_v` if it is too small.
            Default is 0.9.

    Returns:
        int: The adjusted value that is divisible by `divisor` and meets the
            given constraints.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    x,
    filters,
    kernel_size,
    stride,
    expansion_ratio,
    channels_axis,
    data_format,
    block_id,
    sub_block_id,
):
    """MobileNetV2 inverted-residual block: expand 1x1, depthwise, project 1x1.

    Args:
        x: Input feature tensor.
        filters: Output channel count after the projection conv.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        expansion_ratio: Expansion factor applied to the input channels.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        block_id: Stage index used to construct unique layer names.
        sub_block_id: Within-stage block index used to construct unique layer names.

    Returns:
        Output feature tensor with ``filters`` channels.
    """
    inputs = x
    block_name = f"blocks_{block_id}_{sub_block_id}"

    if expansion_ratio > 1:
        x = layers.Conv2D(
            make_divisible(x.shape[channels_axis] * expansion_ratio),
            1,
            1,
            use_bias=False,
            data_format=data_format,
            name=f"{block_name}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name=f"{block_name}_batchnorm_1",
        )(x)
        x = layers.Activation("relu6", name=f"{block_name}_relu1")(x)

    if stride > 1:
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)),
            data_format=data_format,
            name=f"{block_name}_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_2",
    )(x)
    x = layers.Activation("relu6", name=f"{block_name}_relu2")(x)

    x = layers.Conv2D(
        filters,
        1,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_3",
    )(x)

    if stride == 1 and inputs.shape[channels_axis] == filters:
        x = layers.Add(name=f"{block_name}_add")([inputs, x])

    return x


_DEFAULT_BLOCKS = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def mobilenetv2_backbone_feature(
    inputs,
    *,
    width_multiplier,
    depth_multiplier,
    fix_channels,
    data_format,
    channels_axis,
    return_stages=False,
):
    """MobileNetV2 stem + 7 inverted-residual stages + head 1x1 conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        width_multiplier: Multiplier applied to per-stage channel counts.
        depth_multiplier: Multiplier applied to interior stage block repeats.
        fix_channels: If True, keep stem (32) and head (1280) channels fixed
            regardless of ``width_multiplier``.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-head-conv); otherwise return the post-head-conv
            tensor.

    Returns:
        Final 4D feature tensor after the head 1x1 conv (post BN + ReLU6), or a
        list of per-stage feature tensors when ``return_stages`` is True.
    """
    initial_dims = 32 if fix_channels else make_divisible(32 * width_multiplier)
    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), data_format=data_format, name="stem_padding"
    )(inputs)
    x = layers.Conv2D(
        initial_dims,
        3,
        2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation("relu6", name="relu1")(x)

    stages = []
    for layer_idx, layer_config in enumerate(_DEFAULT_BLOCKS):
        expansion_factor, output_channels, depths, initial_stride = layer_config
        scaled_output_channels = make_divisible(output_channels * width_multiplier)

        if layer_idx not in (0, len(_DEFAULT_BLOCKS) - 1):
            depths = int(keras.ops.ceil(depths * depth_multiplier))

        if return_stages and initial_stride == 2:
            stages.append(x)

        for block_idx in range(depths):
            current_stride = initial_stride if block_idx == 0 else 1
            x = inverted_residual_block(
                x,
                filters=scaled_output_channels,
                kernel_size=3,
                stride=current_stride,
                expansion_ratio=expansion_factor,
                channels_axis=channels_axis,
                data_format=data_format,
                block_id=layer_idx,
                sub_block_id=block_idx,
            )

    if return_stages:
        stages.append(x)
        return stages

    head_dims = (
        1280
        if fix_channels or width_multiplier <= 1.0
        else make_divisible(1280 * width_multiplier)
    )
    x = layers.Conv2D(
        head_dims,
        1,
        1,
        use_bias=False,
        data_format=data_format,
        name="head_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="head_batchnorm",
    )(x)
    x = layers.Activation("relu6", name="relu2")(x)

    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV2Model(BaseModel):
    """Instantiates the MobileNetV2 backbone.

    MobileNetV2 is designed for mobile inference and is built around
    inverted residual blocks with linear bottlenecks: each block expands
    the channel count via a 1x1 conv, applies a depthwise separable
    3x3 conv, projects back down with a 1x1 conv (no activation on the
    projection: the "linear bottleneck"), and adds a residual connection
    when the spatial / channel shapes match. The network is composed of
    a stem, seven such inverted-residual stages, and a 1x1 head conv,
    with channel widths controlled by a single width multiplier.

    Output is the last layer output before the classifier head: the
    post-head-conv 4D feature map of shape ``(B, H, W, C)``.
    :class:`MobileNetV2ImageClassify` composes this model and adds a
    GlobalAveragePooling2D + Dense head on top.

    References:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

    Args:
        width_multiplier: Float, multiplier applied to per-stage channel
            counts. Defaults to `1.0`.
        depth_multiplier: Float, multiplier applied to interior stage
            block repeats (ceiling-rounded). Defaults to `1.0`.
        fix_channels: Boolean, if True keep the stem (32) and head (1280)
            channel counts fixed regardless of ``width_multiplier``.
            Defaults to `False`.
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
            Defaults to `"MobileNetV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV2Config
    HUB_REPO_SIBLINGS = MOBILENETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MobileNetV2ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MobileNetV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv2_timm_to_keras import transfer_mobilenetv2_weights

        transfer_mobilenetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        fix_channels=False,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="MobileNetV2Model",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
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

        x = img_input
        x = mobilenetv2_backbone_feature(
            x,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            fix_channels=fix_channels,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.fix_channels = fix_channels
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "fix_channels": self.fix_channels,
                "image_size": self.image_size,
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
class MobileNetV2ImageClassify(BaseModel):
    """Instantiates the MobileNetV2 classifier.

    This classifier wraps a :class:`MobileNetV2Model` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`MobileNetV2Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

    Args:
        width_multiplier: Float, multiplier applied to per-stage channel
            counts. Defaults to `1.0`.
        depth_multiplier: Float, multiplier applied to interior stage
            block repeats (ceiling-rounded). Defaults to `1.0`.
        fix_channels: Boolean, if True keep the stem (32) and head (1280)
            channel counts fixed regardless of ``width_multiplier``.
            Defaults to `False`.
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
            `"MobileNetV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV2Config
    HUB_REPO_SIBLINGS = MOBILENETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv2_timm_to_keras import transfer_mobilenetv2_weights

        transfer_mobilenetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        fix_channels=False,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileNetV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = MobileNetV2Model(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            fix_channels=fix_channels,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            use_bias=True,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.fix_channels = fix_channels
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "fix_channels": self.fix_channels,
                "image_size": self.image_size,
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
