import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.resnetv2.resnetv2_layers import (
    ResNetV2StdConv2D,
    ResNetV2StochasticDepth,
)
from zeromodels.utils import standardize_input_shape

from .resnetv2_config import ResNetV2Config

# The backbone (ResNetV2Model) and classifier (ResNetV2ImageClassify) share the
# variant's weights repo, whose zm_config.json declares ResNetV2ImageClassify.
RESNETV2_HUB_SIBLINGS = frozenset({"ResNetV2Model", "ResNetV2ImageClassify"})


def make_divisible(v, divisor=8):
    """Round ``v`` to the nearest multiple of ``divisor``, never below 90% of ``v``.

    Args:
        v: Numeric value to round (typically a channel count).
        divisor: Multiple to round to. Defaults to ``8``.

    Returns:
        Int channel count that is a multiple of ``divisor`` and at least
        90% of the original ``v``.
    """
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_block(
    x,
    filters,
    kernel_size,
    data_format,
    strides=1,
    padding="same",
    use_bias=False,
    name=None,
):
    """Weight-standardized Conv2D with explicit zero-pad on strided convs.

    Args:
        x: Input Keras tensor.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: Stride of the convolution.
        padding: Padding mode passed to :class:`ResNetV2StdConv2D` when no explicit
            zero-padding is applied.
        use_bias: Whether the convolution uses a bias term.
        name: Optional name for the convolution layer.

    Returns:
        Output tensor after weight-standardized convolution.
    """
    if strides > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad, pad))(x)
        padding = "valid"

    x = ResNetV2StdConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        data_format=data_format,
        name=name,
    )(x)
    return x


def preact_bottleneck(
    x,
    filters,
    data_format,
    channels_axis,
    strides=1,
    downsample=False,
    drop_path_rate=0.0,
    block_prefix=None,
    bottleneck_ratio=0.25,
):
    """Pre-activation bottleneck used by BiT / ResNetV2.

    Args:
        x: Input Keras tensor.
        filters: Number of output filters for the block.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Int axis for the channel dimension.
        strides: Stride for the 3x3 convolution.
        downsample: Whether to project the shortcut to the new spatial /
            channel shape.
        drop_path_rate: Stochastic-depth drop probability. ``0`` disables.
        block_prefix: Name prefix for layers in the block.
        bottleneck_ratio: Reduction factor for the bottleneck channels.

    Returns:
        Output tensor of the residual block.
    """
    shortcut = x
    mid_channels = make_divisible(filters * bottleneck_ratio)

    preact = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_1"
    )(x)
    preact = layers.Activation("relu", name=f"{block_prefix}_relu_1")(preact)

    if downsample:
        shortcut = conv_block(
            preact,
            filters=filters,
            kernel_size=1,
            data_format=data_format,
            strides=strides,
            use_bias=False,
            name=f"{block_prefix}_downsample_conv",
        )

    x = conv_block(
        preact,
        filters=mid_channels,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_1",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_2"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_2")(x)

    x = conv_block(
        x,
        filters=mid_channels,
        kernel_size=3,
        data_format=data_format,
        strides=strides,
        use_bias=False,
        name=f"{block_prefix}_conv_2",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_3"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_3")(x)

    x = conv_block(
        x,
        filters=filters,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_3",
    )

    if drop_path_rate > 0:
        x = ResNetV2StochasticDepth(drop_path_rate)(x)

    x = layers.Add(name=f"{block_prefix}_add")([shortcut, x])
    return x


def resnetv2_backbone_feature(
    inputs,
    depths,
    filters,
    width_factor,
    stem_width,
    drop_path_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """ResNetV2 stem + stages, returning the final stage feature map.

    Args:
        inputs: Input image tensor.
        depths: Number of pre-activation bottlenecks per stage.
        filters: Base filter count per stage.
        width_factor: Multiplier applied to channel counts (BiT widening).
        stem_width: Base width of the stem convolution before width scaling.
        drop_path_rate: Maximum stochastic-depth drop probability; linearly
            interpolated across blocks.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Int axis for the channel dimension.
        return_stages: If True, return a list of per-stage feature maps
            (one tensor per ResNetV2 stage). If False (default), return
            only the final stage map.

    Returns:
        Final stage feature tensor (pre final GroupNorm), or a list of
        per-stage feature maps when ``return_stages=True``.
    """
    x = conv_block(
        inputs,
        filters=make_divisible(stem_width * width_factor),
        kernel_size=7,
        data_format=data_format,
        strides=2,
        use_bias=False,
        name="stem_conv",
    )
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        data_format=data_format,
        padding="valid",
        name="stem_maxpool",
    )(x)

    n = sum(depths)
    dpr = [drop_path_rate * i / max(n - 1, 1) for i in range(n)]
    block_idx = 0
    stages = []
    for stage_idx, depths in enumerate(depths):
        nb_channels_stage = make_divisible(filters[stage_idx] * width_factor)
        for block_idx_in_stage in range(depths):
            block_prefix = f"stages_{stage_idx}_blocks_{block_idx_in_stage}"
            x = preact_bottleneck(
                x,
                filters=nb_channels_stage,
                data_format=data_format,
                channels_axis=channels_axis,
                strides=2 if (stage_idx > 0 and block_idx_in_stage == 0) else 1,
                downsample=block_idx_in_stage == 0,
                drop_path_rate=dpr[block_idx],
                block_prefix=block_prefix,
            )
            block_idx += 1
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetV2Model(BaseModel):
    """Instantiates the ResNetV2 (Pre-activation ResNet / BiT) backbone.

    ResNetV2 is the pre-activation variant of ResNet: it moves
    GroupNorm + ReLU to the start of each residual branch, which
    improves gradient flow and stabilizes training of very deep
    networks. The output tensor is the last layer output before the
    classifier head: the raw final-stage feature map ``(B, H, W, C)``
    (pre final GroupNorm). :class:`ResNetV2ImageClassify` composes this model
    and applies the final GroupNorm + ReLU + GlobalAveragePooling2D +
    optional Dropout + Dense head to produce logits.

    References:
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    - [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

    Args:
        depths: Tuple of ints, number of pre-activation
            bottlenecks per stage. Defaults to `(3, 4, 6, 3)`.
        filters: Tuple of ints, base filter counts per stage before
            ``width_factor`` scaling. Defaults to `(256, 512, 1024, 2048)`.
        width_factor: Integer, multiplier applied to channel counts
            (BiT widening). Defaults to `1`.
        stem_width: Integer, base width of the stem convolution before
            ``width_factor`` scaling. Defaults to `64`.
        drop_path_rate: Float, maximum stochastic-depth drop
            probability; linearly interpolated across all blocks.
            Defaults to `0.0`.
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
            per-stage feature maps. Defaults to `False`.
        name: String, the name of the model. Defaults to `"ResNetV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNetV2Config
    HUB_REPO_SIBLINGS = RESNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ResNetV2ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ResNetV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resnetv2_timm_to_keras import transfer_resnetv2_weights

        transfer_resnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 6, 3),
        filters=(256, 512, 1024, 2048),
        width_factor=1,
        stem_width=64,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="ResNetV2Model",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        for k in ("num_classes", "classifier_activation", "drop_rate", "timm_id"):
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
        x = resnetv2_backbone_feature(
            x,
            depths=depths,
            filters=filters,
            width_factor=width_factor,
            stem_width=stem_width,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = depths
        self.filters = filters
        self.width_factor = width_factor
        self.stem_width = stem_width
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "filters": self.filters,
                "width_factor": self.width_factor,
                "stem_width": self.stem_width,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "as_backbone": self.as_backbone,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetV2ImageClassify(BaseModel):
    """Instantiates the ResNetV2 (Pre-activation ResNet / BiT) classifier.

    This classifier wraps a :class:`ResNetV2Model` backbone and attaches
    a GroupNorm -> ReLU -> GlobalAveragePooling2D -> optional Dropout ->
    Dense head to produce ``num_classes`` class logits. All architectural
    parameters are forwarded to the underlying :class:`ResNetV2Model`;
    only ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    - [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

    Args:
        depths: Tuple of ints, number of pre-activation
            bottlenecks per stage. Defaults to `(3, 4, 6, 3)`.
        filters: Tuple of ints, base filter counts per stage before
            ``width_factor`` scaling. Defaults to `(256, 512, 1024, 2048)`.
        width_factor: Integer, multiplier applied to channel counts
            (BiT widening). Defaults to `1`.
        stem_width: Integer, base width of the stem convolution before
            ``width_factor`` scaling. Defaults to `64`.
        drop_rate: Float, dropout rate applied to the pooled features
            before the final Dense layer. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop
            probability; linearly interpolated across all blocks.
            Defaults to `0.0`.
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
            named `f"{name}_backbone"`. Defaults to `"ResNetV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNetV2Config
    HUB_REPO_SIBLINGS = RESNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resnetv2_timm_to_keras import transfer_resnetv2_weights

        transfer_resnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 6, 3),
        filters=(256, 512, 1024, 2048),
        width_factor=1,
        stem_width=64,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNetV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        backbone = ResNetV2Model(
            depths=depths,
            filters=filters,
            width_factor=width_factor,
            stem_width=stem_width,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GroupNormalization(axis=channels_axis, name="groupnorm")(
            backbone.output
        )
        x = layers.Activation("relu", name="relu")(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, name="dropout")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = depths
        self.filters = filters
        self.width_factor = width_factor
        self.stem_width = stem_width
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "filters": self.filters,
                "width_factor": self.width_factor,
                "stem_width": self.stem_width,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
