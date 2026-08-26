import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .inception_next_config import InceptionNextConfig

# The backbone (InceptionNextModel) and classifier (InceptionNextImageClassify) share
# the variant's repo, whose kf_config.json declares InceptionNextImageClassify.
INCEPTION_NEXT_HUB_SIBLINGS = frozenset(
    {"InceptionNextModel", "InceptionNextImageClassify"}
)


def inception_dwconv2d(
    x,
    square_kernel_size=3,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="token_mixer",
):
    """Inception-style token mixer: square + band depthwise convs over channel splits.

    The input channels are split into 4 groups: an identity branch (the bulk of
    the channels), a square ``k x k`` depthwise branch, and two band ``1 x k`` /
    ``k x 1`` depthwise branches. Outputs are concatenated back along the channel
    axis.

    Args:
        x: Input feature map.
        square_kernel_size: Kernel size of the square depthwise branch.
            Defaults to ``3``.
        band_kernel_size: Length of the band depthwise branches. Defaults to ``11``.
        branch_ratio: Fraction of input channels allocated to each non-identity
            branch. Defaults to ``0.125``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    input_channels = x.shape[channels_axis]
    branch_channels = int(input_channels * branch_ratio)
    split_sizes = [input_channels - 3 * branch_channels] + [branch_channels] * 3
    split_indices = [sum(split_sizes[: i + 1]) for i in range(len(split_sizes) - 1)]

    def calculate_padding(kernel_size):
        return (kernel_size - 1) // 2

    square_padding, band_padding = (
        calculate_padding(square_kernel_size),
        calculate_padding(band_kernel_size),
    )

    x_splits = keras.ops.split(x, split_indices, axis=channels_axis)
    x_id, *x_branches = x_splits

    conv_configs = [
        (square_kernel_size, square_padding, f"{name}_dwconv_hw"),
        ((1, band_kernel_size), (0, band_padding), f"{name}_dwconv_w"),
        ((band_kernel_size, 1), (band_padding, 0), f"{name}_dwconv_h"),
    ]

    x = [
        layers.DepthwiseConv2D(
            kernel, use_bias=True, data_format=data_format, name=lname
        )(layers.ZeroPadding2D(padding)(branch_input))
        for (kernel, padding, lname), branch_input in zip(conv_configs, x_branches)
    ]

    return layers.Concatenate(axis=channels_axis)([x_id, *x])


def inception_next_block(
    x,
    num_filter,
    mlp_ratio=4.0,
    dropout_rate=0.0,
    layer_scale_init=1e-6,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="blocks",
):
    """InceptionNeXt block: token mixer -> BN -> Conv MLP -> InceptionNextLayerScale -> residual.

    Args:
        x: Input feature map.
        num_filter: Channel count of the block (input == output).
        mlp_ratio: Hidden-dim expansion ratio for the MLP. Defaults to ``4.0``.
        dropout_rate: Dropout applied inside the MLP. Defaults to ``0.0``.
        layer_scale_init: Initial value for the InceptionNextLayerScale gamma.
            Defaults to ``1e-6``.
        band_kernel_size: Band length for the token mixer. Defaults to ``11``.
        branch_ratio: Channel fraction per token-mixer branch. Defaults to ``0.125``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    x_input = x

    x = inception_dwconv2d(
        x,
        band_kernel_size=band_kernel_size,
        branch_ratio=branch_ratio,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_token_mixer",
    )

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_batchnorm"
    )(x)

    x = layers.Conv2D(
        int(num_filter * mlp_ratio),
        1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(
        num_filter, 1, use_bias=True, data_format=data_format, name=f"{name}_conv2"
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = InceptionNextLayerScale(layer_scale_init, name=f"{name}_gamma")(x)
    x = layers.Add()([x, x_input])

    return x


def inception_next_backbone_feature(
    inputs,
    *,
    depths,
    num_filters,
    mlp_ratios,
    band_kernel_size,
    branch_ratio,
    data_format,
    channels_axis,
    return_stages=False,
):
    """InceptionNeXt stem + 4 stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        depths: Number of blocks per stage (length 4).
        num_filters: Output channel count per stage (length 4).
        mlp_ratios: MLP expansion ratio per stage (length 4).
        band_kernel_size: Band length for the token mixer (shared by all stages).
        branch_ratio: Channel fraction per token-mixer branch.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 4 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``num_filters[-1]`` channels at spatial
        resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 4 per-stage feature maps.
    """
    x = layers.Conv2D(
        num_filters[0],
        4,
        4,
        use_bias=True,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
    )(x)

    stages = []
    for i in range(len(depths)):
        strides = 2 if i > 0 else 1
        if strides > 1:
            x = layers.BatchNormalization(
                axis=channels_axis,
                momentum=0.9,
                epsilon=1e-5,
                name=f"stages_{i}_downsample_batchnorm",
            )(x)
            x = layers.Conv2D(
                num_filters[i],
                2,
                strides,
                use_bias=True,
                data_format=data_format,
                name=f"stages_{i}_downsample_conv",
            )(x)

        for j in range(depths[i]):
            x = inception_next_block(
                x,
                num_filter=num_filters[i],
                mlp_ratio=mlp_ratios[i],
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"stages_{i}_blocks_{j}",
            )

        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionNextModel(BaseModel):
    """Instantiates the InceptionNeXt backbone.

    InceptionNeXt decomposes the large-kernel depthwise convolution used
    by ConvNeXt into an Inception-style multi-branch token mixer: the
    channels are split into an identity branch plus a small square
    ``k x k`` depthwise branch and two band ``1 x k`` / ``k x 1``
    depthwise branches, then concatenated. These mixers are dropped into
    a ConvNeXt-shaped 4-stage architecture (stem + per-stage downsample
    + repeated blocks with a Conv-MLP and InceptionNextLayerScale).

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first), unpooled and head-free.
    :class:`InceptionNextImageClassify` composes this model and appends the
    MLP head.

    References:
    - [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            4 per-stage feature maps. Defaults to `False`.
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(3, 3, 9, 3)`.
        num_filters: Tuple of integers, output channel count per stage
            (length 4). Defaults to `(96, 192, 384, 768)`.
        mlp_ratios: Tuple of numbers, MLP expansion ratio per stage
            (length 4). Defaults to `(4, 4, 4, 3)`.
        band_kernel_size: Integer, band length for the token mixer
            (shared across stages). Defaults to `11`.
        branch_ratio: Float, fraction of input channels allocated to
            each non-identity token-mixer branch. Defaults to `0.125`.
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
        name: String, the name of the model.
            Defaults to `"InceptionNextModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionNextConfig
    HUB_REPO_SIBLINGS = INCEPTION_NEXT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with InceptionNextImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = InceptionNextImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inception_next_timm_to_keras import (
            transfer_inception_next_weights,
        )

        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="InceptionNextModel",
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
        x = inception_next_backbone_feature(
            x,
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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
class InceptionNextImageClassify(BaseModel):
    """Instantiates the InceptionNeXt classifier.

    This classifier wraps an :class:`InceptionNextModel` backbone and
    attaches a GlobalAveragePooling2D + Dense(`'head_fc'`) + GELU +
    LayerNorm + Dense head to produce ``num_classes`` class logits. All
    architectural parameters are forwarded to the underlying
    :class:`InceptionNextModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

    Args:
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(3, 3, 9, 3)`.
        num_filters: Tuple of integers, output channel count per stage
            (length 4). Defaults to `(96, 192, 384, 768)`.
        mlp_ratios: Tuple of numbers, MLP expansion ratio per stage
            (length 4). Defaults to `(4, 4, 4, 3)`.
        band_kernel_size: Integer, band length for the token mixer
            (shared across stages). Defaults to `11`.
        branch_ratio: Float, fraction of input channels allocated to
            each non-identity token-mixer branch. Defaults to `0.125`.
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
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`.
            Defaults to `"InceptionNextImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionNextConfig
    HUB_REPO_SIBLINGS = INCEPTION_NEXT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inception_next_timm_to_keras import (
            transfer_inception_next_weights,
        )

        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionNextImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionNextModel(
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.Dense(int(num_filters[-1] * 3.0), use_bias=True, name="head_fc")(x)
        x = layers.Activation("gelu")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="head_batchnorm")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
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
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionNextLayerScale(layers.Layer):
    """Learnable per-channel scale (x * gamma), gamma initialized to layer_scale_init."""

    def __init__(self, layer_scale_init, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale_init = layer_scale_init
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        self.scale_axis = (
            1 if self.data_format == "channels_first" and len(input_shape) == 4 else -1
        )
        self.gamma = self.add_weight(
            shape=(input_shape[self.scale_axis],),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )

    def call(self, x):
        if self.scale_axis == 1:
            return x * keras.ops.reshape(self.gamma, (1, -1, 1, 1))
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"layer_scale_init": self.layer_scale_init})
        return config
