import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .densenet_config import DenseNetConfig

# The backbone (DenseNetModel) and classifier (DenseNetImageClassify) share the
# variant's repo, whose zm_config.json declares DenseNetImageClassify.
DENSENET_HUB_SIBLINGS = frozenset({"DenseNetModel", "DenseNetImageClassify"})


def conv_block(
    x,
    growth_rate,
    expansion_ratio,
    channels_axis,
    data_format,
    name,
):
    """Single conv layer inside a DenseNet dense block.

    Args:
        x: Input feature tensor.
        growth_rate: Output channel count of the 3x3 conv.
        expansion_ratio: Multiplier on ``growth_rate`` for the 1x1 bottleneck.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Tensor with ``growth_rate`` channels to be concatenated with prior layers.
    """
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        int(growth_rate * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        growth_rate,
        3,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_2",
    )(x)
    return x


def densenet_block(
    x,
    num_layers,
    growth_rate,
    channels_axis,
    data_format,
    name,
):
    """Dense block: stack of conv blocks with channel-wise concatenation.

    Args:
        x: Input feature tensor entering the block.
        num_layers: Number of internal conv blocks (a.k.a. dense layers).
        growth_rate: Per-layer channel growth.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Concatenated feature tensor whose channel count is
        ``x.channels + num_layers * growth_rate``.
    """
    output = x

    for i in range(num_layers):
        layer_output = conv_block(
            output,
            growth_rate,
            expansion_ratio=4.0,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_denselayer{i + 1}",
        )
        output = layers.Concatenate(axis=channels_axis)([output, layer_output])

    return output


def transition_block(x, reduction, channels_axis, data_format, name):
    """Reduce channels by ``reduction`` and 2x downsample spatially.

    Args:
        x: Input feature tensor from the preceding dense block.
        reduction: Channel reduction factor (e.g. ``0.5`` halves the channels).
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Down-projected tensor with halved spatial resolution.
    """
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_transition_batchnorm",
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        int(x.shape[channels_axis] * reduction),
        1,
        1,
        "same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_transition_conv2d",
    )(x)
    x = layers.AveragePooling2D(
        2, 2, data_format=data_format, name=f"{name}_transition_pool"
    )(x)
    return x


def densenet_backbone_feature(
    inputs,
    *,
    depths,
    growth_rate,
    initial_filter,
    channels_axis,
    data_format,
    return_stages=False,
):
    """Stem + N dense blocks + final BN/ReLU, returning the final feature map.

    Args:
        inputs: Input image tensor (post-normalization).
        depths: Tuple of layer counts per dense block (e.g. ``(6, 12, 24, 16)``).
        growth_rate: Per-layer channel growth inside each dense block.
        initial_filter: Channel count for the 7x7 stem convolution.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps (one
            per dense block; the final stage is the post BN+ReLU output). If
            False (default), return only the final feature map.

    Returns:
        Final feature map ``(B, H, W, C)`` with BN+ReLU applied, or a list of
        per-stage feature maps when ``return_stages=True``.
    """
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), data_format=data_format)(inputs)
    x = layers.Conv2D(
        initial_filter,
        7,
        2,
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_norm"
    )(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(1, data_format=data_format)(x)
    x = layers.MaxPooling2D(3, 2, data_format=data_format, name="stem_pool")(x)

    stages = []
    for i, num_layers in enumerate(depths):
        x = densenet_block(
            x,
            num_layers,
            growth_rate,
            channels_axis,
            data_format,
            name=f"dense_block{i + 1}",
        )

        if i != len(depths) - 1:
            x = transition_block(
                x,
                0.5,
                channels_axis,
                data_format,
                name=f"transition_block{i + 1}",
            )
            stages.append(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="final_batchnorm",
    )(x)
    x = layers.ReLU(name="final_relu")(x)
    stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class DenseNetModel(BaseModel):
    """Instantiates the DenseNet backbone.

    DenseNet introduces dense connectivity within each block: every
    layer receives feature maps from all preceding layers in the block
    via channel-wise concatenation, with transition layers (1x1 conv +
    average pool) between dense blocks to control channel growth and
    downsample. Output is the last layer output before the classifier
    head: the final feature map ``(B, H, W, C)`` after a final
    BatchNorm + ReLU. :class:`DenseNetImageClassify` composes this model and
    attaches a GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one per dense block; the final stage
            is the post BN+ReLU output). Defaults to `False`.
        depths: Tuple of integers, number of conv layers in each
            dense block. Defaults to `(6, 12, 24, 16)`.
        growth_rate: Integer, per-layer channel growth inside each dense
            block. Defaults to `32`.
        initial_filter: Integer, channel count for the 7x7 stem
            convolution. Defaults to `64`.
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
        name: String, the name of the model.
            Defaults to `"DenseNetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = DenseNetConfig
    HUB_REPO_SIBLINGS = DENSENET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with DenseNetImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = DenseNetImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_densenet_timm_to_keras import transfer_densenet_weights

        transfer_densenet_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(6, 12, 24, 16),
        growth_rate=32,
        initial_filter=64,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="DenseNetModel",
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
        x = densenet_backbone_feature(
            x,
            depths=depths,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
            channels_axis=channels_axis,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = depths
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "growth_rate": self.growth_rate,
                "initial_filter": self.initial_filter,
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
class DenseNetImageClassify(BaseModel):
    """Instantiates the DenseNet classifier.

    This classifier wraps a :class:`DenseNetModel` backbone and attaches
    a GlobalAveragePooling2D + Dense head to produce ``num_classes``
    class logits. All architectural parameters are forwarded to the
    underlying :class:`DenseNetModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

    Args:
        depths: Tuple of integers, number of conv layers in each
            dense block. Defaults to `(6, 12, 24, 16)`.
        growth_rate: Integer, per-layer channel growth inside each dense
            block. Defaults to `32`.
        initial_filter: Integer, channel count for the 7x7 stem
            convolution. Defaults to `64`.
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
            named `f"{name}_backbone"`. Defaults to `"DenseNetImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = DenseNetConfig
    HUB_REPO_SIBLINGS = DENSENET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_densenet_timm_to_keras import transfer_densenet_weights

        transfer_densenet_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(6, 12, 24, 16),
        growth_rate=32,
        initial_filter=64,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="DenseNetImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = DenseNetModel(
            depths=depths,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = depths
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "growth_rate": self.growth_rate,
                "initial_filter": self.initial_filter,
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
