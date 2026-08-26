import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .res2net_config import Res2NetConfig

# The backbone (Res2NetModel) and classifier (Res2NetImageClassify) share the
# variant's weights repo, whose kf_config.json declares Res2NetImageClassify.
RES2NET_HUB_SIBLINGS = frozenset({"Res2NetModel", "Res2NetImageClassify"})


def conv_block(
    x,
    filters,
    kernel_size,
    channels_axis,
    data_format,
    strides=1,
    use_relu=True,
    groups=1,
    name=None,
    bn_name=None,
):
    """Applies a convolution block with optional grouped convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride of the convolution.
        use_relu: Whether to apply ReLU activation after convolution.
        groups: Number of groups for grouped convolution.
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    if strides > 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w), data_format=data_format)(x)
        padding_mode = "valid"
    else:
        padding_mode = "same"

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding_mode,
        use_bias=False,
        groups=groups,
        data_format=data_format,
        name=name,
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        name=bn_name,
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def bottle2neck_block(
    x,
    filters,
    block_name,
    data_format,
    stride=1,
    downsample=False,
    cardinality=1,
    base_width=26,
    scale=4,
):
    """Res2Net/ResNeSt Bottle2neck block with multi-scale features.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        block_name: Name prefix for layers in the block.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        stride: Stride for the 3x3 convolution layers.
        downsample: Whether to downsample the input.
        cardinality: Number of groups for grouped convolutions.
        base_width: Base width of the block, controls channel scaling.
        scale: Scale factor that determines number of feature scales.

    Returns:
        Output tensor for the block.

    Notes:
        The block implements multi-scale feature processing by:
        1. Initial 1x1 conv to expand channels
        2. Split features into multiple scales
        3. Hierarchical residual-like connections between scales
        4. Optional average pooling for the last scale
        5. Concatenate all scales and reduce with 1x1 conv

        The expansion factor is fixed at 4, similar to standard ResNet bottleneck blocks.
    """
    channels_axis = -1 if data_format == "channels_last" else 1
    expansion = 4
    is_first = stride > 1 or downsample
    width = int(filters * (base_width / 64.0)) * cardinality
    outplanes = filters * expansion

    identity = x

    x = conv_block(
        x,
        width * scale,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{block_name}_conv_1",
        bn_name=f"{block_name}_batchnorm_1",
    )

    x_splits = ops.split(x, scale, axis=channels_axis)
    spouts = []

    for i in range(scale - 1):
        if i == 0 or is_first:
            sp = x_splits[i]
        else:
            sp = layers.Add()([spouts[-1], x_splits[i]])

        sp = conv_block(
            sp,
            width,
            kernel_size=3,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride if is_first else 1,
            groups=cardinality,
            name=f"{block_name}_conv_s_{i}",
            bn_name=f"{block_name}_batchnorm_s_{i}",
        )
        spouts.append(sp)

    if scale > 1:
        if is_first:
            last = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)), data_format=data_format
            )(x_splits[-1])
            last = layers.AveragePooling2D(
                pool_size=3,
                strides=stride,
                padding="valid",
                data_format=data_format,
            )(last)
        else:
            last = x_splits[-1]
        spouts.append(last)

    out = layers.Concatenate(axis=channels_axis)(spouts)

    out = conv_block(
        out,
        outplanes,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        use_relu=False,
        name=f"{block_name}_conv_3",
        bn_name=f"{block_name}_batchnorm_3",
    )

    if downsample:
        identity = conv_block(
            identity,
            outplanes,
            kernel_size=1,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride,
            use_relu=False,
            name=f"{block_name}_downsample_0",
            bn_name=f"{block_name}_downsample_1",
        )

    out = layers.Add()([identity, out])
    out = layers.ReLU()(out)

    return out


def res2net_backbone_feature(
    inputs,
    depth,
    base_width,
    scale,
    cardinality,
    channels_axis,
    data_format,
    return_stages=False,
):
    """Res2Net stem + stages, returning the final stage feature map.

    Args:
        inputs: Input image tensor.
        depth: Number of Bottle2neck blocks per stage (length-4 list).
        base_width: Base channel width per scale in the Bottle2neck block.
        scale: Number of feature scales in the Bottle2neck block.
        cardinality: Number of groups for grouped convolution.
        channels_axis: Int axis for the channel dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps
            (one tensor per Res2Net stage, typically 4). If False
            (default), return only the final stage map.

    Returns:
        Final stage feature tensor, or a list of per-stage feature maps
        when ``return_stages=True``.
    """
    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(inputs)
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name="bn1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="valid", data_format=data_format
    )(x)

    filters = [64, 128, 256, 512]
    stages = []
    for i, (blocks, filter_size) in enumerate(zip(depth, filters)):
        stride = 1 if i == 0 else 2
        x = bottle2neck_block(
            x,
            filter_size,
            f"layer{i + 1}_0",
            stride=stride,
            downsample=True,
            base_width=base_width,
            cardinality=cardinality,
            scale=scale,
            data_format=data_format,
        )
        for j in range(1, blocks):
            x = bottle2neck_block(
                x,
                filter_size,
                f"layer{i + 1}_{j}",
                base_width=base_width,
                cardinality=cardinality,
                scale=scale,
                data_format=data_format,
            )
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class Res2NetModel(BaseModel):
    """Instantiates the Res2Net (Multi-scale Residual Network) backbone.

    Res2Net replaces the standard 3x3 convolution inside the bottleneck
    with a hierarchical multi-scale residual structure: the intermediate
    feature map is split into ``scale`` groups along the channel axis,
    each group is processed by its own 3x3 conv, and the outputs are
    fused via residual connections to subsequent groups, increasing the
    effective receptive-field range at fine granularity without adding
    depth. The output tensor is the last layer output before the
    classifier head: the final-stage feature map ``(B, H, W, C)``.
    :class:`Res2NetImageClassify` composes this model and applies a
    GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

    Args:
        depth: Tuple of ints, number of Bottle2neck blocks per stage.
            Defaults to `(3, 4, 6, 3)`.
        base_width: Integer, base channel width per scale inside each
            Bottle2neck block. Defaults to `26`.
        scale: Integer, number of feature scales per Bottle2neck block.
            Defaults to `4`.
        cardinality: Integer, number of groups for grouped convolution
            inside each scale. Defaults to `1`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one tensor per Res2Net stage).
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"Res2NetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = Res2NetConfig
    HUB_REPO_SIBLINGS = RES2NET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with Res2NetImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = Res2NetImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_res2net_timm_to_keras import transfer_res2net_weights

        transfer_res2net_weights(keras_model, state_dict)

    def __init__(
        self,
        depth=(3, 4, 6, 3),
        base_width=26,
        scale=4,
        cardinality=1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="Res2NetModel",
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
        x = res2net_backbone_feature(
            x,
            depth=depth,
            base_width=base_width,
            scale=scale,
            cardinality=cardinality,
            channels_axis=channels_axis,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Res2NetImageClassify(BaseModel):
    """Instantiates the Res2Net (Multi-scale Residual Network) classifier.

    This classifier wraps a :class:`Res2NetModel` backbone and attaches
    a GlobalAveragePooling2D + Dense head to produce ``num_classes``
    class logits. All architectural parameters are forwarded to the
    underlying :class:`Res2NetModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

    Args:
        depth: Tuple of ints, number of Bottle2neck blocks per stage.
            Defaults to `(3, 4, 6, 3)`.
        base_width: Integer, base channel width per scale inside each
            Bottle2neck block. Defaults to `26`.
        scale: Integer, number of feature scales per Bottle2neck block.
            Defaults to `4`.
        cardinality: Integer, number of groups for grouped convolution
            inside each scale. Defaults to `1`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
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
            named `f"{name}_backbone"`. Defaults to `"Res2NetImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = Res2NetConfig
    HUB_REPO_SIBLINGS = RES2NET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_res2net_timm_to_keras import transfer_res2net_weights

        transfer_res2net_weights(keras_model, state_dict)

    def __init__(
        self,
        depth=(3, 4, 6, 3),
        base_width=26,
        scale=4,
        cardinality=1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="Res2NetImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = Res2NetModel(
            depth=depth,
            base_width=base_width,
            scale=scale,
            cardinality=cardinality,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
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
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
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
