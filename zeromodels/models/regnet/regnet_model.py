import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .regnet_config import RegNetConfig

REGNET_HUB_SIBLINGS = frozenset({"RegNetModel", "RegNetImageClassify"})


def regnet_conv_layer(
    x,
    filters,
    channels_axis,
    data_format,
    kernel_size=3,
    strides=1,
    groups=1,
    use_activation=True,
    name=None,
):
    """RegNetConvLayer: bias-free conv, batch norm, optional ReLU.

    Args:
        x: Input tensor.
        filters: Number of output channels.
        channels_axis: Channel axis (-1 for channels_last, 1 for channels_first).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        kernel_size: Convolution kernel size.
        strides: Convolution stride.
        groups: Number of groups for the grouped convolution.
        use_activation: Whether to apply a ReLU after the batch norm.
        name: Prefix; the conv is ``{name}_convolution`` and the norm is
            ``{name}_normalization`` (matching the HF module path).

    Returns:
        Output tensor for the block.
    """
    if strides > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad, pad), data_format=data_format)(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_convolution",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_normalization"
    )(x)
    if use_activation:
        x = layers.ReLU()(x)
    return x


def regnet_se_layer(x, reduced_channels, channels_axis, data_format, name=None):
    """Squeeze-and-Excitation (RegNetSELayer): pooled 1x1 conv bottleneck gate.

    Mirrors HF's ``pooler -> attention[0] (conv) -> ReLU -> attention[2] (conv)
    -> Sigmoid -> multiply`` using 1x1 convolutions (both biased) so the weights
    map directly.
    """
    filters = x.shape[channels_axis]
    se = layers.GlobalAveragePooling2D(data_format=data_format, keepdims=True)(x)
    se = layers.Conv2D(
        reduced_channels,
        1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_attention_0",
    )(se)
    se = layers.ReLU()(se)
    se = layers.Conv2D(
        filters,
        1,
        use_bias=True,
        activation="sigmoid",
        data_format=data_format,
        name=f"{name}_attention_2",
    )(se)
    return layers.Multiply()([x, se])


def regnet_block(
    x,
    in_channels,
    out_channels,
    groups_width,
    layer_type,
    channels_axis,
    data_format,
    strides=1,
    name=None,
):
    """One RegNet residual block (``"x"`` bottleneck or ``"y"`` = + SE).

    ``1x1 -> 3x3 grouped (stride) -> [SE] -> 1x1`` with a strided 1x1 shortcut when
    the shape changes, then a residual add and ReLU.
    """
    groups = max(1, out_channels // groups_width)
    should_shortcut = in_channels != out_channels or strides != 1

    residual = x
    if should_shortcut:
        residual = regnet_conv_layer(
            x,
            out_channels,
            kernel_size=1,
            strides=strides,
            use_activation=False,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_shortcut",
        )

    h = regnet_conv_layer(
        x,
        out_channels,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{name}_layer_0",
    )
    h = regnet_conv_layer(
        h,
        out_channels,
        kernel_size=3,
        strides=strides,
        groups=groups,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{name}_layer_1",
    )
    if layer_type == "y":
        h = regnet_se_layer(
            h,
            reduced_channels=int(round(in_channels / 4)),
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_layer_2",
        )
        last_idx = 3
    else:
        last_idx = 2
    h = regnet_conv_layer(
        h,
        out_channels,
        kernel_size=1,
        use_activation=False,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{name}_layer_{last_idx}",
    )

    h = layers.Add()([h, residual])
    h = layers.ReLU()(h)
    return h


def regnet_backbone_feature(
    inputs,
    embedding_size,
    hidden_sizes,
    depths,
    groups_width,
    layer_type,
    downsample_in_first_stage,
    channels_axis,
    data_format,
    return_stages=False,
):
    """Build the RegNet stem + four stages.

    Returns the final stage feature map, or a list of per-stage feature maps
    (one per stage) when ``return_stages=True``.
    """
    x = regnet_conv_layer(
        inputs,
        embedding_size,
        kernel_size=3,
        strides=2,
        channels_axis=channels_axis,
        data_format=data_format,
        name="regnet_embedder_embedder",
    )

    in_channels = embedding_size
    stages = []
    for i, (out_channels, depth) in enumerate(zip(hidden_sizes, depths)):
        stage_stride = 2 if (i > 0 or downsample_in_first_stage) else 1
        for j in range(depth):
            x = regnet_block(
                x,
                in_channels if j == 0 else out_channels,
                out_channels,
                groups_width=groups_width,
                layer_type=layer_type,
                channels_axis=channels_axis,
                data_format=data_format,
                strides=stage_stride if j == 0 else 1,
                name=f"regnet_encoder_stages_{i}_layers_{j}",
            )
        in_channels = out_channels
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class RegNetModel(BaseModel):
    """Instantiates the RegNet backbone.

    RegNet is a quantized-linear ConvNet: a 3x3 stride-2 stem feeds four stages
    of ``1x1 -> 3x3 grouped -> [SE] -> 1x1`` residual blocks whose widths and
    depths follow a simple parametric rule. The output tensor is the last layer
    output before the classifier head: the final stage's 4D feature map
    ``(B, H, W, C)``, unpooled and head-free. :class:`RegNetImageClassify`
    composes this model and applies a GlobalAveragePooling2D + Dense head to
    produce logits.

    References:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

    Args:
        embedding_size: Integer, output width of the stem. Defaults to `32`.
        hidden_sizes: Tuple of ints, output width per stage.
            Defaults to `(128, 192, 512, 1088)`.
        depths: Tuple of ints, number of blocks per stage.
            Defaults to `(2, 6, 12, 2)`.
        groups_width: Integer, channels per group of the 3x3 grouped conv (the
            group count of a block is ``out_channels // groups_width``).
            Defaults to `64`.
        layer_type: String, `"y"` (with Squeeze-and-Excitation) or `"x"`.
            Defaults to `"y"`.
        downsample_in_first_stage: Boolean, whether the first stage downsamples
            (stride 2). Defaults to `True`.
        image_size: Input image spec (int, ``(H, W)``, or a 3-tuple ordered for
            the active ``keras.config.image_data_format()``). Defaults to `224`.
        input_tensor: Optional Keras tensor as input. Defaults to `None`.
        as_backbone: Boolean, when True returns a list of per-stage feature maps.
            Defaults to `False`.
        name: String model name. Defaults to `"RegNetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = RegNetConfig
    HUB_REPO_SIBLINGS = REGNET_HUB_SIBLINGS
    HF_MODEL_TYPE = "regnet"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = RegNetImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embedding_size": hf_config["embedding_size"],
            "hidden_sizes": tuple(hf_config["hidden_sizes"]),
            "depths": tuple(hf_config["depths"]),
            "groups_width": hf_config["groups_width"],
            "layer_type": hf_config["layer_type"],
            "downsample_in_first_stage": hf_config.get(
                "downsample_in_first_stage", True
            ),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_regnet_hf_to_keras import transfer_regnet_weights

        transfer_regnet_weights(keras_model, state_dict)

    def __init__(
        self,
        embedding_size=32,
        hidden_sizes=(128, 192, 512, 1088),
        depths=(2, 6, 12, 2),
        groups_width=64,
        layer_type="y",
        downsample_in_first_stage=True,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="RegNetModel",
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

        x = img_input
        x = regnet_backbone_feature(
            x,
            embedding_size=embedding_size,
            hidden_sizes=hidden_sizes,
            depths=depths,
            groups_width=groups_width,
            layer_type=layer_type,
            downsample_in_first_stage=downsample_in_first_stage,
            channels_axis=channels_axis,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.groups_width = groups_width
        self.layer_type = layer_type
        self.downsample_in_first_stage = downsample_in_first_stage
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "hidden_sizes": self.hidden_sizes,
                "depths": self.depths,
                "groups_width": self.groups_width,
                "layer_type": self.layer_type,
                "downsample_in_first_stage": self.downsample_in_first_stage,
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
class RegNetImageClassify(BaseModel):
    """Instantiates the RegNet classifier.

    Wraps a :class:`RegNetModel` backbone and attaches a GlobalAveragePooling2D +
    Dense head to produce ``num_classes`` class logits. All architectural
    parameters forward to the underlying :class:`RegNetModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

    Args:
        embedding_size: Integer, output width of the stem. Defaults to `32`.
        hidden_sizes: Tuple of ints, output width per stage.
            Defaults to `(128, 192, 512, 1088)`.
        depths: Tuple of ints, number of blocks per stage.
            Defaults to `(2, 6, 12, 2)`.
        groups_width: Integer, channels per group of the 3x3 grouped conv.
            Defaults to `64`.
        layer_type: String, `"y"` (with SE) or `"x"`. Defaults to `"y"`.
        downsample_in_first_stage: Boolean, whether the first stage downsamples.
            Defaults to `True`.
        image_size: Input image spec. Defaults to `224`.
        input_tensor: Optional Keras tensor as input. Defaults to `None`.
        num_classes: Integer, number of output classes. Defaults to `1000`.
        classifier_activation: String/callable for the final Dense. Use
            `"linear"` for logits or `"softmax"` for probabilities.
            Defaults to `"linear"`.
        name: String model name. The internal backbone is named
            `f"{name}_backbone"`. Defaults to `"RegNetImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = RegNetConfig
    HUB_REPO_SIBLINGS = REGNET_HUB_SIBLINGS
    HF_MODEL_TYPE = "regnet"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embedding_size": hf_config["embedding_size"],
            "hidden_sizes": tuple(hf_config["hidden_sizes"]),
            "depths": tuple(hf_config["depths"]),
            "groups_width": hf_config["groups_width"],
            "layer_type": hf_config["layer_type"],
            "downsample_in_first_stage": hf_config.get(
                "downsample_in_first_stage", True
            ),
            "num_classes": hf_config.get("num_labels", 1000),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_regnet_hf_to_keras import transfer_regnet_weights

        transfer_regnet_weights(keras_model, state_dict)

    def __init__(
        self,
        embedding_size=32,
        hidden_sizes=(128, 192, 512, 1088),
        depths=(2, 6, 12, 2),
        groups_width=64,
        layer_type="y",
        downsample_in_first_stage=True,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="RegNetImageClassify",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()

        backbone = RegNetModel(
            embedding_size=embedding_size,
            hidden_sizes=hidden_sizes,
            depths=depths,
            groups_width=groups_width,
            layer_type=layer_type,
            downsample_in_first_stage=downsample_in_first_stage,
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
            kernel_initializer="zeros",
            name="classifier_1",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.groups_width = groups_width
        self.layer_type = layer_type
        self.downsample_in_first_stage = downsample_in_first_stage
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_size": self.embedding_size,
                "hidden_sizes": self.hidden_sizes,
                "depths": self.depths,
                "groups_width": self.groups_width,
                "layer_type": self.layer_type,
                "downsample_in_first_stage": self.downsample_in_first_stage,
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
