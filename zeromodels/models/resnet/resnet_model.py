from typing import Optional

import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .resnet_config import ResNetConfig

# The ResNet backbone (ResNetModel) and classifier (ResNetImageClassify) share the
# variant's weights repo, whose kf_config.json declares the canonical
# ResNetImageClassify. Per-variant arch for the timm ``hf:`` path.
RESNET_HUB_SIBLINGS = frozenset({"ResNetModel", "ResNetImageClassify"})


def conv_block(
    x: layers.Layer,
    filters: int,
    kernel_size: int,
    channels_axis,
    data_format,
    strides: int = 1,
    use_relu: bool = True,
    groups: int = 1,
    group_width: Optional[int] = None,
    name: Optional[str] = None,
    bn_name: Optional[str] = None,
) -> layers.Layer:
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
        group_width: Width per group (used if groups > 1).
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    pad_h = pad_w = kernel_size // 2

    if strides > 1:
        x = layers.ZeroPadding2D(data_format=data_format, padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    if groups > 1:
        assert filters % groups == 0, (
            f"Filters ({filters}) must be divisible by groups ({groups})"
        )
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            groups=groups,
            kernel_initializer="he_normal",
            data_format=data_format,
            name=name,
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
            data_format=data_format,
            name=name,
        )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=bn_name
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def squeeze_excitation_block(
    x: layers.Layer, data_format, reduction_ratio: int = 16, name: Optional[str] = None
) -> layers.Layer:
    """
    Squeeze-and-Excitation block that properly handles both channels_first and channels_last formats.

    Args:
        x: Input tensor
        data_format: String, either 'channels_first' or 'channels_last'
        reduction_ratio: Integer, reduction ratio for the bottleneck
        name: String, optional name prefix for layers

    Returns:
        Tensor with same shape as input after applying SE attention
    """
    if data_format == "channels_first":
        channel_axis = 1
        filters = x.shape[channel_axis]
    else:
        channel_axis = -1
        filters = x.shape[channel_axis]

    se = layers.GlobalAveragePooling2D(data_format=data_format)(x)

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    reduced_filters = max(1, filters // reduction_ratio)
    se = layers.Reshape((filters,))(se)
    se = layers.Dense(
        reduced_filters,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense1" if name else None,
    )(se)
    se = layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense2" if name else None,
    )(se)

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply(name=f"{name}_scale" if name else None)([x, se])


def bottleneck_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
    strides: int = 1,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """Bottleneck ResNet block.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride for the main convolution layer.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4

    x = conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters,
        kernel_size=3,
        strides=strides,
        name=f"{block_name}_conv2",
        bn_name=f"{block_name}_batchnorm2",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters * expansion,
        kernel_size=1,
        use_relu=False,
        name=f"{block_name}_conv3",
        bn_name=f"{block_name}_batchnorm3",
        channels_axis=channels_axis,
        data_format=data_format,
    )

    if senet:
        x = squeeze_excitation_block(
            x, data_format=data_format, name=f"{block_name}_se"
        )

    if (
        downsample
        or strides != 1
        or x.shape[channels_axis] != residual.shape[channels_axis]
    ):
        residual = conv_block(
            residual,
            filters * expansion,
            kernel_size=1,
            strides=strides,
            use_relu=False,
            name=f"{block_name}_downsample_conv",
            bn_name=f"{block_name}_downsample_batchnorm",
            channels_axis=channels_axis,
            data_format=data_format,
        )

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x


def resnet_backbone_feature(
    inputs,
    block_fn,
    depths,
    filters,
    channels_axis,
    data_format,
    groups,
    senet,
    width_factor,
    return_stages=False,
):
    """Build the ResNet stem + stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        block_fn: Callable building one residual block (e.g. ``bottleneck_block``
            or ``resnext_block``).
        depths: Number of blocks per stage (length-4 list).
        filters: Base filter count per stage (length-4 list).
        channels_axis: Int axis for the channel dimension (-1 for channels-last,
            1 for channels-first).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        groups: Number of groups for grouped convolution (forwarded to
            ResNeXt blocks).
        senet: Whether to enable Squeeze-and-Excitation in each block.
        width_factor: Width scaling factor (forwarded to ResNeXt blocks).
        return_stages: If True, return a list of per-stage feature maps
            (4 tensors, one per ResNet stage). If False (default), return
            only the final stage map.

    Returns:
        Final stage feature tensor at stride 32 of the input, or a
        list of 4 per-stage feature maps when ``return_stages=True``.
    """
    x = conv_block(
        inputs,
        filters[0],
        kernel_size=7,
        strides=2,
        name="conv1",
        bn_name="batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        data_format=data_format, pool_size=3, strides=2, padding="valid"
    )(x)

    common_args = {
        "channels_axis": channels_axis,
        "data_format": data_format,
        "senet": senet,
    }
    if isinstance(block_fn, dict):
        if block_fn.get("module") == "zeromodels.models.resnext.resnext_model":
            common_args.update({"groups": groups, "width_factor": width_factor})
    elif hasattr(block_fn, "__module__") and "resnext" in block_fn.__module__:
        common_args.update({"groups": groups, "width_factor": width_factor})

    stages = []
    for i, depths in enumerate(depths):
        for j in range(depths):
            common_args["block_name"] = f"resnet_layer{i + 1}_{j}"
            if j == 0 and i > 0:
                x = block_fn(x, filters[i], strides=2, downsample=True, **common_args)
            else:
                x = block_fn(x, filters[i], **common_args)
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetModel(BaseModel):
    """Instantiates the Residual Network (ResNet) backbone.

    ResNet stacks 4 stages of residual bottleneck blocks at progressively
    halved spatial resolution and doubled channel width, with identity
    skip connections enabling very deep networks to train without
    degradation. The output tensor is the last layer output before the
    classifier head: the final stage's 4D feature map ``(B, H, W, C)``,
    unpooled and head-free. :class:`ResNetImageClassify` composes this model
    and applies a GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Args:
        block_fn: Callable, the residual block builder. Should accept
            ``(x, filters, strides=1, downsample=False, block_name=None)``
            and additional keyword arguments from the backbone (e.g.
            ``groups``, ``width_factor`` for ResNeXt).
            Defaults to `bottleneck_block`.
        depths: List of ints, number of residual blocks per stage.
            Defaults to `[2, 2, 2, 2]`.
        filters: List of ints, base filter counts per stage (the final
            output width is ``filters[i] * expansion``).
            Defaults to `[64, 128, 256, 512]`.
        groups: Integer, number of groups for grouped convolution
            (forwarded to ResNeXt blocks). Defaults to `32`.
        senet: Boolean, whether to apply Squeeze-and-Excitation inside
            each block. Defaults to `False`.
        width_factor: Integer, width scaling factor (forwarded to
            ResNeXt blocks). Defaults to `2`.
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
            per-stage feature maps (4 tensors, one per ResNet stage).
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"ResNetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNetConfig
    HUB_REPO_SIBLINGS = RESNET_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # The backbone shares the variant's weights repo with ResNetImageClassify
        # (which the kf_config declares); build it from the repo's kf_config, then
        # copy the matching backbone weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ResNetImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resnet_timm_to_keras import transfer_resnet_weights

        transfer_resnet_weights(keras_model, state_dict)

    def __init__(
        self,
        block_fn=bottleneck_block,
        depths=(2, 2, 2, 2),
        filters=(64, 128, 256, 512),
        groups=32,
        senet=False,
        width_factor=2,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="ResNetModel",
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
        x = resnet_backbone_feature(
            x,
            block_fn=block_fn,
            depths=depths,
            filters=filters,
            channels_axis=channels_axis,
            data_format=data_format,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.block_fn = block_fn
        self.depths = depths
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        if hasattr(self.block_fn, "__module__"):
            block_fn_config = {
                "class_name": "function",
                "config": self.block_fn.__name__,
                "module": self.block_fn.__module__,
                "registered_name": "function",
            }
        else:
            block_fn_config = {
                "class_name": "function",
                "config": "bottleneck_block",
                "module": "zeromodels.models.resnet.resnet_model",
                "registered_name": "function",
            }
        config.update(
            {
                "block_fn": block_fn_config,
                "depths": self.depths,
                "filters": self.filters,
                "groups": self.groups,
                "senet": self.senet,
                "width_factor": self.width_factor,
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
        if isinstance(config.get("block_fn"), dict):
            block_fn_name = config["block_fn"]["config"]
            module_path = config["block_fn"]["module"]
            if module_path == "zeromodels.models.resnet.resnet_model":
                if block_fn_name == "bottleneck_block":
                    config["block_fn"] = bottleneck_block
            elif module_path == "zeromodels.models.resnext.resnext_model":
                from zeromodels.models.resnext.resnext_model import resnext_block

                if block_fn_name == "resnext_block":
                    config["block_fn"] = resnext_block
            else:
                raise ValueError(
                    f"Unknown block function: {block_fn_name} from module {module_path}"
                )
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetImageClassify(BaseModel):
    """Instantiates the Residual Network (ResNet) classifier.

    This classifier wraps a :class:`ResNetModel` backbone and attaches a
    GlobalAveragePooling2D + Dense head to produce ``num_classes`` class
    logits. All architectural parameters are forwarded to the underlying
    :class:`ResNetModel`; only ``num_classes`` and ``classifier_activation``
    are head-specific.

    References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Args:
        block_fn: Callable, the residual block builder. Should accept
            ``(x, filters, strides=1, downsample=False, block_name=None)``
            and additional keyword arguments from the backbone (e.g.
            ``groups``, ``width_factor`` for ResNeXt).
            Defaults to `bottleneck_block`.
        depths: List of ints, number of residual blocks per stage.
            Defaults to `[2, 2, 2, 2]`.
        filters: List of ints, base filter counts per stage (the final
            output width is ``filters[i] * expansion``).
            Defaults to `[64, 128, 256, 512]`.
        groups: Integer, number of groups for grouped convolution
            (forwarded to ResNeXt blocks). Defaults to `32`.
        senet: Boolean, whether to apply Squeeze-and-Excitation inside
            each block. Defaults to `False`.
        width_factor: Integer, width scaling factor (forwarded to
            ResNeXt blocks). Defaults to `2`.
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
            named `f"{name}_backbone"`. Defaults to `"ResNetImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNetConfig
    HUB_REPO_SIBLINGS = RESNET_HUB_SIBLINGS
    HF_MODEL_TYPE = None  # timm-ported; no HF transformers passthrough.

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resnet_timm_to_keras import transfer_resnet_weights

        transfer_resnet_weights(keras_model, state_dict)

    def __init__(
        self,
        block_fn=bottleneck_block,
        depths=(2, 2, 2, 2),
        filters=(64, 128, 256, 512),
        groups=32,
        senet=False,
        width_factor=2,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNetImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ResNetModel(
            block_fn=block_fn,
            depths=depths,
            filters=filters,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
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
            num_classes,
            activation=classifier_activation,
            kernel_initializer="zeros",
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.block_fn = block_fn
        self.depths = depths
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()

        if hasattr(self.block_fn, "__module__"):
            block_fn_config = {
                "class_name": "function",
                "config": self.block_fn.__name__,
                "module": self.block_fn.__module__,
                "registered_name": "function",
            }
        else:
            block_fn_config = {
                "class_name": "function",
                "config": "bottleneck_block",
                "module": "zeromodels.models.resnet.resnet_model",
                "registered_name": "function",
            }

        config.update(
            {
                "block_fn": block_fn_config,
                "depths": self.depths,
                "filters": self.filters,
                "groups": self.groups,
                "senet": self.senet,
                "width_factor": self.width_factor,
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
        if isinstance(config["block_fn"], dict):
            block_fn_name = config["block_fn"]["config"]
            module_path = config["block_fn"]["module"]

            if module_path == "zeromodels.models.resnet.resnet_model":
                if block_fn_name == "bottleneck_block":
                    config["block_fn"] = bottleneck_block
            elif module_path == "zeromodels.models.resnext.resnext_model":
                from zeromodels.models.resnext.resnext_model import resnext_block

                if block_fn_name == "resnext_block":
                    config["block_fn"] = resnext_block
            else:
                raise ValueError(
                    f"Unknown block function: {block_fn_name} from module {module_path}"
                )

        return cls(**config)
