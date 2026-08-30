from typing import Optional

import keras
from keras import layers

from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.resnet.resnet_model import (
    ResNetImageClassify,
    ResNetModel,
    conv_block,
    squeeze_excitation_block,
)

from .resnext_config import ResNeXtConfig

# The backbone (ResNeXtModel) and classifier (ResNeXtImageClassify) share the
# variant's weights repo, whose zm_config.json declares ResNeXtImageClassify.
RESNEXT_HUB_SIBLINGS = frozenset({"ResNeXtModel", "ResNeXtImageClassify"})


def resnext_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
    strides: int = 1,
    groups: int = 32,
    width_factor: int = 2,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """ResNeXt block with group convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the block.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride for the main convolution layer.
        groups: Number of groups for grouped convolution.
        width_factor: Factor to determine width for grouped convolution.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4
    width = filters * width_factor

    x = conv_block(
        x,
        width,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    group_width = width // groups
    x = conv_block(
        x,
        width,
        kernel_size=3,
        strides=strides,
        groups=groups,
        group_width=group_width,
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


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNeXtModel(ResNetModel):
    """Instantiates the ResNeXt (Aggregated Residual Transformations) backbone.

    ResNeXt augments ResNet by replacing the bottleneck's 3x3 convolution
    with a grouped convolution of ``groups`` parallel paths
    (``cardinality``), exposing a third axis besides depth and width for
    capacity at near-constant FLOPs. The output tensor is the last layer
    output before the classifier head: the final-stage feature map
    ``(B, H, W, C)``, unpooled and head-free. :class:`ResNeXtImageClassify`
    composes this model and applies a GlobalAveragePooling2D + Dense
    head to produce logits.

    References:
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

    Args:
        block_fn: Callable, the residual block builder. Should accept
            ``(x, filters, strides=1, downsample=False, block_name=None)``
            and the ResNeXt-specific ``groups`` / ``width_factor``
            keyword arguments. Defaults to `resnext_block`.
        depths: List of ints, number of residual blocks per stage.
            Defaults to `[3, 4, 6, 3]`.
        filters: List of ints, base filter counts per stage (the final
            output width is ``filters[i] * expansion``).
            Defaults to `[64, 128, 256, 512]`.
        groups: Integer, number of groups (cardinality) for the grouped
            3x3 convolution inside each block. Defaults to `32`.
        width_factor: Integer, width scaling factor applied to the
            grouped convolution channels. Defaults to `2`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (4 tensors, one per ResNeXt stage).
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"ResNeXtModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNeXtConfig
    HUB_REPO_SIBLINGS = RESNEXT_HUB_SIBLINGS

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ResNeXtImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ResNeXtImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        block_fn=resnext_block,
        depths=(3, 4, 6, 3),
        filters=(64, 128, 256, 512),
        groups=32,
        width_factor=2,
        as_backbone=False,
        name="ResNeXtModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        super().__init__(
            block_fn=block_fn,
            depths=depths,
            filters=filters,
            groups=groups,
            width_factor=width_factor,
            as_backbone=as_backbone,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNeXtImageClassify(ResNetImageClassify):
    """Instantiates the ResNeXt (Aggregated Residual Transformations) classifier.

    This classifier wraps a :class:`ResNeXtModel` backbone and attaches
    a GlobalAveragePooling2D + Dense head to produce ``num_classes``
    class logits. All architectural parameters are forwarded to the
    underlying :class:`ResNeXtModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

    Args:
        block_fn: Callable, the residual block builder. Should accept
            ``(x, filters, strides=1, downsample=False, block_name=None)``
            and the ResNeXt-specific ``groups`` / ``width_factor``
            keyword arguments. Defaults to `resnext_block`.
        depths: List of ints, number of residual blocks per stage.
            Defaults to `[3, 4, 6, 3]`.
        filters: List of ints, base filter counts per stage (the final
            output width is ``filters[i] * expansion``).
            Defaults to `[64, 128, 256, 512]`.
        groups: Integer, number of groups (cardinality) for the grouped
            3x3 convolution inside each block. Defaults to `32`.
        senet: Boolean, whether to apply Squeeze-and-Excitation inside
            each block. Defaults to `False`.
        width_factor: Integer, width scaling factor applied to the
            grouped convolution channels. Defaults to `2`.
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
            named `f"{name}_backbone"`. Defaults to `"ResNeXtImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResNeXtConfig
    HUB_REPO_SIBLINGS = RESNEXT_HUB_SIBLINGS

    def __init__(
        self,
        block_fn=resnext_block,
        depths=(3, 4, 6, 3),
        filters=(64, 128, 256, 512),
        groups=32,
        senet=False,
        width_factor=2,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNeXtImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ResNeXtModel(
            block_fn=block_fn,
            depths=depths,
            filters=filters,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
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
            name="predictions",
        )(x)

        super(ResNetImageClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.block_fn = block_fn
        self.depths = depths
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
