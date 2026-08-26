import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .convmixer_config import ConvMixerConfig

# The backbone (ConvMixerModel) and classifier (ConvMixerImageClassify) share the
# variant's weights repo, whose kf_config.json declares ConvMixerImageClassify.
CONVMIXER_HUB_SIBLINGS = frozenset({"ConvMixerModel", "ConvMixerImageClassify"})


def convmixer_block(
    x, filters, kernel_size, activation, channels_axis, data_format, name
):
    """A building block for the ConvMixer architecture.

    Args:
        x: input tensor.
        filters: int, the number of output filters for the convolution layers.
        kernel_size: int, the size of the convolution kernel.
        activation: string, name of the activation function to be applied within
            the Conv2D layers (e.g., 'gelu', 'relu').
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: string, block name.

    Returns:
        Output tensor for the block.
    """
    inputs = x
    x = layers.DepthwiseConv2D(
        kernel_size,
        1,
        padding="same",
        use_bias=True,
        activation=activation,
        data_format=data_format,
        name=f"{name}_depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)

    x = layers.Add(name=f"{name}_add")([inputs, x])

    x = layers.Conv2D(
        filters,
        1,
        1,
        activation=activation,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)

    return x


def convmixer_backbone_feature(
    inputs,
    *,
    embed_dim,
    depth,
    kernel_size,
    patch_size,
    activation,
    data_format,
    channels_axis,
    return_stages=False,
):
    """Stem + N ConvMixer blocks, returning the final feature map.

    Args:
        inputs: Input image tensor (post-normalization).
        embed_dim: Channel dimension carried throughout the model.
        depth: Total number of ConvMixer blocks stacked after the stem.
        kernel_size: Depthwise convolution kernel size inside each block.
        patch_size: Stride/kernel of the patch-embedding stem convolution.
        activation: Activation name applied inside conv layers (e.g. ``"gelu"``).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis index of the channels dimension.
        return_stages: If True, return a singleton list ``[final]`` (ConvMixer
            has no natural multi-stage hierarchy: all blocks share the same
            spatial resolution and channel count). If False (default), return
            the final feature map directly.

    Returns:
        Final feature map ``(B, H, W, C)``, or a singleton list
        ``[final]`` when ``return_stages=True``.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        use_bias=True,
        activation=activation,
        data_format=data_format,
        name="stem_conv2d",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
    )(x)

    for i in range(depth):
        x = convmixer_block(
            x,
            embed_dim,
            kernel_size,
            activation,
            channels_axis,
            data_format,
            f"mixer_block_{i}",
        )

    if return_stages:
        return [x]
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvMixerModel(BaseModel):
    """Instantiates the ConvMixer backbone.

    ConvMixer is a patch-based mixer that alternates depthwise and
    pointwise convolutions at a single spatial resolution throughout the
    network: after a patch-embedding stem there is no further
    downsampling or channel hierarchy. Output is the last layer output
    before the classifier head: the final feature map ``(B, H, W, C)``.
    :class:`ConvMixerImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a singleton
            list ``[final]`` (ConvMixer has no natural multi-stage
            hierarchy). Defaults to `False`.
        embed_dim: Integer, channel dimension carried throughout the model.
            Defaults to `768`.
        depth: Integer, number of ConvMixer blocks stacked after the
            patch-embedding stem. Defaults to `32`.
        kernel_size: Integer, depthwise convolution kernel size inside
            each block. Defaults to `7`.
        patch_size: Integer, stride and kernel of the patch-embedding
            stem convolution. Defaults to `7`.
        activation: String, activation name applied inside conv layers.
            Defaults to `"gelu"`.
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
            Defaults to `"ConvMixerModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvMixerConfig
    HUB_REPO_SIBLINGS = CONVMIXER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ConvMixerImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ConvMixerImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_convmixer_timm_to_keras import transfer_convmixer_weights

        transfer_convmixer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dim=768,
        depth=32,
        kernel_size=7,
        patch_size=7,
        activation="gelu",
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="ConvMixerModel",
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
        x = convmixer_backbone_feature(
            x,
            embed_dim=embed_dim,
            depth=depth,
            kernel_size=kernel_size,
            patch_size=patch_size,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
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
class ConvMixerImageClassify(BaseModel):
    """Instantiates the ConvMixer classifier.

    This classifier wraps a :class:`ConvMixerModel` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`ConvMixerModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    Args:
        embed_dim: Integer, channel dimension carried throughout the model.
            Defaults to `768`.
        depth: Integer, number of ConvMixer blocks stacked after the
            patch-embedding stem. Defaults to `32`.
        kernel_size: Integer, depthwise convolution kernel size inside
            each block. Defaults to `7`.
        patch_size: Integer, stride and kernel of the patch-embedding
            stem convolution. Defaults to `7`.
        activation: String, activation name applied inside conv layers.
            Defaults to `"gelu"`.
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
            named `f"{name}_backbone"`. Defaults to `"ConvMixerImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvMixerConfig
    HUB_REPO_SIBLINGS = CONVMIXER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_convmixer_timm_to_keras import transfer_convmixer_weights

        transfer_convmixer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dim=768,
        depth=32,
        kernel_size=7,
        patch_size=7,
        activation="gelu",
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvMixerImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvMixerModel(
            embed_dim=embed_dim,
            depth=depth,
            kernel_size=kernel_size,
            patch_size=patch_size,
            activation=activation,
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
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation
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
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
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
