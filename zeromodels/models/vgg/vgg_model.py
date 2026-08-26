import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .vgg_config import VGGConfig

# The backbone (VGGModel) and classifier (VGGImageClassify) share the variant's
# weights repo, whose zm_config.json declares VGGImageClassify.
VGG_HUB_SIBLINGS = frozenset({"VGGModel", "VGGImageClassify"})


def vgg_block(
    inputs,
    num_filters,
    channels_axis,
    data_format,
    batch_norm=False,
    collect_stages=False,
):
    """Stack of Conv2D / [BN] / ReLU and MaxPool layers per the VGG recipe.

    ``num_filters`` is a list mixing ints (filter counts) and ``"M"`` markers
    for MaxPooling.

    Args:
        inputs: Input image tensor.
        num_filters: Mixed list of ints (Conv2D filter counts) and ``"M"``
            strings (MaxPool boundaries) defining the VGG architecture.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        batch_norm: If True, insert BatchNormalization after each Conv2D.
        collect_stages: If True, also return a list of feature maps captured
            right after each MaxPool.

    Returns:
        If ``collect_stages`` is False: the tensor produced after all VGG
        stages. If True: a tuple ``(x, stages)`` where ``stages`` is a list
        of per-MaxPool feature maps.
    """
    x = inputs
    layer_idx = 0
    stages = []

    for v in num_filters:
        if v == "M":
            x = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format=data_format,
                name=f"Max_Pool_{layer_idx}",
            )(x)
            layer_idx += 1
            stages.append(x)
        else:
            x = layers.Conv2D(
                v,
                3,
                padding="same",
                data_format=data_format,
                name=f"conv2d_{layer_idx}",
            )(x)
            layer_idx += 1

            if batch_norm:
                x = layers.BatchNormalization(
                    axis=channels_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    name=f"batchnorm_{layer_idx}",
                )(x)
                layer_idx += 1

            x = layers.ReLU(name=f"relu_{layer_idx}")(x)
            layer_idx += 1

    if collect_stages:
        return x, stages
    return x


def vgg_backbone_feature(
    inputs,
    *,
    num_filters,
    batch_norm,
    data_format,
    channels_axis,
    return_stages=False,
):
    """Convolutional stack + classification-head pre-logit convs.

    Args:
        inputs: Input image tensor (post-normalization).
        num_filters: VGG recipe (ints + ``"M"`` markers) describing the conv stack.
        batch_norm: Whether to apply BatchNormalization after each Conv2D.
        data_format: ``"channels_last"`` or ``"channels_first"`` Keras format.
        channels_axis: Axis index of the channels dimension.
        return_stages: If True, return a list of per-stage feature maps (one
            tensor captured right after each MaxPool). If False (default),
            return only the final pre-logits 4096-channel feature tensor.

    Returns:
        Final pre-logits 4096-channel feature tensor, or a list of per-stage
        feature maps when ``return_stages=True``.
    """
    x, stages = vgg_block(
        inputs,
        num_filters,
        batch_norm=batch_norm,
        channels_axis=channels_axis,
        data_format=data_format,
        collect_stages=True,
    )

    x = layers.Conv2D(4096, 7, data_format=data_format, name="conv_fc1")(x)
    x = layers.ReLU(name="relu_fc1")(x)
    x = layers.Dropout(0.5, name="dropout_fc1")(x)
    x = layers.Conv2D(4096, 1, data_format=data_format, name="conv_fc2")(x)
    x = layers.ReLU(name="relu_fc2")(x)
    x = layers.Dropout(0.5, name="dropout_fc2")(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class VGGModel(BaseModel):
    """Instantiates the VGG backbone.

    VGG is a sequential stack of 3x3 convolutions and 2x2 max-pooling
    layers arranged in 5 progressively-strided stages, followed by two
    pre-logit fully-connected layers (implemented here as 7x7 and 1x1
    convolutions) that form the classifier head's feature extractor.
    Output is the last layer output before the classifier head: the
    final pre-logits 4096-channel feature map ``(B, H, W, C)``.
    :class:`VGGImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + Dropout + Dense head to produce logits.

    References:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one tensor captured right after each
            MaxPool). Defaults to `False`.
        num_filters: List mixing integers (Conv2D filter counts) and
            ``"M"`` strings (MaxPool boundaries) that defines the VGG
            recipe. Must be provided. Defaults to `None`.
        batch_norm: Boolean, whether to insert BatchNormalization after
            each Conv2D. Defaults to `False`.
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
        name: String, the name of the model. Defaults to `"VGGModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = VGGConfig
    HUB_REPO_SIBLINGS = VGG_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with VGGImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = VGGImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_vgg_timm_to_keras import transfer_vgg_weights

        transfer_vgg_weights(keras_model, state_dict)

    def __init__(
        self,
        num_filters=None,
        batch_norm=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="VGGModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        if num_filters is None:
            raise ValueError("`num_filters` must be provided.")

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
        x = vgg_backbone_feature(
            x,
            num_filters=num_filters,
            batch_norm=batch_norm,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.num_filters = num_filters
        self.batch_norm = batch_norm
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
class VGGImageClassify(BaseModel):
    """Instantiates the VGG classifier.

    This classifier wraps a :class:`VGGModel` backbone and attaches a
    GlobalAveragePooling2D + Dropout + Dense head (the Dropout has
    rate 0, mirroring the original implementation) to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`VGGModel`; only ``num_classes``
    and ``classifier_activation`` are head-specific.

    References:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

    Args:
        num_filters: List mixing integers (Conv2D filter counts) and
            ``"M"`` strings (MaxPool boundaries) that defines the VGG
            recipe. Must be provided. Defaults to `None`.
        batch_norm: Boolean, whether to insert BatchNormalization after
            each Conv2D. Defaults to `False`.
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
            named `f"{name}_backbone"`. Defaults to `"VGGImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = VGGConfig
    HUB_REPO_SIBLINGS = VGG_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_vgg_timm_to_keras import transfer_vgg_weights

        transfer_vgg_weights(keras_model, state_dict)

    def __init__(
        self,
        num_filters=None,
        batch_norm=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="VGGImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if num_filters is None:
            raise ValueError("`num_filters` must be provided.")

        data_format = keras.config.image_data_format()

        backbone = VGGModel(
            num_filters=num_filters,
            batch_norm=batch_norm,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.Dropout(rate=0, name="dropout")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.num_filters = num_filters
        self.batch_norm = batch_norm
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
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
