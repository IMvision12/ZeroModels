import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.convnext.convnext_layers import (
    ConvNeXtGlobalResponseNorm,
    ConvNeXtLayerScale,
    ConvNeXtStochasticDepth,
)
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .convnext_config import ConvNeXtConfig

# The backbone (ConvNeXtModel) and classifier (ConvNeXtImageClassify) share the
# variant's weights repo, whose kf_config.json declares ConvNeXtImageClassify.
CONVNEXT_HUB_SIBLINGS = frozenset({"ConvNeXtModel", "ConvNeXtImageClassify"})


def spatial_layer_norm(x, data_format, epsilon=1e-6, name=None):
    """LayerNorm over channels for spatial feature maps.

    Args:
        x: Input feature tensor with spatial dims.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        epsilon: Small constant added to the variance for numerical stability.
        name: Optional name prefix for the underlying layers.

    Returns:
        Tensor with the same shape as ``x`` after channel-wise LayerNorm.
    """
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=f"{name}_to_cl" if name else None)(x)
    x = layers.LayerNormalization(axis=-1, epsilon=epsilon, name=name)(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=f"{name}_to_cf" if name else None)(x)
    return x


def convnext_block(
    inputs,
    projection_dim,
    channels_axis,
    data_format,
    drop_path_rate=0.0,
    layer_scale_init=1e-6,
    name=None,
    use_grn=False,
    use_conv=False,
):
    """ConvNeXt block: DWConv -> LN -> (Conv|Dense) -> GELU -> (GRN) -> (Conv|Dense).

    Args:
        inputs: Input feature map for the residual block.
        projection_dim: Output channel count of the block.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        drop_path_rate: Stochastic depth drop probability for this block.
        layer_scale_init: Initial value for ConvNeXtLayerScale; pass ``None`` to skip.
        name: Name prefix for sub-layers inside the block.
        use_grn: Whether to apply ConvNeXtGlobalResponseNorm (ConvNeXtV2 style).
        use_conv: If True, use 1x1 Conv2D for the MLP; else use Dense layers.

    Returns:
        Output tensor with shape matching ``inputs`` and ``projection_dim`` channels.
    """
    x = layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name=name + "_depthwise_conv",
    )(inputs)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=name + "_to_nhwc")(x)

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=name + "_layernorm")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim * 4,
            1,
            data_format="channels_last",
            name=name + "_conv_1",
        )(x)
    else:
        x = layers.Dense(4 * projection_dim, name=name + "_dense_1")(x)
    x = layers.Activation("gelu", name=name + "_gelu")(x)
    if use_grn:
        x = ConvNeXtGlobalResponseNorm(name=name + "_grn")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim,
            1,
            data_format="channels_last",
            name=name + "_conv_2",
        )(x)
    else:
        x = layers.Dense(projection_dim, name=name + "_dense_2")(x)

    if layer_scale_init is not None:
        x = ConvNeXtLayerScale(layer_scale_init, name=name + "_layer_scale")(x)

    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=name + "_to_nchw")(x)

    if drop_path_rate:
        x = ConvNeXtStochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)

    return layers.Add(name=name + "_add")([inputs, x])


def convnext_backbone_feature(
    inputs,
    *,
    depths,
    projection_dim,
    drop_path_rate,
    layer_scale_init,
    use_conv,
    use_grn,
    data_format,
    channels_axis,
    return_stages=False,
):
    """ConvNeXt stem + 4 stages.

    Args:
        inputs: Input image tensor (post-normalization).
        depths: Number of blocks per stage (length-4 list).
        projection_dim: Channel count per stage (length-4 list).
        drop_path_rate: Maximum stochastic-depth rate; linearly scaled across blocks.
        layer_scale_init: ConvNeXtLayerScale init; pass ``None`` to disable.
        use_conv: Use 1x1 Conv2D inside blocks instead of Dense.
        use_grn: Enable ConvNeXtGlobalResponseNorm (ConvNeXtV2 style).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis index of the channels dimension.
        return_stages: If True, return a list of per-stage feature maps
            (one per ConvNeXt stage). If False (default), return only the
            final stage feature map.

    Returns:
        Final stage feature map ``(B, H, W, C)``, or a list of per-stage
        feature maps when ``return_stages=True``.
    """
    x = layers.Conv2D(
        projection_dim[0],
        kernel_size=4,
        strides=4,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = spatial_layer_norm(x, data_format, epsilon=1e-6, name="stem_layernorm")

    n = sum(depths)
    depth_drop_rates = [drop_path_rate * i / max(n - 1, 1) for i in range(n)]
    cur = 0
    stages = []
    for i in range(len(depths)):
        if i > 0:
            x = spatial_layer_norm(
                x,
                data_format,
                epsilon=1e-6,
                name=f"stages_{i}_downsampling_layernorm",
            )
            x = layers.Conv2D(
                projection_dim[i],
                kernel_size=2,
                strides=2,
                data_format=data_format,
                name=f"stages_{i}_downsampling_conv",
            )(x)
        for j in range(depths[i]):
            x = convnext_block(
                x,
                projection_dim=projection_dim[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init=layer_scale_init,
                use_grn=use_grn,
                use_conv=use_conv,
                channels_axis=channels_axis,
                data_format=data_format,
                name=f"stages_{i}_blocks_{j}",
            )
        cur += depths[i]
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtModel(BaseModel):
    """Instantiates the ConvNeXt backbone.

    ConvNeXt is a modernized ConvNet that adapts ViT design principles to
    pure CNNs: depthwise 7x7 convolutions, LayerNorm in place of
    BatchNorm, GELU activations, and inverted-bottleneck blocks organized
    into 4 hierarchical stages. Output is the last layer output before
    the classifier head: the final stage feature map ``(B, H, W, C)``.
    :class:`ConvNeXtImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + LayerNorm + Dense head to produce logits.

    References:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one per ConvNeXt stage).
            Defaults to `False`.
        depths: Tuple of 4 integers, number of ConvNeXt blocks per stage.
            Defaults to `(3, 3, 9, 3)`.
        projection_dim: Tuple of 4 integers, channel count per stage.
            Defaults to `(96, 192, 384, 768)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly scaled from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init: Float, initial value for per-channel
            ConvNeXtLayerScale. Pass ``None`` to disable ConvNeXtLayerScale.
            Defaults to `1e-6`.
        use_conv: Boolean, if True, use 1x1 Conv2D layers inside each
            block's MLP; otherwise use Dense layers. Defaults to `False`.
        use_grn: Boolean, whether to apply ConvNeXtGlobalResponseNorm inside each
            block (ConvNeXtV2 recipe). Defaults to `False`.
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
        name: String, the name of the model. Defaults to `"ConvNeXtModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvNeXtConfig
    HUB_REPO_SIBLINGS = CONVNEXT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ConvNeXtImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ConvNeXtImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_convnext_timm_to_keras import transfer_convnext_weights

        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dim=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="ConvNeXtModel",
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
        x = convnext_backbone_feature(
            x,
            depths=depths,
            projection_dim=projection_dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            use_conv=use_conv,
            use_grn=use_grn,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.projection_dim = list(projection_dim)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.use_conv = use_conv
        self.use_grn = use_grn
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
                "projection_dim": self.projection_dim,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
                "use_conv": self.use_conv,
                "use_grn": self.use_grn,
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
class ConvNeXtImageClassify(BaseModel):
    """Instantiates the ConvNeXt classifier.

    This classifier wraps a :class:`ConvNeXtModel` backbone and attaches
    a GlobalAveragePooling2D + LayerNorm + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`ConvNeXtModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        depths: Tuple of 4 integers, number of ConvNeXt blocks per stage.
            Defaults to `(3, 3, 9, 3)`.
        projection_dim: Tuple of 4 integers, channel count per stage.
            Defaults to `(96, 192, 384, 768)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly scaled from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init: Float, initial value for per-channel
            ConvNeXtLayerScale. Pass ``None`` to disable ConvNeXtLayerScale.
            Defaults to `1e-6`.
        use_conv: Boolean, if True, use 1x1 Conv2D layers inside each
            block's MLP; otherwise use Dense layers. Defaults to `False`.
        use_grn: Boolean, whether to apply ConvNeXtGlobalResponseNorm inside each
            block (ConvNeXtV2 recipe). Defaults to `False`.
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
            named `f"{name}_backbone"`. Defaults to `"ConvNeXtImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvNeXtConfig
    HUB_REPO_SIBLINGS = CONVNEXT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_convnext_timm_to_keras import transfer_convnext_weights

        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dim=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvNeXtImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvNeXtModel(
            depths=depths,
            projection_dim=projection_dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            use_conv=use_conv,
            use_grn=use_grn,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = list(depths)
        self.projection_dim = list(projection_dim)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.use_conv = use_conv
        self.use_grn = use_grn
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
                "projection_dim": self.projection_dim,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
                "use_conv": self.use_conv,
                "use_grn": self.use_grn,
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
