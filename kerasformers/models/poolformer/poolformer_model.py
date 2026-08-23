import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .poolformer_config import PoolFormerConfig

# The backbone (PoolFormerModel) and classifier (PoolFormerImageClassify) share the
# variant's repo, whose kf_config.json declares PoolFormerImageClassify.
POOLFORMER_HUB_SIBLINGS = frozenset({"PoolFormerModel", "PoolFormerImageClassify"})


def mlp_block(x, hidden_dim, embed_dim, drop_rate, data_format, name):
    """
    MLP block using 1x1 convolutions for vision models.

    Args:
        x: Input tensor to the MLP block.
        hidden_dim: Number of filters in the first (hidden) convolution layer.
        embed_dim: Number of filters in the second (output) convolution layer.
        drop_rate: Dropout rate applied after each convolution layer.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Base name for the layers in this block, used for layer naming.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Conv2D(
        filters=hidden_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_1")(x)

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_2")(x)

    return x


def poolformer_block(
    x,
    embed_dim,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    init_scale,
    data_format,
    channels_axis,
    name,
):
    """PoolFormer block: pooling-based token mixer + Conv MLP with PoolFormerLayerScale.

    Two residual branches each wrapped with GroupNorm + PoolFormerLayerScale +
    stochastic depth. Token mixing is implemented as
    ``avg_pool(x) - x`` (equivalent to subtracting the pooled signal).

    Args:
        x: Input feature map.
        embed_dim: Channel dimension of this block.
        mlp_ratio: Hidden-dim multiplier inside the Conv MLP.
        drop_rate: Dropout rate inside the MLP.
        drop_path_rate: Stochastic-depth drop rate (no-op when 0).
        init_scale: Initial PoolFormerLayerScale value for the residual branches.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis of the channel dimension.
        name: Layer-name prefix.

    Returns:
        Output tensor with the same shape and layout as ``x``.
    """
    shortcut = x

    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_1"
    )(x)

    x_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, padding="same", data_format=data_format
    )(x)
    x = layers.Subtract(name=f"{name}_token_mixer")([x_pool, x])

    layer_scale_1 = PoolFormerLayerScale(init_scale, name=f"{name}_layerscale_1")(x)

    if drop_path_rate > 0:
        layer_scale_1 = PoolFormerStochasticDepth(drop_path_rate)(layer_scale_1)

    x = layers.Add(name=f"{name}_add_1")([shortcut, layer_scale_1])

    shortcut = x
    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_2"
    )(x)
    x = mlp_block(
        x,
        hidden_dim=int(embed_dim * mlp_ratio),
        embed_dim=embed_dim,
        drop_rate=drop_rate,
        data_format=data_format,
        name=f"{name}_mlp",
    )

    layer_scale_2 = PoolFormerLayerScale(init_scale, name=f"{name}_layerscale_2")(x)

    if drop_path_rate > 0:
        layer_scale_2 = PoolFormerStochasticDepth(drop_path_rate)(layer_scale_2)

    x = layers.Add(name=f"{name}_add_2")([shortcut, layer_scale_2])

    return x


def poolformer_backbone_feature(
    inputs,
    *,
    embed_dim,
    depths,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    init_scale,
    data_format,
    channels_axis,
    return_stages=False,
):
    """PoolFormer stem (7x7 stride-4 conv) + 4 stages with 3x3 stride-2 downsamples.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        embed_dim: Per-stage channel dimensions (length-4 sequence).
        depths: Per-stage PoolFormer block counts (length-4 sequence).
        mlp_ratio: Hidden-dim multiplier inside each block's MLP.
        drop_rate: Dropout rate inside MLPs.
        drop_path_rate: Maximum stochastic-depth rate (linearly scaled across blocks).
        init_scale: Initial PoolFormerLayerScale value for the residual branches.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis of the channel dimension.
        return_stages: If True, return a list of the 4 per-stage feature maps
            (each captured post-stage, pre-downsample). If False (default),
            return the final stage feature map only.

    Returns:
        Final stage feature map ``(B, H, W, C)``, or a list of 4 per-stage
        feature maps when ``return_stages=True``.
    """
    x = layers.ZeroPadding2D(
        padding=((2, 2), (2, 2)), data_format=data_format, name="stem_pad"
    )(inputs)
    x = layers.Conv2D(
        filters=embed_dim[0],
        kernel_size=7,
        strides=4,
        use_bias=True,
        padding="valid",
        data_format=data_format,
        name="stem_conv",
    )(x)

    total_blocks = sum(depths)
    dpr = [val * drop_path_rate / total_blocks for val in range(total_blocks)]
    cur = 0

    stages = []
    for stage_idx in range(len(depths)):
        for block_idx in range(depths[stage_idx]):
            x = poolformer_block(
                x,
                embed_dim=embed_dim[stage_idx],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[cur],
                init_scale=init_scale,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"stage_{stage_idx}_block_{block_idx}",
            )
            cur += 1

        stages.append(x)

        if stage_idx < len(depths) - 1:
            x = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)),
                data_format=data_format,
                name=f"stage_{stage_idx + 1}_downsample_pad",
            )(x)
            x = layers.Conv2D(
                filters=embed_dim[stage_idx + 1],
                kernel_size=3,
                strides=2,
                use_bias=True,
                padding="valid",
                data_format=data_format,
                name=f"stage_{stage_idx + 1}_downsample_conv",
            )(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class PoolFormerModel(BaseModel):
    """Instantiates the PoolFormer backbone.

    PoolFormer is a MetaFormer instantiation where the token-mixing
    primitive is a simple average pooling layer (implemented as
    ``avg_pool(x) - x``) rather than self-attention or an MLP. By
    matching transformer-style results with such a trivial mixer, it
    demonstrates that the MetaFormer template: norm + token mixer +
    norm + channel MLP, each wrapped in residual + PoolFormerLayerScale + drop
    path: is what matters, not the mixer itself. The network has 4
    stages with stride-2 conv downsamples between them.

    Output is the last layer output before the classifier head: the
    final stage feature map ``(B, H, W, C)`` (or ``(B, C, H, W)`` for
    channels_first). :class:`PoolFormerImageClassify` composes this model
    and applies a GlobalAveragePooling2D + LayerNorm + Dense head.

    References:
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

    Args:
        embed_dim: Tuple of integers, per-stage channel dimensions
            (length-4). Defaults to `(64, 128, 320, 512)`.
        depths: Tuple of integers, per-stage PoolFormer block counts
            (length-4). Defaults to `(2, 2, 6, 2)`.
        mlp_ratio: Float, hidden-dim multiplier inside each block's MLP.
            Defaults to `4.0`.
        drop_rate: Float, dropout rate inside MLPs. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled across blocks. Defaults to `0.0`.
        init_scale: Float, initial PoolFormerLayerScale value for the residual
            branches. Defaults to `1e-5`.
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
            use as a backbone network. When True, returns a list of the
            4 per-stage feature maps. Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"PoolFormerModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PoolFormerConfig
    HUB_REPO_SIBLINGS = POOLFORMER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with PoolFormerImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = PoolFormerImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_poolformer_timm_to_keras import transfer_poolformer_weights

        transfer_poolformer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dim=(64, 128, 320, 512),
        depths=(2, 2, 6, 2),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="PoolFormerModel",
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
        x = poolformer_backbone_feature(
            x,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_scale=init_scale,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
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
                "depths": self.depths,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class PoolFormerImageClassify(BaseModel):
    """Instantiates the PoolFormer classifier.

    This classifier wraps a :class:`PoolFormerModel` backbone and
    attaches a GlobalAveragePooling2D + LayerNorm + Dense head on the
    final feature map to produce ``num_classes`` class logits. All
    architectural parameters are forwarded to the underlying
    :class:`PoolFormerModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

    Args:
        embed_dim: Tuple of integers, per-stage channel dimensions
            (length-4). Defaults to `(64, 128, 320, 512)`.
        depths: Tuple of integers, per-stage PoolFormer block counts
            (length-4). Defaults to `(2, 2, 6, 2)`.
        mlp_ratio: Float, hidden-dim multiplier inside each block's MLP.
            Defaults to `4.0`.
        drop_rate: Float, dropout rate inside MLPs. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled across blocks. Defaults to `0.0`.
        init_scale: Float, initial PoolFormerLayerScale value for the residual
            branches. Defaults to `1e-5`.
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
            named `f"{name}_backbone"`. Defaults to `"PoolFormerImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PoolFormerConfig
    HUB_REPO_SIBLINGS = POOLFORMER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_poolformer_timm_to_keras import transfer_poolformer_weights

        transfer_poolformer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dim=(64, 128, 320, 512),
        depths=(2, 2, 6, 2),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PoolFormerImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = PoolFormerModel(
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_scale=init_scale,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.LayerNormalization(epsilon=1e-6, name="layernorm")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
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
                "depths": self.depths,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class PoolFormerLayerScale(layers.Layer):
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


@keras.saving.register_keras_serializable(package="kerasformers")
class PoolFormerStochasticDepth(keras.layers.Layer):
    """Stochastic depth: randomly drops the residual path during training (identity at inference)."""

    def __init__(self, drop_path_rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= drop_path_rate <= 1:
            raise ValueError(
                f"drop_path_rate should be between 0 and 1, got {drop_path_rate}"
            )
        self.drop_path_rate = drop_path_rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(keras.ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = keras.ops.cast(keras.ops.floor(random_tensor), x.dtype)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate, "seed": self.seed})
        return config
