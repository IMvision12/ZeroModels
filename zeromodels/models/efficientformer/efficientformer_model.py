import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.efficientformer.efficientformer_layers import (
    EfficientFormerAttention4D,
    EfficientFormerLayerScale,
    EfficientFormerStochasticDepth,
)
from zeromodels.utils import standardize_input_shape

from .efficientformer_config import EfficientFormerConfig

# The backbone (EfficientFormerModel) and classifier (EfficientFormerImageClassify)
# share the variant's repo, whose zm_config.json declares
# EfficientFormerImageClassify.
EFFICIENTFORMER_HUB_SIBLINGS = frozenset(
    {"EfficientFormerModel", "EfficientFormerImageClassify"}
)


def conv_mlp_block(
    inputs,
    hidden_features,
    out_features,
    drop=0.0,
    channels_axis=-1,
    data_format="channels_last",
    name=None,
):
    """MLP block built from 1x1 convolutions for 2D spatial feature maps.

    Args:
        inputs: Input feature tensor in NHWC or NCHW format.
        hidden_features: Channel width of the intermediate (hidden) projection.
        out_features: Channel width of the output projection.
        drop: Dropout rate applied after each conv.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        name: Prefix used to name the layers inside the block.

    Returns:
        Output feature tensor with ``out_features`` channels.
    """
    x = layers.Conv2D(
        hidden_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_1"
    )(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)

    x = layers.Conv2D(
        out_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_2"
    )(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def mlp_block(inputs, hidden_features, out_features, drop=0.0, name=None):
    """Standard two-layer Dense MLP block for 1D token sequences.

    Args:
        inputs: Input token tensor of shape ``(B, N, C)``.
        hidden_features: Width of the intermediate (hidden) projection.
        out_features: Width of the output projection.
        drop: Dropout rate applied after each dense layer.
        name: Prefix used to name the layers inside the block.

    Returns:
        Output token tensor of shape ``(B, N, out_features)``.
    """
    x = layers.Dense(hidden_features, use_bias=True, name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"{name}_dense_2")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def meta_block_2d(
    inputs,
    dim,
    pool_size=3,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init=1e-5,
    channels_axis=-1,
    data_format="channels_last",
    name=None,
):
    """2D MetaBlock with a pooling token mixer for convolutional stages.

    Args:
        inputs: Input 2D feature tensor.
        dim: Channel width of the block.
        pool_size: Kernel size of the average-pool token mixer.
        mlp_ratio: Hidden-feature multiplier for the conv MLP.
        drop: Dropout rate applied inside the MLP.
        drop_path: Stochastic-depth drop rate applied on each residual branch.
        layer_scale_init: Initial value for the EfficientFormerLayerScale gamma parameter.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        name: Prefix used to name the layers inside the block.

    Returns:
        Output 2D feature tensor with ``dim`` channels.
    """
    # Token mixer (pooling)
    pooled = layers.AveragePooling2D(
        pool_size=pool_size,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name}_pool_pool",
    )(inputs)
    x = layers.Subtract(name=f"{name}_pool_sub")([pooled, inputs])
    x = EfficientFormerLayerScale(layer_scale_init, name=f"{name}_ls1")(x)
    if drop_path > 0.0:
        x = EfficientFormerStochasticDepth(drop_path, name=f"{name}_drop_path1")(x)
    x = layers.Add(name=f"{name}_add1")([inputs, x])

    # MLP
    y = conv_mlp_block(
        x,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{name}_mlp",
    )
    y = EfficientFormerLayerScale(layer_scale_init, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = EfficientFormerStochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


def meta_block_1d(
    inputs,
    dim,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init=1e-5,
    resolution=7,
    name=None,
):
    """1D MetaBlock with a self-attention token mixer for transformer stages.

    Args:
        inputs: Input token tensor of shape ``(B, N, dim)``.
        dim: Channel width of the block.
        mlp_ratio: Hidden-feature multiplier for the Dense MLP.
        drop: Dropout rate applied inside the MLP.
        drop_path: Stochastic-depth drop rate applied on each residual branch.
        layer_scale_init: Initial value for the EfficientFormerLayerScale gamma parameter.
        resolution: Spatial side length used to size the attention biases.
        name: Prefix used to name the layers inside the block.

    Returns:
        Output token tensor of shape ``(B, N, dim)``.
    """
    y = layers.LayerNormalization(epsilon=1e-6, axis=-1, name=f"{name}_norm1")(inputs)
    y = EfficientFormerAttention4D(dim=dim, resolution=resolution, name=f"{name}_attn")(
        y
    )
    y = EfficientFormerLayerScale(layer_scale_init, name=f"{name}_ls1")(y)
    if drop_path > 0.0:
        y = EfficientFormerStochasticDepth(drop_path, name=f"{name}_drop_path1")(y)
    x = layers.Add(name=f"{name}_add1")([inputs, y])

    # MLP
    y = layers.LayerNormalization(epsilon=1e-6, axis=-1, name=f"{name}_norm2")(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        name=f"{name}_mlp",
    )
    y = EfficientFormerLayerScale(layer_scale_init, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = EfficientFormerStochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


def efficientformer_backbone_feature(
    inputs,
    *,
    depths,
    embed_dim,
    num_vit,
    mlp_ratio,
    pool_size,
    drop_rate,
    drop_path_rate,
    layer_scale_init,
    image_h,
    data_format,
    channels_axis,
    return_stages=False,
):
    """EfficientFormer stem + four hybrid (pool + optional transformer) stages.

    The output is 1D ``(B, N, C)`` if the final stage has any transformer
    (``num_vit > 0``) blocks, otherwise 2D.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        depths: Per-stage block counts (length = number of stages).
        embed_dim: Per-stage channel widths.
        num_vit: Number of transformer blocks at the tail of the final stage.
        mlp_ratio: Hidden-feature multiplier shared by all MLP sub-blocks.
        pool_size: Kernel size of the pooling token mixer used by 2D blocks.
        drop_rate: Dropout rate applied inside the MLPs.
        drop_path_rate: Maximum stochastic-depth drop rate (linearly ramped).
        layer_scale_init: Initial value for EfficientFormerLayerScale gamma parameters.
        image_h: Input image height; used to compute the final-stage resolution.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps (one
            tensor per element of ``depths``); otherwise return the final
            stage's feature tensor.

    Returns:
        Final-stage feature tensor, or a list of per-stage feature tensors when
        ``return_stages`` is True.
    """
    x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad1")(
        inputs
    )
    x = layers.Conv2D(
        embed_dim[0] // 2,
        kernel_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
        name="stem_conv1",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, epsilon=1e-5, name="stem_norm1")(
        x
    )
    x = layers.Activation("relu", name="stem_act1")(x)

    x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad2")(x)
    x = layers.Conv2D(
        embed_dim[0],
        kernel_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
        name="stem_conv2",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, epsilon=1e-5, name="stem_norm2")(
        x
    )
    x = layers.Activation("relu", name="stem_act2")(x)

    num_stages = len(depths)
    n = sum(depths)
    dpr = [drop_path_rate * i / max(n - 1, 1) for i in range(n)]
    cur = 0

    stages = []
    for i in range(num_stages):
        if i > 0:
            x = layers.ZeroPadding2D(
                padding=1,
                data_format=data_format,
                name=f"stages_{i}_downsample_pad",
            )(x)
            x = layers.Conv2D(
                embed_dim[i],
                kernel_size=3,
                strides=2,
                padding="valid",
                data_format=data_format,
                name=f"stages_{i}_downsample_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                name=f"stages_{i}_downsample_norm",
            )(x)

        is_last_stage = i == num_stages - 1
        use_transformer = is_last_stage and num_vit > 0

        for j in range(depths[i]):
            remain_idx = depths[i] - j - 1

            if use_transformer and num_vit > remain_idx and j == depths[i] - num_vit:
                resolution = image_h // (4 * (2**i))
                ch = x.shape[channels_axis]
                if data_format == "channels_first":
                    x = layers.Permute((2, 3, 1), name=f"stages_{i}_to_nhwc")(x)
                x = layers.Reshape((-1, ch), name=f"stages_{i}_flat")(x)

            if use_transformer and num_vit > remain_idx:
                x = meta_block_1d(
                    x,
                    dim=embed_dim[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    layer_scale_init=layer_scale_init,
                    resolution=resolution,
                    name=f"stages_{i}_blocks_{j}",
                )
            else:
                x = meta_block_2d(
                    x,
                    dim=embed_dim[i],
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    layer_scale_init=layer_scale_init,
                    channels_axis=channels_axis,
                    data_format=data_format,
                    name=f"stages_{i}_blocks_{j}",
                )

        if return_stages:
            stages.append(x)

        cur += depths[i]

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientFormerModel(BaseModel):
    """Instantiates the EfficientFormer backbone.

    EfficientFormer is a hybrid CNN-Transformer designed for fast
    on-device inference. The network keeps pure-convolutional stages
    (MetaBlock2D with an average-pool token mixer + EfficientFormerLayerScale +
    stochastic depth) at high resolution where attention would be too
    expensive, then switches to transformer (MetaBlock1D) blocks at the
    lowest resolution where attention is cheap. Each stage is preceded
    by a 3x3 strided downsampling conv, and the final stage's tail
    ``num_vit`` blocks operate on a flattened token sequence so that
    self-attention can be applied.

    Output is the last layer output before the classifier head: a 1D
    token tensor of shape ``(B, N, C)`` when the final stage uses
    transformer blocks (``num_vit > 0``), otherwise a 2D feature map of
    shape ``(B, H, W, C)``. :class:`EfficientFormerImageClassify` composes
    this model and applies a LayerNorm + mean-pool + Dropout + dual
    Dense (head + head_dist) + Average head on top.

    References:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        depths: Sequence of integers, per-stage block counts. The length
            sets the number of stages.
        embed_dim: Sequence of integers, per-stage channel widths.
        num_vit: Integer, number of transformer (MetaBlock1D) blocks
            placed at the tail of the final stage. Defaults to `1`.
        mlp_ratio: Float, hidden-feature multiplier shared by every MLP
            sub-block. Defaults to `4.0`.
        pool_size: Integer, kernel size of the average-pool token mixer
            used by the 2D MetaBlocks. Defaults to `3`.
        drop_rate: Float, dropout rate applied inside the MLPs (and the
            head Dropout in the classifier). Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly ramped across all blocks. Defaults to `0.0`.
        layer_scale_init: Float, initial value for the per-channel
            EfficientFormerLayerScale gamma applied on every residual branch.
            Defaults to `1e-5`.
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
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one tensor per element of
            ``depths``). Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"EfficientFormerModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientFormerConfig
    HUB_REPO_SIBLINGS = EFFICIENTFORMER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with EfficientFormerImageClassify (which
        # the zm_config declares); build from zm_config, then copy backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = EfficientFormerImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientformer_timm_to_keras import (
            transfer_efficientformer_weights,
        )

        transfer_efficientformer_weights(keras_model, state_dict)

    def __init__(
        self,
        depths,
        embed_dim,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init=1e-5,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="EfficientFormerModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        image_size = standardize_input_shape(image_size, data_format)

        if data_format == "channels_last":
            image_h = image_size[0]
        else:
            image_h = image_size[1]

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = img_input
        x = efficientformer_backbone_feature(
            x,
            depths=depths,
            embed_dim=embed_dim,
            num_vit=num_vit,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            image_h=image_h,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = depths
        self.embed_dim = embed_dim
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
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
class EfficientFormerImageClassify(BaseModel):
    """Instantiates the EfficientFormer classifier.

    This classifier wraps a :class:`EfficientFormerModel` backbone and
    attaches a LayerNorm + mean-pool + Dropout + dual Dense (``head``
    and ``head_dist`` for DeiT-style hard distillation) + Average head
    to produce ``num_classes`` class logits. The two Dense logits are
    averaged before the optional final activation. All architectural
    parameters are forwarded to the underlying
    :class:`EfficientFormerModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        depths: Sequence of integers, per-stage block counts in the
            backbone. The length sets the number of stages.
        embed_dim: Sequence of integers, per-stage channel widths.
        num_vit: Integer, number of transformer (MetaBlock1D) blocks
            placed at the tail of the final backbone stage.
            Defaults to `1`.
        mlp_ratio: Float, hidden-feature multiplier shared by every MLP
            sub-block. Defaults to `4.0`.
        pool_size: Integer, kernel size of the average-pool token mixer
            used by the 2D MetaBlocks. Defaults to `3`.
        drop_rate: Float, dropout rate applied inside the MLPs and
            before the dual Dense classifier. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly ramped across all blocks. Defaults to `0.0`.
        layer_scale_init: Float, initial value for the per-channel
            EfficientFormerLayerScale gamma applied on every residual branch.
            Defaults to `1e-5`.
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
            applied on top of the averaged ``head`` + ``head_dist``
            logits. Use `"linear"` to return raw logits or `"softmax"`
            to return class probabilities. Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to
            `"EfficientFormerImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = EfficientFormerConfig
    HUB_REPO_SIBLINGS = EFFICIENTFORMER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_efficientformer_timm_to_keras import (
            transfer_efficientformer_weights,
        )

        transfer_efficientformer_weights(keras_model, state_dict)

    def __init__(
        self,
        depths,
        embed_dim,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init=1e-5,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="EfficientFormerImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        backbone = EfficientFormerModel(
            depths=depths,
            embed_dim=embed_dim,
            num_vit=num_vit,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = backbone.output
        x = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_norm")(x)
        x = layers.Lambda(lambda v: ops.mean(v, axis=1), name="global_pool")(x)
        x = layers.Dropout(drop_rate, name="head_drop")(x)

        x_cls = layers.Dense(num_classes, activation=None, name="head", use_bias=True)(
            x
        )
        x_dist = layers.Dense(
            num_classes, activation=None, name="head_dist", use_bias=True
        )(x)

        out = layers.Average(name="avg_predictions")([x_cls, x_dist])
        if classifier_activation:
            out = layers.Activation(classifier_activation, name="predictions")(out)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = depths
        self.embed_dim = embed_dim
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init": self.layer_scale_init,
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
