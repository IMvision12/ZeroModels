import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .maxvit_config import MaxViTConfig
from .maxvit_layers import (
    MaxViTAttention,
    MaxViTGridPartition,
    MaxViTGridReverse,
    MaxViTWindowPartition,
    MaxViTWindowReverse,
)

# The backbone (MaxViTModel) and classifier (MaxViTImageClassify) share the variant's
# weights repo, whose zm_config.json declares MaxViTImageClassify.
MAXVIT_HUB_SIBLINGS = frozenset({"MaxViTModel", "MaxViTImageClassify"})


def maxvit_gelu_approximate(x):
    """GELU activation with tanh approximation, matching timm's GELUTanh.

    Args:
        x: Input tensor.

    Returns:
        Tensor with the tanh-approximated GELU applied element-wise.
    """
    return keras.activations.gelu(x, approximate=True)


def maxvit_mbconv_block(
    x,
    in_channels,
    out_channels,
    expand_ratio=4,
    se_ratio=0.0625,
    stride=1,
    data_format="channels_last",
    channels_axis=-1,
    prefix="",
):
    """Mobile Inverted Bottleneck Convolution (MBConv) block for MaxViT.

    Implements a pre-norm MBConv block with Squeeze-and-Excitation. The block
    consists of: BatchNorm -> 1x1 expand -> BN+GELU -> 3x3 depthwise ->
    BN+GELU -> SE -> 1x1 project -> residual add.

    Args:
        x: Input tensor.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Channel expansion ratio. Defaults to ``4``.
        se_ratio: Squeeze-and-Excitation reduction ratio. Defaults to ``0.0625``.
        stride: Spatial stride for the depthwise convolution. Defaults to ``1``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index (``-1`` or ``1``).
        prefix: String prefix for layer names.

    Returns:
        Output tensor.
    """
    expanded = out_channels * expand_ratio
    se_reduced = max(1, int(expanded * se_ratio))

    shortcut = x
    if stride > 1:
        shortcut = layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding="same",
            data_format=data_format,
            name=prefix + "conv_shortcut_pool",
        )(shortcut)
    if in_channels != out_channels:
        shortcut = layers.Conv2D(
            out_channels,
            1,
            use_bias=True,
            data_format=data_format,
            name=prefix + "conv_shortcut_expand",
        )(shortcut)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_pre_norm",
    )(x)
    x = layers.Conv2D(
        expanded,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "conv_conv1_1x1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_norm1",
    )(x)
    x = layers.Activation(maxvit_gelu_approximate, name=prefix + "conv_act1")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=prefix + "conv_conv2_kxk",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.9,
        name=prefix + "conv_norm2",
    )(x)
    x = layers.Activation(maxvit_gelu_approximate, name=prefix + "conv_act2")(x)

    # Squeeze-and-Excitation
    x_se = layers.GlobalAveragePooling2D(
        data_format=data_format,
        keepdims=True,
        name=prefix + "se_pool",
    )(x)
    x_se = layers.Conv2D(
        se_reduced,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "se_fc1",
    )(x_se)
    x_se = layers.Activation("silu", name=prefix + "se_act")(x_se)
    x_se = layers.Conv2D(
        expanded,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "se_fc2",
    )(x_se)
    x_se = layers.Activation("sigmoid", name=prefix + "se_gate")(x_se)
    x = layers.Multiply()([x, x_se])

    x = layers.Conv2D(
        out_channels,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "conv_conv3_1x1",
    )(x)
    x = layers.Add()([x, shortcut])
    return x


def maxvit_partition_attn_block(
    x,
    dim,
    num_heads,
    window_size,
    img_size,
    partition_type="block",
    mlp_ratio=4.0,
    data_format="channels_last",
    prefix="",
):
    """Partition-based attention block with MLP for MaxViT.

    Applies multi-head self-attention within spatial partitions (local windows
    or dilated grids), followed by an MLP branch. Both branches use pre-norm
    residual connections.

    The partition layers convert to channels-last for attention and convert
    back to the original ``data_format`` after.

    Args:
        x: Input tensor.
        dim: Channel dimension.
        num_heads: Number of attention heads.
        window_size: Window / grid size for partitioning.
        img_size: Tuple ``(H, W)`` of the current spatial dimensions.
        partition_type: ``"block"`` or ``"grid"``. Defaults to ``"block"``.
        mlp_ratio: MLP hidden-dimension expansion ratio. Defaults to ``4.0``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor of same shape as input.
    """
    part_size = (
        (window_size, window_size) if isinstance(window_size, int) else window_size
    )

    if partition_type == "block":
        partitioned = MaxViTWindowPartition(
            part_size,
            data_format=data_format,
            name=prefix + "win_part",
        )(x)
    else:
        partitioned = MaxViTGridPartition(
            part_size,
            data_format=data_format,
            name=prefix + "grid_part",
        )(x)

    y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm1")(partitioned)
    y = MaxViTAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=part_size,
        prefix=prefix,
    )(y)

    if partition_type == "block":
        y = MaxViTWindowReverse(
            part_size,
            img_size,
            data_format=data_format,
            name=prefix + "win_rev",
        )(y)
    else:
        y = MaxViTGridReverse(
            part_size,
            img_size,
            data_format=data_format,
            name=prefix + "grid_rev",
        )(y)

    x = layers.Add()([x, y])

    residual = x
    if data_format == "channels_first":
        y = layers.Permute((2, 3, 1), name=prefix + "mlp_to_cl")(x)
        y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm2")(y)
        y = layers.Dense(int(dim * mlp_ratio), use_bias=True, name=prefix + "mlp_fc1")(
            y
        )
        y = layers.Activation(maxvit_gelu_approximate, name=prefix + "mlp_gelu")(y)
        y = layers.Dense(dim, use_bias=True, name=prefix + "mlp_fc2")(y)
        y = layers.Permute((3, 1, 2), name=prefix + "mlp_to_cf")(y)
    else:
        y = layers.LayerNormalization(epsilon=1e-5, name=prefix + "norm2")(x)
        y = layers.Dense(int(dim * mlp_ratio), use_bias=True, name=prefix + "mlp_fc1")(
            y
        )
        y = layers.Activation(maxvit_gelu_approximate, name=prefix + "mlp_gelu")(y)
        y = layers.Dense(dim, use_bias=True, name=prefix + "mlp_fc2")(y)

    x = layers.Add()([residual, y])
    return x


def maxvit_backbone_feature(
    inputs,
    *,
    stem_width,
    depths,
    embed_dim,
    num_heads,
    window_size,
    mlp_ratio,
    se_ratio,
    expand_ratio,
    data_format,
    channels_axis,
    return_stages=False,
):
    """MaxViT 2-conv stem + 4 stages of (MBConv + block-attn + grid-attn) blocks.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        stem_width: Output channel count of both stem convs.
        depths: Number of blocks per stage (length 4).
        embed_dim: Output channel count per stage (length 4).
        num_heads: Number of attention heads per stage (length 4).
        window_size: Side length used by both window and grid partitions.
        mlp_ratio: Hidden-dim expansion ratio inside the attention-block MLPs.
        se_ratio: Squeeze-and-Excitation reduction ratio for MBConv blocks.
        expand_ratio: Channel expansion ratio for MBConv blocks.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 4 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``embed_dim[-1]`` channels at spatial
        resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 4 per-stage feature maps.
    """
    if data_format == "channels_first":
        H, W = inputs.shape[2], inputs.shape[3]
    else:
        H, W = inputs.shape[1], inputs.shape[2]

    x = layers.Conv2D(
        stem_width,
        3,
        strides=2,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name="stem_conv1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-3, momentum=0.9, name="stem_norm1"
    )(x)
    x = layers.Activation(maxvit_gelu_approximate, name="stem_act1")(x)
    x = layers.Conv2D(
        stem_width,
        3,
        strides=1,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name="stem_conv2",
    )(x)

    cur_H = H // 2
    cur_W = W // 2

    in_ch = stem_width
    stages = []
    for stage_idx in range(len(depths)):
        out_ch = embed_dim[stage_idx]
        for block_idx in range(depths[stage_idx]):
            prefix = f"stages_{stage_idx}_blocks_{block_idx}_"
            stride = 2 if block_idx == 0 else 1
            block_in_ch = in_ch if block_idx == 0 else out_ch

            x = maxvit_mbconv_block(
                x,
                in_channels=block_in_ch,
                out_channels=out_ch,
                expand_ratio=expand_ratio,
                se_ratio=se_ratio,
                stride=stride,
                data_format=data_format,
                channels_axis=channels_axis,
                prefix=prefix,
            )
            if stride == 2:
                cur_H //= 2
                cur_W //= 2

            x = maxvit_partition_attn_block(
                x,
                dim=out_ch,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                img_size=(cur_H, cur_W),
                partition_type="block",
                mlp_ratio=mlp_ratio,
                data_format=data_format,
                prefix=prefix + "attn_block_",
            )
            x = maxvit_partition_attn_block(
                x,
                dim=out_ch,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                img_size=(cur_H, cur_W),
                partition_type="grid",
                mlp_ratio=mlp_ratio,
                data_format=data_format,
                prefix=prefix + "attn_grid_",
            )

        stages.append(x)
        in_ch = out_ch

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTModel(BaseModel):
    """Instantiates the MaxViT backbone.

    MaxViT introduces multi-axis attention by interleaving three block
    types in each of its 4 hierarchical stages: an MBConv block (with
    Squeeze-and-Excitation) for local convolutional mixing, a
    window-based self-attention block over fixed-size local windows for
    local receptive field, and a grid-based self-attention block over
    dilated, regularly-spaced grids for a global receptive field. By
    combining all three at every stage, MaxViT achieves both local and
    global attention at linear complexity while remaining
    fully-convolutional in spatial shape.

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first), unpooled and head-free.
    :class:`MaxViTImageClassify` composes this model and appends the head.

    References:
    - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            4 per-stage feature maps. Defaults to `False`.
        stem_width: Integer, output channel count of both stem convs.
            Defaults to `64`.
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(2, 2, 5, 2)`.
        embed_dim: Tuple of integers, output channel count per stage
            (length 4). Defaults to `(64, 128, 256, 512)`.
        num_heads: Tuple of integers, number of attention heads per
            stage (length 4). Defaults to `(2, 4, 8, 16)`.
        window_size: Integer, side length used by both window and grid
            partitions. Defaults to `7`.
        mlp_ratio: Float, hidden-dim expansion ratio inside the
            attention-block MLPs. Defaults to `4.0`.
        se_ratio: Float, Squeeze-and-Excitation reduction ratio for
            MBConv blocks. Defaults to `0.0625`.
        expand_ratio: Integer, channel expansion ratio for MBConv
            blocks. Defaults to `4`.
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
        name: String, the name of the model. Defaults to `"MaxViTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MaxViTConfig
    HUB_REPO_SIBLINGS = MAXVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MaxViTImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MaxViTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_maxvit_timm_to_keras import transfer_maxvit_weights

        transfer_maxvit_weights(keras_model, state_dict)

    def __init__(
        self,
        stem_width=64,
        depths=(2, 2, 5, 2),
        embed_dim=(64, 128, 256, 512),
        num_heads=(2, 4, 8, 16),
        window_size=7,
        mlp_ratio=4.0,
        se_ratio=0.0625,
        expand_ratio=4,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="MaxViTModel",
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
        x = maxvit_backbone_feature(
            x,
            stem_width=stem_width,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            se_ratio=se_ratio,
            expand_ratio=expand_ratio,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.stem_width = stem_width
        self.depths = list(depths)
        self.embed_dim = list(embed_dim)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_width": self.stem_width,
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "se_ratio": self.se_ratio,
                "expand_ratio": self.expand_ratio,
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
class MaxViTImageClassify(BaseModel):
    """Instantiates the MaxViT classifier.

    This classifier wraps a :class:`MaxViTModel` backbone and attaches a
    GlobalAveragePooling2D + LayerNorm + Dense + tanh + Dense head to
    produce ``num_classes`` class logits. All architectural parameters
    are forwarded to the underlying :class:`MaxViTModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)

    Args:
        stem_width: Integer, output channel count of both stem convs.
            Defaults to `64`.
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(2, 2, 5, 2)`.
        embed_dim: Tuple of integers, output channel count per stage
            (length 4). Defaults to `(64, 128, 256, 512)`.
        num_heads: Tuple of integers, number of attention heads per
            stage (length 4). Defaults to `(2, 4, 8, 16)`.
        window_size: Integer, side length used by both window and grid
            partitions. Defaults to `7`.
        mlp_ratio: Float, hidden-dim expansion ratio inside the
            attention-block MLPs. Defaults to `4.0`.
        se_ratio: Float, Squeeze-and-Excitation reduction ratio for
            MBConv blocks. Defaults to `0.0625`.
        expand_ratio: Integer, channel expansion ratio for MBConv
            blocks. Defaults to `4`.
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
            named `f"{name}_backbone"`. Defaults to `"MaxViTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MaxViTConfig
    HUB_REPO_SIBLINGS = MAXVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_maxvit_timm_to_keras import transfer_maxvit_weights

        transfer_maxvit_weights(keras_model, state_dict)

    def __init__(
        self,
        stem_width=64,
        depths=(2, 2, 5, 2),
        embed_dim=(64, 128, 256, 512),
        num_heads=(2, 4, 8, 16),
        window_size=7,
        mlp_ratio=4.0,
        se_ratio=0.0625,
        expand_ratio=4,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MaxViTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = MaxViTModel(
            stem_width=stem_width,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            se_ratio=se_ratio,
            expand_ratio=expand_ratio,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(
            data_format=data_format, name="head_global_pool"
        )(backbone.output)
        x = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="head_norm")(x)
        x = layers.Dense(embed_dim[-1], use_bias=True, name="head_pre_logits_fc")(x)
        x = layers.Activation("tanh", name="head_pre_logits_act")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="head_fc"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.stem_width = stem_width
        self.depths = list(depths)
        self.embed_dim = list(embed_dim)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
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
                "stem_width": self.stem_width,
                "depths": self.depths,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "se_ratio": self.se_ratio,
                "expand_ratio": self.expand_ratio,
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
