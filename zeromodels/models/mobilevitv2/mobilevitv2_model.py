import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.mobilevit.mobilevit_layers import (
    MobileViTImageToPatchesLayer,
    MobileViTPatchesToImageLayer,
)
from zeromodels.models.mobilevit.mobilevit_model import mobilevit_aspp_head
from zeromodels.utils import standardize_input_shape

from .mobilevitv2_config import MobileViTV2Config

# The backbone (MobileViTV2Model) and classifier (MobileViTV2ImageClassify) share the
# classification variant's repo, whose zm_config.json declares
# MobileViTV2ImageClassify.
MOBILEVITV2_HUB_SIBLINGS = frozenset({"MobileViTV2Model", "MobileViTV2ImageClassify"})

# Each *_deeplabv3 repo carries its own zm_config.json declaring the segment model.
MOBILEVITV2_SEGMENT_HUB_SIBLINGS = frozenset({"MobileViTV2SemanticSegment"})


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Args:
        v: Channel count to round (may be float-valued from a width multiplier).
        divisor: Multiple to snap to. Defaults to ``8``.
        min_value: Floor for the rounded value; defaults to ``divisor`` when ``None``.
        round_limit: If the rounded value is less than ``round_limit * v``, bump it
            up by one more ``divisor`` step. Defaults to ``0.9``.

    Returns:
        Integer channel count that is a multiple of ``divisor`` and at least
        ``min_value``.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    inputs,
    filters,
    channels_axis,
    data_format,
    strides=1,
    expansion_ratio=2.0,
    dilation=1,
    name="inverted_residual_block",
):
    """MobileViTV2 inverted residual (MBConv) block.

    Args:
        inputs: Input feature map.
        filters: Output channel count.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: Spatial stride for the depthwise conv. Defaults to ``1``.
        expansion_ratio: Channel expansion factor for the hidden layer.
            Defaults to ``2.0``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``filters`` channels and spatial size reduced by
        ``strides``.
    """
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        make_divisible(inputs.shape[channels_axis] * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_1")(x)

    if strides > 1 and dilation == 1:
        x = layers.ZeroPadding2D(
            padding=1,
            data_format=data_format,
            name=f"{name}_ir_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_2")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_3",
    )(x)

    if residual_connection:
        x = layers.Add(name=f"{name}_ir_add")([x, inputs])

    return x


def linear_self_attention(
    inputs, dim, data_format, use_bias=True, name="linear_self_attention"
):
    """Separable (linear) self-attention block used by MobileViTV2.

    Args:
        inputs: Input tensor in patch-folded form.
        dim: Channel dimension for the query/key/value projections.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        use_bias: Whether the projection convs use bias. Defaults to ``True``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``dim`` channels and the same spatial shape as
        ``inputs``.
    """
    num_patch_axis = -2 if data_format == "channels_last" else -1

    x = layers.Conv2D(1 + (2 * dim), 1, use_bias=use_bias, name=f"{name}_attn_conv_1")(
        inputs
    )

    if data_format == "channels_last":
        query = x[..., :1]
        key = x[..., 1 : dim + 1]
        value = x[..., dim + 1 :]
    else:
        query = x[:, :1]
        key = x[:, 1 : dim + 1]
        value = x[:, dim + 1 :]

    context_scores = layers.Softmax(axis=num_patch_axis, name=f"{name}_attn_softmax")(
        query
    )
    context_vector = layers.Multiply(name=f"{name}_attn_multiply_1")(
        [key, context_scores]
    )
    context_vector = keras.ops.sum(context_vector, axis=num_patch_axis, keepdims=True)

    out = layers.ReLU(name=f"{name}_attn_relu")(value)
    out = layers.Multiply(name=f"{name}_attn_multiply_2")([out, context_vector])
    out = layers.Conv2D(dim, 1, use_bias=use_bias, name=f"{name}_attn_conv_2")(out)

    return out


def mobilevitv2_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    kernel_size=3,
    expansion_ratio=2.0,
    transformer_dim=None,
    transformer_depth=2,
    patch_size=2,
    name="mobilevitv2_block",
):
    """MobileViTV2 transformer fusion block with linear self-attention.

    Args:
        inputs: Input feature map.
        block_dims: Output channel count of the block.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        kernel_size: Kernel size of the leading depthwise conv. Defaults to ``3``.
        expansion_ratio: Multiplier used to derive ``transformer_dim`` when it is
            not given. Defaults to ``2.0``.
        transformer_dim: Channel dimension inside the transformer. If ``None``,
            computed as ``make_divisible(inputs.channels * expansion_ratio)``.
        transformer_depth: Number of stacked transformer encoder blocks.
            Defaults to ``2``.
        patch_size: Side length of square patches unfolded for self-attention.
            Defaults to ``2``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``block_dims`` channels and the same spatial size as
        ``inputs``.
    """
    transformer_dim = transformer_dim or make_divisible(
        inputs.shape[channels_axis] * expansion_ratio
    )

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_dwconv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mc2_act_1")(x)

    x = layers.Conv2D(
        transformer_dim,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_conv_1",
    )(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = MobileViTImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    for i in range(transformer_depth):
        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_1",
        )(x)
        x = linear_self_attention(
            x,
            transformer_dim,
            data_format,
            use_bias=True,
            name=f"{name}_transformer_{i}",
        )
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual, x])

        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_2",
        )(x)
        mlp_hidden_dim = int(transformer_dim * 2.0)

        x = layers.Conv2D(
            mlp_hidden_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Conv2D(
            transformer_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_2",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual, x])

    x = layers.GroupNormalization(
        1,
        axis=channels_axis,
        epsilon=1e-5,
        name=f"{name}_groupnorm",
    )(x)

    fold_layer = MobileViTPatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_proj_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_proj_batchnorm",
    )(x)

    return x


def mobilevitv2_backbone_feature(
    inputs,
    *,
    multiplier,
    data_format,
    channels_axis,
    output_stride=32,
    return_stages=False,
):
    """MobileViTV2 stem + 5 stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        multiplier: Width multiplier applied to every stage's channel count.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 5 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``int(512 * multiplier)`` channels at
        spatial resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 5 per-stage feature maps.
    """
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(inputs)
    x = layers.Conv2D(
        int(32 * multiplier),
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation("swish", name="stem_act")(x)

    stage_strides_default = [1, 2, 2, 2, 2]
    stage_dilations_default = [1, 1, 1, 1, 1]
    if output_stride == 16:
        stage_strides_default[4] = 1
        stage_dilations_default[4] = 2
    elif output_stride == 8:
        stage_strides_default[3] = 1
        stage_strides_default[4] = 1
        stage_dilations_default[3] = 2
        stage_dilations_default[4] = 4
    elif output_stride != 32:
        raise ValueError(f"output_stride must be 8, 16, or 32, got {output_stride}")

    stages = []
    for stage in range(5):
        channels = int(([64, 128, 256, 384, 512][stage]) * multiplier)

        # For atrous output strides the last stage(s) keep stride 1 and the
        # downsampling inverted-residual carries the dilation (the reference applies
        # ``dilation // 2`` to it); the MobileViTV2 block's local conv is never
        # dilated.
        stage_dilation = stage_dilations_default[stage]
        down_dilation = stage_dilation // 2 if stage_dilation > 1 else 1
        x = inverted_residual_block(
            x,
            channels,
            channels_axis,
            data_format,
            strides=stage_strides_default[stage],
            expansion_ratio=2.0,
            dilation=down_dilation,
            name=f"stages_{stage}_0",
        )

        if stage <= 1:
            if stage == 1:
                x = inverted_residual_block(
                    x,
                    channels,
                    channels_axis,
                    data_format,
                    strides=1,
                    expansion_ratio=2.0,
                    name=f"stages_{stage}_1",
                )
        else:
            x = mobilevitv2_block(
                x,
                channels,
                channels_axis,
                data_format,
                kernel_size=3,
                expansion_ratio=0.5,
                transformer_depth=[2, 4, 3][stage - 2],
                patch_size=2,
                name=f"stages_{stage}_1",
            )

        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTV2Model(BaseModel):
    """Instantiates the MobileViTV2 backbone.

    MobileViTV2 builds on MobileViT by replacing the standard quadratic
    multi-head self-attention with a separable, linear-complexity
    self-attention that scales linearly in the number of patches. The
    transformer block is also simplified to a more lightweight design
    that uses GroupNorm + Conv1x1 projections in place of LayerNorm +
    Dense, making it more efficient for mobile inference while keeping
    the same 5-stage hierarchical layout.

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first), unpooled and head-free.
    :class:`MobileViTV2ImageClassify` composes this model and appends GAP +
    Dense.

    References:
    - [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            5 per-stage feature maps. Defaults to `False`.
        multiplier: Float, width multiplier applied to every stage's
            channel count. Defaults to `1.0`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `256`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'`, `'dpn'`,
            `'clip'`, `'zero_to_one'` (default), or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model.
            Defaults to `"MobileViTV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTV2Config
    HUB_REPO_SIBLINGS = MOBILEVITV2_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevitv2"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MobileViTV2ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MobileViTV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "multiplier": float(hf_config.get("width_multiplier", 1.0)),
            "image_size": hf_config["image_size"],
            "output_stride": hf_config.get("output_stride", 32),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevitv2_hf_to_keras import transfer_mobilevitv2_weights

        transfer_mobilevitv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=256,
        output_stride=32,
        input_tensor=None,
        as_backbone=False,
        name="MobileViTV2Model",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        # Shared MobileViTV2Config also carries classifier / segmentation-head fields.
        for k in (
            "num_classes",
            "classifier_activation",
            "atrous_rates",
            "aspp_out_channels",
            "aspp_dropout_prob",
            "timm_id",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = img_input
        x = mobilevitv2_backbone_feature(
            x,
            multiplier=multiplier,
            data_format=data_format,
            channels_axis=channels_axis,
            output_stride=output_stride,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.multiplier = multiplier
        self.image_size = image_size
        self.output_stride = output_stride
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
                "image_size": self.image_size,
                "output_stride": self.output_stride,
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
class MobileViTV2ImageClassify(BaseModel):
    """Instantiates the MobileViTV2 classifier.

    This classifier wraps a :class:`MobileViTV2Model` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`MobileViTV2Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)

    Args:
        multiplier: Float, width multiplier applied to every stage's
            channel count. Defaults to `1.0`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `256`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'`, `'dpn'`,
            `'clip'`, `'zero_to_one'` (default), or `'minus_one_to_one'`.
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
            named `f"{name}_backbone"`.
            Defaults to `"MobileViTV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTV2Config
    HUB_REPO_SIBLINGS = MOBILEVITV2_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevitv2"

    @classmethod
    def config_from_hf(cls, hf_config):
        num_classes = hf_config.get("num_labels")
        if num_classes is None and hf_config.get("id2label"):
            num_classes = len(hf_config["id2label"])
        return {
            "multiplier": float(hf_config.get("width_multiplier", 1.0)),
            "image_size": hf_config["image_size"],
            "num_classes": num_classes if num_classes is not None else 1000,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevitv2_hf_to_keras import transfer_mobilevitv2_weights

        transfer_mobilevitv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=256,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileViTV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        # Shared MobileViTV2Config also carries segmentation-head fields (and
        # output_stride, which the classifier always runs at 32); drop them.
        for k in (
            "output_stride",
            "atrous_rates",
            "aspp_out_channels",
            "aspp_dropout_prob",
            "timm_id",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

        backbone = MobileViTV2Model(
            multiplier=multiplier,
            image_size=image_size,
            output_stride=32,
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

        self.multiplier = multiplier
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
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


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTV2SemanticSegment(BaseModel):
    """MobileViTV2 + DeepLabV3 semantic segmentation head.

    Composes :class:`MobileViTV2Model` with ASPP and a 1x1 classifier conv to
    produce per-pixel class logits at the segmentation feature resolution
    (input ``H / output_stride``). The backbone runs with
    ``output_stride=16`` and atrous convolutions in the last stage to
    preserve spatial detail.

    References:
    - [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)
    - [DeepLabV3](https://arxiv.org/abs/1706.05587)
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTV2Config
    HUB_REPO_SIBLINGS = MOBILEVITV2_SEGMENT_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevitv2"

    @classmethod
    def config_from_hf(cls, hf_config):
        num_classes = hf_config.get("num_labels")
        if num_classes is None and hf_config.get("id2label"):
            num_classes = len(hf_config["id2label"])
        return {
            "multiplier": float(hf_config.get("width_multiplier", 1.0)),
            "image_size": hf_config["image_size"],
            "output_stride": hf_config.get("output_stride", 16),
            "atrous_rates": list(hf_config.get("atrous_rates", [6, 12, 18])),
            "aspp_out_channels": hf_config.get("aspp_out_channels", 512),
            "aspp_dropout_prob": hf_config.get("aspp_dropout_prob", 0.1),
            "num_classes": num_classes if num_classes is not None else 21,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevitv2_hf_to_keras import transfer_mobilevitv2_weights

        transfer_mobilevitv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=512,
        output_stride=16,
        atrous_rates: tuple = (6, 12, 18),
        aspp_out_channels=512,
        aspp_dropout_prob=0.1,
        input_tensor=None,
        num_classes=21,
        name="MobileViTV2SemanticSegment",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        backbone = MobileViTV2Model(
            multiplier=multiplier,
            image_size=image_size,
            output_stride=output_stride,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        features = backbone.output
        out = mobilevit_aspp_head(
            features,
            aspp_out_channels=aspp_out_channels,
            atrous_rates=atrous_rates,
            aspp_dropout_prob=aspp_dropout_prob,
            num_classes=num_classes,
            data_format=data_format,
            channels_axis=channels_axis,
            name="seg",
        )

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.multiplier = multiplier
        self.image_size = backbone.image_size
        self.output_stride = output_stride
        self.atrous_rates = atrous_rates
        self.aspp_out_channels = aspp_out_channels
        self.aspp_dropout_prob = aspp_dropout_prob
        self.input_tensor = input_tensor
        self.num_classes = num_classes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
                "image_size": self.image_size,
                "output_stride": self.output_stride,
                "atrous_rates": self.atrous_rates,
                "aspp_out_channels": self.aspp_out_channels,
                "aspp_dropout_prob": self.aspp_dropout_prob,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
