import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.mobilevit.mobilevit_layers import (
    MobileViTImageToPatchesLayer,
    MobileViTMultiHeadSelfAttention,
    MobileViTPatchesToImageLayer,
)
from zeromodels.utils import standardize_input_shape

from .mobilevit_config import MobileViTConfig

# The backbone (MobileViTModel) and classifier (MobileViTImageClassify) share the
# classification variant's repo, whose zm_config.json declares MobileViTImageClassify.
MOBILEVIT_HUB_SIBLINGS = frozenset({"MobileViTModel", "MobileViTImageClassify"})

# Each *_deeplabv3 repo carries its own zm_config.json declaring the segment model.
MOBILEVIT_SEGMENT_HUB_SIBLINGS = frozenset({"MobileViTSemanticSegment"})


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
    expansion_ratio=1.0,
    dilation=1,
    name: str = "inverted_residual_block",
):
    """Inverted residual (MBConv) block as used in MobileNetV2 / MobileViT.

    Args:
        inputs: Input feature map.
        filters: Output channel count.
        channels_axis: Channel axis index (``-1`` for channels-last, ``-3`` for
            channels-first).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: Spatial stride for the depthwise conv. Defaults to ``1``.
        expansion_ratio: Channel expansion factor for the hidden layer.
            Defaults to ``1.0``.
        dilation: Dilation rate for the depthwise conv. When ``>1``, the
            spatial stride is effectively replaced with atrous convolution
            (DeepLab-style output_stride reduction). Defaults to ``1``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``filters`` channels and spatial size reduced by
        ``strides``.
    """
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        filters=make_divisible(inputs.shape[channels_axis] * expansion_ratio),
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
            padding=(1, 1),
            data_format=data_format,
            name=f"{name}_zeropadding",
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
        x = layers.Add(name=f"{name}_add")([x, inputs])

    return x


def mobilevit_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    attention_dims=None,
    num_attention_blocks=2,
    patch_size=8,
    name="mobilevit_transformer_block",
):
    """MobileViT transformer fusion block (local conv + global self-attention).

    Args:
        inputs: Input feature map.
        block_dims: Output channel count of the block.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        attention_dims: Channel dimension used inside the transformer. If
            ``None``, defaults to ``make_divisible(inputs.shape[channels_axis])``.
        num_attention_blocks: Number of stacked transformer encoder blocks.
            Defaults to ``2``.
        patch_size: Side length of the square patches unfolded for self-attention.
            Defaults to ``8``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``block_dims`` channels and the same spatial size as
        ``inputs``.
    """
    if attention_dims is None:
        attention_dims = make_divisible(inputs.shape[channels_axis])

    x = inputs

    x = layers.Conv2D(
        inputs.shape[channels_axis],
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_mv_batchnorm_1"
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_1")(x)

    x = layers.Conv2D(attention_dims, 1, use_bias=False, name=f"{name}_mv_conv_2")(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = MobileViTImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)

    for i in range(num_attention_blocks):
        residual_1 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_1"
        )(x)
        x = MobileViTMultiHeadSelfAttention(
            attention_dims,
            num_heads=4,
            qkv_bias=True,
            block_prefix=f"{name}_transformer_{i}",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual_1, x])

        residual_2 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_2"
        )(x)
        mlp_hidden_dim = int(attention_dims * 2.0)
        x = layers.Dense(
            mlp_hidden_dim,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_1")(x)
        x = layers.Dense(
            attention_dims,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc2",
        )(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_2")(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual_2, x])

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_layernorm")(x)

    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)

    fold_layer = MobileViTPatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_3",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_2")(x)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")([inputs, x])

    x = layers.Conv2D(
        block_dims,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_4",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_3",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_3")(x)

    return x


def mobilevit_backbone_feature(
    inputs,
    *,
    initial_dims,
    block_dims,
    expansion_ratio,
    attention_dims,
    data_format,
    channels_axis,
    output_stride=32,
    return_stages=False,
):
    """MobileViT stem + 5 stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        initial_dims: Stem output channel count.
        block_dims: Per-stage output channel counts (length 5).
        expansion_ratio: Per-stage MBConv expansion ratios (length 5).
        attention_dims: Per-stage transformer attention dims (length 5). Entries
            may be ``None`` for stages that don't use a transformer block.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 5 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``block_dims[-1]`` channels at spatial
        resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 5 per-stage feature maps.
    """
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=data_format)(inputs)
    x = layers.Conv2D(
        initial_dims,
        kernel_size=3,
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
    for i in range(5):
        # For atrous output strides the last stage(s) keep stride 1 and the
        # downsampling inverted-residual carries the dilation (the reference applies
        # ``dilation // 2`` to the downsampling depthwise conv; the MobileViT
        # block's local conv is never dilated).
        stage_dilation = stage_dilations_default[i]
        down_dilation = stage_dilation // 2 if stage_dilation > 1 else 1
        x = inverted_residual_block(
            x,
            filters=block_dims[i],
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stage_strides_default[i],
            expansion_ratio=expansion_ratio[i],
            dilation=down_dilation,
            name=f"stages_{i}_0",
        )

        if i == 1:
            x = inverted_residual_block(
                x,
                filters=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                strides=1,
                expansion_ratio=expansion_ratio[i],
                name=f"stages_{i}_1",
            )
            x = inverted_residual_block(
                x,
                filters=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                strides=1,
                expansion_ratio=expansion_ratio[i],
                name=f"stages_{i}_2",
            )

        if i >= 2:
            x = mobilevit_block(
                x,
                block_dims=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                attention_dims=attention_dims[i],
                num_attention_blocks=2 if i == 2 else 4 if i == 3 else 3,
                patch_size=2,
                name=f"stages_{i}_1",
            )

        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTModel(BaseModel):
    """Instantiates the MobileViT backbone.

    MobileViT is a hybrid CNN-Transformer backbone designed for mobile
    inference: it interleaves MobileNetV2 inverted-residual (MBConv)
    blocks with MobileViT blocks that fold local self-attention over
    fixed-size patches, mixing convolutional locality with transformer
    global context. The network is organized as 5 stages of progressively
    reduced spatial resolution.

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first) with ``block_dims[-1]`` channels at
    spatial resolution ``H/32``. :class:`MobileViTImageClassify` composes this
    model and appends the final 1x1 conv + GAP + Dense head.

    References:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            5 per-stage feature maps. Defaults to `False`.
        initial_dims: Integer, stem output channel count.
            Defaults to `16`.
        head_dims: Integer, channel count of the 1x1 final conv used by
            the classifier head. Defaults to `640`.
        block_dims: List of integers, per-stage output channel counts
            (length 5). Defaults to `[32, 64, 96, 128, 160]`.
        expansion_ratio: List of floats, per-stage MBConv expansion
            ratios (length 5). Defaults to `[4.0, 4.0, 4.0, 4.0, 4.0]`.
        attention_dims: List, per-stage transformer attention dims
            (length 5). Entries may be `None` for stages without a
            transformer block. Defaults to `[None, None, 144, 192, 240]`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `256`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model. Defaults to `"MobileViTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTConfig
    HUB_REPO_SIBLINGS = MOBILEVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevit"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MobileViTImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MobileViTImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        neck = hf_config["neck_hidden_sizes"]
        hidden_sizes = hf_config["hidden_sizes"]
        return {
            "initial_dims": neck[0],
            "block_dims": list(neck[1:6]),
            "head_dims": neck[6],
            "attention_dims": [None, None] + list(hidden_sizes),
            "expansion_ratio": [float(hf_config["expand_ratio"])] * 5,
            "image_size": hf_config["image_size"],
            "output_stride": hf_config.get("output_stride", 32),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevit_hf_to_keras import transfer_mobilevit_weights

        transfer_mobilevit_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: tuple = (32, 64, 96, 128, 160),
        expansion_ratio: tuple = (4.0, 4.0, 4.0, 4.0, 4.0),
        attention_dims: tuple = (None, None, 144, 192, 240),
        image_size=256,
        output_stride=32,
        input_tensor=None,
        as_backbone=False,
        name="MobileViTModel",
        **kwargs,
    ):
        # Shared MobileViTConfig also carries classifier / segmentation-head fields.
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
        x = mobilevit_backbone_feature(
            x,
            initial_dims=initial_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            data_format=data_format,
            channels_axis=channels_axis,
            output_stride=output_stride,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
        self.image_size = image_size
        self.output_stride = output_stride
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
class MobileViTImageClassify(BaseModel):
    """Instantiates the MobileViT classifier.

    This classifier wraps a :class:`MobileViTModel` backbone and attaches
    a 1x1 final conv + GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`MobileViTModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

    Args:
        initial_dims: Integer, stem output channel count.
            Defaults to `16`.
        head_dims: Integer, channel count of the 1x1 final conv used by
            the classifier head. Defaults to `640`.
        block_dims: List of integers, per-stage output channel counts
            (length 5). Defaults to `[32, 64, 96, 128, 160]`.
        expansion_ratio: List of floats, per-stage MBConv expansion
            ratios (length 5). Defaults to `[4.0, 4.0, 4.0, 4.0, 4.0]`.
        attention_dims: List, per-stage transformer attention dims
            (length 5). Entries may be `None` for stages without a
            transformer block. Defaults to `[None, None, 144, 192, 240]`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `256`.
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
            named `f"{name}_backbone"`. Defaults to `"MobileViTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTConfig
    HUB_REPO_SIBLINGS = MOBILEVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevit"

    @classmethod
    def config_from_hf(cls, hf_config):
        neck = hf_config["neck_hidden_sizes"]
        hidden_sizes = hf_config["hidden_sizes"]
        num_classes = hf_config.get("num_labels")
        if num_classes is None and hf_config.get("id2label"):
            num_classes = len(hf_config["id2label"])
        return {
            "initial_dims": neck[0],
            "block_dims": list(neck[1:6]),
            "head_dims": neck[6],
            "attention_dims": [None, None] + list(hidden_sizes),
            "expansion_ratio": [float(hf_config["expand_ratio"])] * 5,
            "image_size": hf_config["image_size"],
            "num_classes": num_classes if num_classes is not None else 1000,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevit_hf_to_keras import transfer_mobilevit_weights

        transfer_mobilevit_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: tuple = (32, 64, 96, 128, 160),
        expansion_ratio: tuple = (4.0, 4.0, 4.0, 4.0, 4.0),
        attention_dims: tuple = (None, None, 144, 192, 240),
        image_size=256,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileViTImageClassify",
        **kwargs,
    ):
        # Shared MobileViTConfig also carries segmentation-head fields (and
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
        channels_axis = -1 if data_format == "channels_last" else -3

        backbone = MobileViTModel(
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            image_size=image_size,
            output_stride=32,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.Conv2D(
            head_dims,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="final_conv",
        )(backbone.output)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="final_batchnorm",
        )(x)
        x = layers.Activation("swish", name="final_act")(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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


def mobilevit_aspp_head(
    inputs,
    *,
    aspp_out_channels,
    atrous_rates,
    aspp_dropout_prob,
    num_classes,
    data_format,
    channels_axis,
    name="seg",
):
    """DeepLabV3 ASPP + 1x1 classifier head for MobileViT segmentation."""
    branches = []

    # Branch 0: 1x1 conv
    b = layers.Conv2D(
        aspp_out_channels,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_aspp_conv1_1x1_conv",
    )(inputs)
    b = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_aspp_conv1_1x1_batchnorm",
    )(b)
    b = layers.Activation("relu", name=f"{name}_aspp_conv1_1x1_act")(b)
    branches.append(b)

    # Branches 1-3: 3x3 dilated convs
    for i, rate in enumerate(atrous_rates, start=2):
        b = layers.Conv2D(
            aspp_out_channels,
            kernel_size=3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_aspp_conv{i}_3x3_conv",
        )(inputs)
        b = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name=f"{name}_aspp_conv{i}_3x3_batchnorm",
        )(b)
        b = layers.Activation("relu", name=f"{name}_aspp_conv{i}_3x3_act")(b)
        branches.append(b)

    # Branch 4: image pooling + 1x1 conv + upsample
    if data_format == "channels_first":
        h_axis, w_axis = 2, 3
    else:
        h_axis, w_axis = 1, 2

    pooled = layers.GlobalAveragePooling2D(
        data_format=data_format, keepdims=True, name=f"{name}_aspp_pool_gap"
    )(inputs)
    pooled = layers.Conv2D(
        aspp_out_channels,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_aspp_pool_1x1_conv",
    )(pooled)
    pooled = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_aspp_pool_1x1_batchnorm",
    )(pooled)
    pooled = layers.Activation("relu", name=f"{name}_aspp_pool_1x1_act")(pooled)

    def resize_pooled(args):
        feat, ref = args
        size = (keras.ops.shape(ref)[h_axis], keras.ops.shape(ref)[w_axis])
        feat = keras.ops.image.resize(
            feat, size, interpolation="bilinear", data_format=data_format
        )
        return feat

    pooled = layers.Lambda(resize_pooled, name=f"{name}_aspp_pool_resize")(
        [pooled, inputs]
    )
    branches.append(pooled)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_aspp_concat")(branches)
    x = layers.Conv2D(
        aspp_out_channels,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_aspp_project_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_aspp_project_batchnorm",
    )(x)
    x = layers.Activation("relu", name=f"{name}_aspp_project_act")(x)
    x = layers.Dropout(aspp_dropout_prob, name=f"{name}_aspp_dropout")(x)
    x = layers.Conv2D(
        num_classes,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name="seg_classifier_conv",
    )(x)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTSemanticSegment(BaseModel):
    """MobileViT + DeepLabV3 semantic segmentation head.

    Composes :class:`MobileViTModel` (with ``output_stride=16`` and atrous
    convolutions in the last stage) with an ASPP module and a 1x1
    classifier convolution to produce per-pixel class logits at the input
    spatial resolution.

    References:
    - [MobileViT](https://arxiv.org/abs/2110.02178)
    - [DeepLabV3](https://arxiv.org/abs/1706.05587)
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MobileViTConfig
    HUB_REPO_SIBLINGS = MOBILEVIT_SEGMENT_HUB_SIBLINGS
    HF_MODEL_TYPE = "mobilevit"

    @classmethod
    def config_from_hf(cls, hf_config):
        neck = hf_config["neck_hidden_sizes"]
        hidden_sizes = hf_config["hidden_sizes"]
        num_classes = hf_config.get("num_labels")
        if num_classes is None and hf_config.get("id2label"):
            num_classes = len(hf_config["id2label"])
        return {
            "initial_dims": neck[0],
            "block_dims": list(neck[1:6]),
            "head_dims": neck[6],
            "attention_dims": [None, None] + list(hidden_sizes),
            "expansion_ratio": [float(hf_config["expand_ratio"])] * 5,
            "image_size": hf_config["image_size"],
            "output_stride": hf_config.get("output_stride", 16),
            "atrous_rates": list(hf_config.get("atrous_rates", [6, 12, 18])),
            "aspp_out_channels": hf_config.get("aspp_out_channels", 256),
            "aspp_dropout_prob": hf_config.get("aspp_dropout_prob", 0.1),
            "num_classes": num_classes if num_classes is not None else 21,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mobilevit_hf_to_keras import transfer_mobilevit_weights

        transfer_mobilevit_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 320,
        block_dims: tuple = (16, 24, 48, 64, 80),
        expansion_ratio: tuple = (2.0, 2.0, 2.0, 2.0, 2.0),
        attention_dims: tuple = (None, None, 64, 80, 96),
        image_size=512,
        output_stride=16,
        atrous_rates: tuple = (6, 12, 18),
        aspp_out_channels=256,
        aspp_dropout_prob=0.1,
        input_tensor=None,
        num_classes=21,
        name="MobileViTSemanticSegment",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        backbone = MobileViTModel(
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
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

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
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
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
