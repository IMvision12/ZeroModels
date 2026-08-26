import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .xception_config import XceptionConfig

# The backbone (XceptionModel) and classifier (XceptionImageClassify) share the
# variant's repo, whose kf_config.json declares XceptionImageClassify.
XCEPTION_HUB_SIBLINGS = frozenset({"XceptionModel", "XceptionImageClassify"})

# Per-variant block configs. Each entry is a list of dicts; each dict configures
# one ``xception_module`` (or ``pre_xception_module`` when ``preact=True``).
# Matches timm/models/xception_aligned.py.
_XCEPTION_BLOCK_CFGS = {
    "41": [
        # entry flow
        {"in_chs": 64, "out_chs": 128, "stride": 2},
        {"in_chs": 128, "out_chs": 256, "stride": 2},
        {"in_chs": 256, "out_chs": 728, "stride": 2},
        # middle flow (8 blocks)
        *[{"in_chs": 728, "out_chs": 728, "stride": 1} for _ in range(8)],
        # exit flow
        {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
        {
            "in_chs": 1024,
            "out_chs": (1536, 1536, 2048),
            "stride": 1,
            "no_skip": True,
            "start_with_relu": False,
        },
    ],
    "65": [
        {"in_chs": 64, "out_chs": 128, "stride": 2},
        {"in_chs": 128, "out_chs": 256, "stride": 2},
        {"in_chs": 256, "out_chs": 728, "stride": 2},
        *[{"in_chs": 728, "out_chs": 728, "stride": 1} for _ in range(16)],
        {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
        {
            "in_chs": 1024,
            "out_chs": (1536, 1536, 2048),
            "stride": 1,
            "no_skip": True,
            "start_with_relu": False,
        },
    ],
    "71": [
        {"in_chs": 64, "out_chs": 128, "stride": 2},
        {"in_chs": 128, "out_chs": 256, "stride": 1},
        {"in_chs": 256, "out_chs": 256, "stride": 2},
        {"in_chs": 256, "out_chs": 728, "stride": 1},
        {"in_chs": 728, "out_chs": 728, "stride": 2},
        *[{"in_chs": 728, "out_chs": 728, "stride": 1} for _ in range(16)],
        {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
        {
            "in_chs": 1024,
            "out_chs": (1536, 1536, 2048),
            "stride": 1,
            "no_skip": True,
            "start_with_relu": False,
        },
    ],
    "41p": [
        # preact variants don't take ``start_with_relu``: PreXceptionModule
        # always applies a leading norm+act block-wide.
        {"in_chs": 64, "out_chs": 128, "stride": 2},
        {"in_chs": 128, "out_chs": 256, "stride": 2},
        {"in_chs": 256, "out_chs": 728, "stride": 2},
        *[{"in_chs": 728, "out_chs": 728, "stride": 1} for _ in range(8)],
        {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
        {"in_chs": 1024, "out_chs": (1536, 1536, 2048), "stride": 1, "no_skip": True},
    ],
    "65p": [
        {"in_chs": 64, "out_chs": 128, "stride": 2},
        {"in_chs": 128, "out_chs": 256, "stride": 2},
        {"in_chs": 256, "out_chs": 728, "stride": 2},
        *[{"in_chs": 728, "out_chs": 728, "stride": 1} for _ in range(16)],
        {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
        {"in_chs": 1024, "out_chs": (1536, 1536, 2048), "stride": 1, "no_skip": True},
    ],
}


def separable_conv_block(
    x,
    out_chs,
    *,
    kernel_size=3,
    stride=1,
    act_layer="relu",
    bn_epsilon=1e-3,
    channels_axis=-1,
    data_format="channels_last",
    name,
):
    """Aligned Xception ``SeparableConv2d``: depthwise → bn_dw → (act) →
    pointwise → bn_pw → (act).

    When ``act_layer`` is ``None`` the two internal activations are
    Identity (the timm ``start_with_relu=True`` case where activations
    sit between the sub-convs of ``XceptionModule`` rather than inside
    each ``SeparableConv2d``).
    """
    if stride > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)),
            data_format=data_format,
            name=f"{name}_pad",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.9,
        name=f"{name}_bn_dw",
    )(x)
    if act_layer is not None:
        x = layers.Activation(act_layer, name=f"{name}_act_dw")(x)

    x = layers.Conv2D(
        out_chs,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv_pw",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.9,
        name=f"{name}_bn_pw",
    )(x)
    if act_layer is not None:
        x = layers.Activation(act_layer, name=f"{name}_act_pw")(x)
    return x


def pre_separable_conv_block(
    x,
    out_chs,
    *,
    kernel_size=3,
    stride=1,
    first_act=True,
    act_layer="relu",
    bn_epsilon=1e-5,
    channels_axis=-1,
    data_format="channels_last",
    name,
):
    """Aligned Xception ``PreSeparableConv2d``: (optional norm+act) →
    depthwise → pointwise. No BN between the two convs: the next
    sub-conv's norm acts as that BN.
    """
    if first_act:
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=0.9,
            name=f"{name}_norm_bn",
        )(x)
        x = layers.Activation(act_layer, name=f"{name}_norm_act")(x)

    if stride > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)),
            data_format=data_format,
            name=f"{name}_pad",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_dwconv",
    )(x)
    x = layers.Conv2D(
        out_chs,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv_pw",
    )(x)
    return x


def xception_module(
    x,
    *,
    in_chs,
    out_chs,
    stride=1,
    start_with_relu=True,
    no_skip=False,
    act_layer="relu",
    bn_epsilon=1e-3,
    channels_axis=-1,
    data_format="channels_last",
    name,
):
    """Standard (non-preact) ``XceptionModule``: stack of 3 SeparableConv2d
    with an outer shortcut. The block's ``stride`` is applied to the third
    sub-conv's depthwise.

    When ``start_with_relu=True`` the activations are *outside* each
    SeparableConv2d (one ReLU before each sub-conv) and the SeparableConv2d
    itself has no internal activations. When ``start_with_relu=False`` the
    activations are *inside* each SeparableConv2d (so each one ends with
    two ReLUs, post-dw and post-pw).
    """
    if not isinstance(out_chs, (list, tuple)):
        out_chs = (out_chs, out_chs, out_chs)
    out_channels = out_chs[-1]

    skip = x
    separable_act = None if start_with_relu else act_layer

    for i in range(3):
        if start_with_relu:
            x = layers.Activation(act_layer, name=f"{name}_stack_act{i + 1}")(x)
        cur_stride = stride if i == 2 else 1
        x = separable_conv_block(
            x,
            out_chs[i],
            stride=cur_stride,
            act_layer=separable_act,
            bn_epsilon=bn_epsilon,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_stack_conv{i + 1}",
        )

    if not no_skip:
        if out_channels != in_chs or stride != 1:
            shortcut = layers.Conv2D(
                out_channels,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                data_format=data_format,
                name=f"{name}_shortcut_conv",
            )(skip)
            shortcut = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=bn_epsilon,
                momentum=0.9,
                name=f"{name}_shortcut_bn",
            )(shortcut)
        else:
            shortcut = skip
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x


def pre_xception_module(
    x,
    *,
    in_chs,
    out_chs,
    stride=1,
    no_skip=False,
    act_layer="relu",
    bn_epsilon=1e-5,
    channels_axis=-1,
    data_format="channels_last",
    name,
):
    """Preact ``PreXceptionModule``: an outer norm+act feeds both the
    shortcut path and a stack of 3 ``PreSeparableConv2d``. The first
    PreSeparableConv2d skips its own norm+act (the module's outer norm
    serves that role); the next two carry their own norm+act.
    """
    if not isinstance(out_chs, (list, tuple)):
        out_chs = (out_chs, out_chs, out_chs)
    out_channels = out_chs[-1]

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.9,
        name=f"{name}_norm_bn",
    )(x)
    x = layers.Activation(act_layer, name=f"{name}_norm_act")(x)
    skip = x  # shortcut path branches AFTER the outer norm+act in preact

    for i in range(3):
        cur_stride = stride if i == 2 else 1
        x = pre_separable_conv_block(
            x,
            out_chs[i],
            stride=cur_stride,
            first_act=i > 0,
            act_layer=act_layer,
            bn_epsilon=bn_epsilon,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_stack_conv{i + 1}",
        )

    if not no_skip:
        if out_channels != in_chs or stride != 1:
            shortcut = layers.Conv2D(
                out_channels,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                data_format=data_format,
                name=f"{name}_shortcut_conv",
            )(skip)
        else:
            shortcut = skip
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x


def xception_aligned_backbone(
    inputs,
    *,
    config,
    preact,
    bn_epsilon,
    data_format,
    channels_axis,
    return_stages=False,
):
    """Aligned Xception stem + blocks + (preact-only) trailing activation.

    Args:
        inputs: Input image tensor.
        config: Variant string keying into :data:`_XCEPTION_BLOCK_CFGS`.
        preact: If True, use PreXception modules and a preact-style stem
            (second stem conv has no BN/act).
        bn_epsilon: Epsilon for every BatchNormalization layer.
        data_format: Keras data-format string.
        channels_axis: Channel axis index.
        return_stages: If True, return a list of feature maps captured
            before each stride-2 block (plus the final output).

    Returns:
        Final feature tensor, or a list of stage feature tensors when
        ``return_stages=True``.
    """
    block_cfg = _XCEPTION_BLOCK_CFGS[config]

    # stem.0: ConvNormAct(3 -> 32, 3x3, stride=2)
    pad = 3 // 2
    x = layers.ZeroPadding2D(
        padding=((pad, pad), (pad, pad)),
        data_format=data_format,
        name="stem_0_pad",
    )(inputs)
    x = layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_0_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.9,
        name="stem_0_bn",
    )(x)
    x = layers.Activation("relu", name="stem_0_act")(x)

    # stem.1: ConvNormAct(32 -> 64, 3x3, stride=1) for non-preact;
    #         bare Conv2d for preact (the first block's norm absorbs the
    #         post-conv normalization).
    x = layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="stem_1_conv",
    )(x)
    if not preact:
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=0.9,
            name="stem_1_bn",
        )(x)
        x = layers.Activation("relu", name="stem_1_act")(x)

    stage_outputs = []
    module_fn = pre_xception_module if preact else xception_module
    for i, b in enumerate(block_cfg):
        if return_stages and b.get("stride", 1) > 1:
            stage_outputs.append(x)
        x = module_fn(
            x,
            bn_epsilon=bn_epsilon,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"block_{i}",
            **b,
        )

    if preact:
        x = layers.Activation("relu", name="final_act")(x)

    if return_stages:
        stage_outputs.append(x)
        return stage_outputs
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class XceptionModel(BaseModel):
    """Instantiates the Aligned Xception backbone (timm-compatible).

    Aligned Xception (Chen et al., DeepLab) is a refinement of the
    original Xception (Chollet 2017): each block stacks three
    SeparableConv2d, the strided downsamples move to the depthwise
    convs at the end of each block, and the entry/middle/exit flows
    are merged into a single ``block_cfg``-driven sequence. The
    ``*p`` variants are pre-activation variants where the block's
    leading BN+ReLU feeds both the shortcut and the stack.

    Output is the last layer output before the classifier head: the
    final 2048-channel feature map ``(B, H, W, C)``.
    :class:`XceptionImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
    - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

    Args:
        config: String, variant key selecting the block configuration.
            One of ``"41"``, ``"41p"``, ``"65"``, ``"65p"``, ``"71"``.
            Defaults to `"41"`.
        preact: Boolean, whether to use the preact (``*p``) module
            structure. Must match ``config``. Defaults to `False`.
        bn_epsilon: Float, epsilon for every BatchNormalization layer.
            timm uses 1e-3 for all variants except ``xception41p`` which
            uses 1e-5. Defaults to `1e-3`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps grouped by stride boundary.
            Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"XceptionModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = XceptionConfig
    HUB_REPO_SIBLINGS = XCEPTION_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with XceptionImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = XceptionImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_xception_timm_to_keras import transfer_xception_weights

        transfer_xception_weights(keras_model, state_dict, keras_model.preact)

    def __init__(
        self,
        config="41",
        preact=False,
        bn_epsilon=1e-3,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="XceptionModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "dropout_rate", "timm_id"):
            kwargs.pop(k, None)

        if config not in _XCEPTION_BLOCK_CFGS:
            raise ValueError(
                f"Invalid config. Expected one of {sorted(_XCEPTION_BLOCK_CFGS)}, "
                f"got {config!r}"
            )

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
        x = xception_aligned_backbone(
            x,
            config=config,
            preact=preact,
            bn_epsilon=bn_epsilon,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.arch = config
        self.preact = preact
        self.bn_epsilon = bn_epsilon
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "config": self.arch,
                "preact": self.preact,
                "bn_epsilon": self.bn_epsilon,
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
class XceptionImageClassify(BaseModel):
    """Instantiates the Aligned Xception classifier.

    This classifier wraps an :class:`XceptionModel` backbone and
    attaches a GlobalAveragePooling2D + Dropout + Dense head to produce
    ``num_classes`` class logits (mirrors timm's ``ClassifierHead``).
    All architectural parameters are forwarded to the underlying
    :class:`XceptionModel`; ``num_classes``, ``classifier_activation``,
    and ``dropout_rate`` are head-specific.

    References:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
    - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

    Args:
        config: String, variant key selecting the block configuration.
            One of ``"41"``, ``"41p"``, ``"65"``, ``"65p"``, ``"71"``.
            Defaults to `"41"`.
        preact: Boolean, whether to use the preact (``*p``) module
            structure. Must match ``config``. Defaults to `False`.
        bn_epsilon: Float, epsilon for every BatchNormalization layer.
            Defaults to `1e-3`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
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
        dropout_rate: Float, dropout rate applied between the pool and
            the final Dense classifier (skipped when ``<= 0``). Defaults to `0.0`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"XceptionImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = XceptionConfig
    HUB_REPO_SIBLINGS = XCEPTION_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_xception_timm_to_keras import transfer_xception_weights

        transfer_xception_weights(keras_model, state_dict, keras_model.preact)

    def __init__(
        self,
        config="41",
        preact=False,
        bn_epsilon=1e-3,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        dropout_rate=0.0,
        name="XceptionImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = XceptionModel(
            config=config,
            preact=preact,
            bn_epsilon=bn_epsilon,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="head_dropout")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.arch = config
        self.preact = preact
        self.bn_epsilon = bn_epsilon
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.dropout_rate = dropout_rate

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "config": self.arch,
                "preact": self.preact,
                "bn_epsilon": self.bn_epsilon,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "dropout_rate": self.dropout_rate,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
