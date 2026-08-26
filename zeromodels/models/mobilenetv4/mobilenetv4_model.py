import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .mobilenetv4_config import MobileNetV4Config
from .mobilenetv4_layers import (
    conv_bn_act,
    decode_block_str,
    edge_residual,
    make_divisible,
    mobile_attention,
    universal_inverted_residual,
)

# The backbone (MobileNetV4Model) and classifier (MobileNetV4ImageClassify) share the
# variant's repo, whose zm_config.json declares MobileNetV4ImageClassify.
MOBILENETV4_HUB_SIBLINGS = frozenset({"MobileNetV4Model", "MobileNetV4ImageClassify"})

# conv_head output width (head_hidden_size), shared across every MobileNetV4 variant.
HEAD_HIDDEN_SIZE = 1280

# Per-variant stem width, activation, layer-scale, and arch-def block schedule,
# transcribed verbatim from timm's ``_gen_mobilenet_v4`` (channel_multiplier=1.0).
# Each stage is a list of timm block strings; ``r`` in a string repeats the block.
MOBILENETV4_VARIANTS = {
    "conv_small": {
        "stem_size": 32,
        "activation": "relu",
        "layer_scale_init": None,
        "arch": [
            ["cn_r1_k3_s2_e1_c32", "cn_r1_k1_s1_e1_c32"],
            ["cn_r1_k3_s2_e1_c96", "cn_r1_k1_s1_e1_c64"],
            [
                "uir_r1_a5_k5_s2_e3_c96",
                "uir_r4_a0_k3_s1_e2_c96",
                "uir_r1_a3_k0_s1_e4_c96",
            ],
            [
                "uir_r1_a3_k3_s2_e6_c128",
                "uir_r1_a5_k5_s1_e4_c128",
                "uir_r1_a0_k5_s1_e4_c128",
                "uir_r1_a0_k5_s1_e3_c128",
                "uir_r2_a0_k3_s1_e4_c128",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "conv_medium": {
        "stem_size": 32,
        "activation": "relu",
        "layer_scale_init": None,
        "arch": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c80", "uir_r1_a3_k3_s1_e2_c80"],
            [
                "uir_r1_a3_k5_s2_e6_c160",
                "uir_r2_a3_k3_s1_e4_c160",
                "uir_r1_a3_k5_s1_e4_c160",
                "uir_r1_a3_k3_s1_e4_c160",
                "uir_r1_a3_k0_s1_e4_c160",
                "uir_r1_a0_k0_s1_e2_c160",
                "uir_r1_a3_k0_s1_e4_c160",
            ],
            [
                "uir_r1_a5_k5_s2_e6_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "uir_r2_a3_k5_s1_e4_c256",
                "uir_r1_a0_k0_s1_e4_c256",
                "uir_r1_a3_k0_s1_e4_c256",
                "uir_r1_a3_k5_s1_e2_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "uir_r2_a0_k0_s1_e4_c256",
                "uir_r1_a5_k0_s1_e2_c256",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "conv_large": {
        "stem_size": 24,
        "activation": "relu",
        "layer_scale_init": None,
        "arch": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c96", "uir_r1_a3_k3_s1_e4_c96"],
            [
                "uir_r1_a3_k5_s2_e4_c192",
                "uir_r3_a3_k3_s1_e4_c192",
                "uir_r1_a3_k5_s1_e4_c192",
                "uir_r5_a5_k3_s1_e4_c192",
                "uir_r1_a3_k0_s1_e4_c192",
            ],
            [
                "uir_r4_a5_k5_s2_e4_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r2_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r1_a5_k5_s1_e4_c512",
                "uir_r3_a5_k0_s1_e4_c512",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "hybrid_medium": {
        "stem_size": 32,
        "activation": "relu",
        "layer_scale_init": 1e-5,
        "arch": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c80", "uir_r1_a3_k3_s1_e2_c80"],
            [
                "uir_r1_a3_k5_s2_e6_c160",
                "uir_r1_a0_k0_s1_e2_c160",
                "uir_r1_a3_k3_s1_e4_c160",
                "uir_r1_a3_k5_s1_e4_c160",
                "mqa_r1_k3_h4_s1_v2_d64_c160",
                "uir_r1_a3_k3_s1_e4_c160",
                "mqa_r1_k3_h4_s1_v2_d64_c160",
                "uir_r1_a3_k0_s1_e4_c160",
                "mqa_r1_k3_h4_s1_v2_d64_c160",
                "uir_r1_a3_k3_s1_e4_c160",
                "mqa_r1_k3_h4_s1_v2_d64_c160",
                "uir_r1_a3_k0_s1_e4_c160",
            ],
            [
                "uir_r1_a5_k5_s2_e6_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "uir_r2_a3_k5_s1_e4_c256",
                "uir_r1_a0_k0_s1_e2_c256",
                "uir_r1_a3_k5_s1_e2_c256",
                "uir_r1_a0_k0_s1_e2_c256",
                "uir_r1_a0_k0_s1_e4_c256",
                "mqa_r1_k3_h4_s1_d64_c256",
                "uir_r1_a3_k0_s1_e4_c256",
                "mqa_r1_k3_h4_s1_d64_c256",
                "uir_r1_a5_k5_s1_e4_c256",
                "mqa_r1_k3_h4_s1_d64_c256",
                "uir_r1_a5_k0_s1_e4_c256",
                "mqa_r1_k3_h4_s1_d64_c256",
                "uir_r1_a5_k0_s1_e4_c256",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
    "hybrid_large": {
        "stem_size": 24,
        "activation": "gelu",
        "layer_scale_init": 1e-5,
        "arch": [
            ["er_r1_k3_s2_e4_c48"],
            ["uir_r1_a3_k5_s2_e4_c96", "uir_r1_a3_k3_s1_e4_c96"],
            [
                "uir_r1_a3_k5_s2_e4_c192",
                "uir_r3_a3_k3_s1_e4_c192",
                "uir_r1_a3_k5_s1_e4_c192",
                "uir_r2_a5_k3_s1_e4_c192",
                "mqa_r1_k3_h8_s1_v2_d48_c192",
                "uir_r1_a5_k3_s1_e4_c192",
                "mqa_r1_k3_h8_s1_v2_d48_c192",
                "uir_r1_a5_k3_s1_e4_c192",
                "mqa_r1_k3_h8_s1_v2_d48_c192",
                "uir_r1_a5_k3_s1_e4_c192",
                "mqa_r1_k3_h8_s1_v2_d48_c192",
                "uir_r1_a3_k0_s1_e4_c192",
            ],
            [
                "uir_r4_a5_k5_s2_e4_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r2_a5_k0_s1_e4_c512",
                "uir_r1_a5_k3_s1_e4_c512",
                "uir_r1_a5_k5_s1_e4_c512",
                "mqa_r1_k3_h8_s1_d64_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "mqa_r1_k3_h8_s1_d64_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "mqa_r1_k3_h8_s1_d64_c512",
                "uir_r1_a5_k0_s1_e4_c512",
                "mqa_r1_k3_h8_s1_d64_c512",
                "uir_r1_a5_k0_s1_e4_c512",
            ],
            ["cn_r1_k1_s1_c960"],
        ],
    },
}


def build_block(
    x,
    block_type,
    options,
    noskip,
    stride,
    activation,
    layer_scale_init,
    prefix,
    data_format,
    channels_axis,
    bn_epsilon,
):
    """Dispatch a decoded arch-def block onto its Keras block builder."""
    out_chs = make_divisible(int(options["c"]))
    if block_type == "cn":
        return conv_bn_act(
            x,
            filters=out_chs,
            kernel_size=int(options["k"]),
            stride=stride,
            activation=activation,
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            bn_epsilon=bn_epsilon,
        )
    if block_type == "er":
        return edge_residual(
            x,
            filters=out_chs,
            exp_kernel_size=int(options["k"]),
            stride=stride,
            exp_ratio=float(options["e"]),
            activation=activation,
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            bn_epsilon=bn_epsilon,
            noskip=noskip,
        )
    if block_type == "uir":
        return universal_inverted_residual(
            x,
            filters=out_chs,
            dw_start_kernel=int(options["a"]),
            dw_mid_kernel=int(options["k"]),
            stride=stride,
            exp_ratio=float(options["e"]),
            activation=activation,
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            layer_scale_init=layer_scale_init,
            bn_epsilon=bn_epsilon,
            noskip=noskip,
        )
    if block_type == "mqa":
        key_dim = int(options["d"])
        return mobile_attention(
            x,
            dim_out=out_chs,
            num_heads=int(options["h"]),
            key_dim=key_dim,
            value_dim=key_dim,
            kv_stride=int(options.get("v", 1)),
            dw_kernel_size=int(options["k"]),
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            layer_scale_init=layer_scale_init,
            bn_epsilon=bn_epsilon,
            noskip=noskip,
        )
    raise ValueError(f"Unknown MobileNetV4 block type: {block_type!r}")


def mobilenetv4_backbone_feature(
    inputs,
    *,
    config,
    data_format,
    channels_axis,
    return_stages=False,
    bn_epsilon=1e-5,
):
    """MobileNetV4 stem + arch-def stages (forward_features), ending at 960 channels.

    Args:
        inputs: Input image tensor.
        config: Variant key selecting the schedule from :data:`MOBILENETV4_VARIANTS`.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` channels-last, ``1`` channels-first).
        return_stages: If True, return per-stride feature maps for backbone use.
        bn_epsilon: Epsilon for every BatchNormalization layer.

    Returns:
        The final 4D feature tensor (post last stage), or a list of per-stride
        feature tensors when ``return_stages`` is True.
    """
    spec = MOBILENETV4_VARIANTS[config]
    activation = spec["activation"]
    layer_scale_init = spec["layer_scale_init"]
    stem_size = make_divisible(spec["stem_size"])

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), data_format=data_format, name="stem_padding"
    )(inputs)
    x = layers.Conv2D(
        stem_size,
        kernel_size=3,
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="conv_stem",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=0.9,
        name="bn1",
    )(x)
    x = layers.Activation(activation, name="stem_act")(x)

    stage_outputs = []
    for stage_idx, stage in enumerate(spec["arch"]):
        block_idx = 0
        for block_str in stage:
            block_type, options, noskip = decode_block_str(block_str)
            repeat = int(options.get("r", 1))
            for rep in range(repeat):
                stride = int(options["s"]) if rep == 0 else 1
                if return_stages and stride == 2:
                    stage_outputs.append(x)
                x = build_block(
                    x,
                    block_type=block_type,
                    options=options,
                    noskip=noskip,
                    stride=stride,
                    activation=activation,
                    layer_scale_init=layer_scale_init,
                    prefix=f"blocks_{stage_idx}_{block_idx}",
                    data_format=data_format,
                    channels_axis=channels_axis,
                    bn_epsilon=bn_epsilon,
                )
                block_idx += 1

    if return_stages:
        stage_outputs.append(x)
        return stage_outputs
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV4Model(BaseModel):
    """Instantiates the MobileNetV4 backbone.

    MobileNetV4 (MNv4) is built from the Universal Inverted Bottleneck (UIB) block,
    which generalizes the inverted residual with optional starting and mid depthwise
    convolutions (the IB / ConvNeXt / ExtraDW / FFN shapes), plus EdgeResidual
    (FusedIB) and plain ConvBnAct blocks. The hybrid variants additionally interleave
    a Mobile Multi-Query Attention (Mobile MQA) block with a single shared key/value
    head and depthwise-strided key/value downsampling. The network is a 3x3 conv stem
    followed by the variant's arch-def stages, ending in a 1x1 ConvBnAct that yields
    960 channels.

    Output is the last layer output before the classifier head: the 4D feature map
    of shape ``(B, H, W, 960)``. :class:`MobileNetV4ImageClassify` composes this model
    and adds a pooling + conv_head + norm + classifier head on top.

    References:
    - [MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

    Args:
        config: String, variant key selecting the block schedule, stem width,
            activation, and layer-scale. One of ``"conv_small"``, ``"conv_medium"``,
            ``"conv_large"``, ``"hybrid_medium"``, ``"hybrid_large"``. Defaults to
            `"conv_small"`.
        bn_epsilon: Float, epsilon for every BatchNormalization layer. Defaults to
            `1e-5`.
        image_size: Input image specification. Accepts an integer ``N`` (builds an
            ``N x N x 3`` square input), a 2-tuple ``(H, W)`` (assumes 3 channels),
            or a 3-tuple ordered to match the active
            ``keras.config.image_data_format()``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend image normalization.
            When True, inputs should be uint8 in ``[0, 255]``. Defaults to `True`.
        normalization_mode: String, normalization mode (see
            :func:`normalize_image_for_classify_models`). Defaults to `"imagenet"`.
        input_tensor: Optional Keras tensor as input. Defaults to `None`.
        as_backbone: Boolean, whether to output per-stride intermediate features.
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"MobileNetV4Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV4Config
    HUB_REPO_SIBLINGS = MOBILENETV4_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MobileNetV4ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MobileNetV4ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv4_timm_to_keras import transfer_mobilenetv4_weights

        transfer_mobilenetv4_weights(keras_model, state_dict)

    def __init__(
        self,
        config="conv_small",
        bn_epsilon=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="MobileNetV4Model",
        **kwargs,
    ):
        for k in (
            "num_classes",
            "classifier_activation",
            "dropout_rate",
            "timm_id",
        ):
            kwargs.pop(k, None)

        if config not in MOBILENETV4_VARIANTS:
            raise ValueError(
                f"Invalid config {config!r}. Expected one of "
                f"{sorted(MOBILENETV4_VARIANTS)}"
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
        x = mobilenetv4_backbone_feature(
            x,
            config=config,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
            bn_epsilon=bn_epsilon,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.arch = config
        self.bn_epsilon = bn_epsilon
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.arch,
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
class MobileNetV4ImageClassify(BaseModel):
    """Instantiates the MobileNetV4 classifier.

    Wraps a :class:`MobileNetV4Model` backbone and attaches the MobileNetV4 head:
    GlobalAveragePooling2D -> conv_head (1x1, 960->1280, no bias) -> BatchNorm +
    activation -> flatten -> Dropout -> Dense classifier producing ``num_classes``
    logits. The post-pool 1x1 conv followed by a normalization + activation is the
    MobileNetV4-specific ``head_norm`` head.

    References:
    - [MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

    Args:
        config: String, variant key selecting the block schedule. One of
            ``"conv_small"``, ``"conv_medium"``, ``"conv_large"``,
            ``"hybrid_medium"``, ``"hybrid_large"``. Defaults to `"conv_small"`.
        bn_epsilon: Float, epsilon for every BatchNormalization layer. Defaults to
            `1e-5`.
        image_size: Input image specification (see :class:`MobileNetV4Model`).
            Defaults to `224`.
        include_normalization: Boolean, whether to prepend image normalization.
            Defaults to `True`.
        normalization_mode: String, normalization mode. Defaults to `"imagenet"`.
        input_tensor: Optional Keras tensor as input. Defaults to `None`.
        num_classes: Integer, number of output classes. Defaults to `1000`.
        classifier_activation: String or callable, final Dense activation. Use
            `"linear"` for logits or `"softmax"` for probabilities. Defaults to
            `"linear"`.
        dropout_rate: Float, dropout applied before the classifier (skipped when
            ``<= 0``). Defaults to `0.0`.
        name: String, the name of the model. The internal backbone is named
            `f"{name}_backbone"`. Defaults to `"MobileNetV4ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV4Config
    HUB_REPO_SIBLINGS = MOBILENETV4_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv4_timm_to_keras import transfer_mobilenetv4_weights

        transfer_mobilenetv4_weights(keras_model, state_dict)

    def __init__(
        self,
        config="conv_small",
        bn_epsilon=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        dropout_rate=0.0,
        name="MobileNetV4ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if config not in MOBILENETV4_VARIANTS:
            raise ValueError(
                f"Invalid config {config!r}. Expected one of "
                f"{sorted(MOBILENETV4_VARIANTS)}"
            )

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        backbone = MobileNetV4Model(
            config=config,
            bn_epsilon=bn_epsilon,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        activation = MOBILENETV4_VARIANTS[config]["activation"]
        x = layers.GlobalAveragePooling2D(
            data_format=data_format, keepdims=True, name="global_pool"
        )(backbone.output)
        x = layers.Conv2D(
            HEAD_HIDDEN_SIZE,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name="conv_head",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=0.9,
            name="norm_head",
        )(x)
        x = layers.Activation(activation, name="head_act")(x)
        x = layers.Flatten(data_format=data_format, name="flatten")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="head_dropout")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="classifier",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.arch = config
        self.bn_epsilon = bn_epsilon
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.arch,
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
