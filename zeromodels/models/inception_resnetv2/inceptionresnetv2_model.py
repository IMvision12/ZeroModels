import keras
from keras import layers, utils
from keras.src.utils.argument_validation import standardize_tuple

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .inception_resnetv2_config import InceptionResNetV2Config

# The backbone (InceptionResNetV2Model) and classifier
# (InceptionResNetV2ImageClassify) share the variant's repo, whose kf_config.json
# declares InceptionResNetV2ImageClassify.
INCEPTION_RESNETV2_HUB_SIBLINGS = frozenset(
    {"InceptionResNetV2Model", "InceptionResNetV2ImageClassify"}
)


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    strides=1,
    bn_momentum=0.9,
    bn_epsilon=1e-3,
    padding="valid",
    name="conv2d_block",
):
    """Conv -> BatchNorm -> ReLU with optional asymmetric ZeroPadding.

    When ``padding`` is ``None``, computes timm-style asymmetric padding for the
    given ``kernel_size`` and ``strides`` and applies it via ZeroPadding2D before
    a ``valid``-padded conv. Otherwise the Keras ``padding`` mode is used directly.

    Args:
        inputs: Input feature map.
        filters: Number of output channels.
        kernel_size: Conv kernel size (int or 2-tuple). Defaults to ``1``.
        strides: Conv strides. Defaults to ``1``.
        bn_momentum: BatchNormalization momentum. Defaults to ``0.9``.
        bn_epsilon: BatchNormalization epsilon. Defaults to ``1e-3``.
        padding: Keras padding mode, or ``None`` to apply timm-style asymmetric
            ZeroPadding2D. Defaults to ``"valid"``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor after Conv -> BN -> ReLU.
    """
    kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    x = inputs

    if padding is None:

        def calculate_padding(kernel_dim):
            pad_total = kernel_dim - 1
            pad_size = pad_total // 2
            pad_extra = (kernel_dim - 1) % 2
            return pad_size, pad_extra

        pad_h, extra_h = calculate_padding(kernel_size[0])
        pad_w, extra_w = calculate_padding(kernel_size[1])

        if strides > 1:
            padding_config = ((pad_h + extra_h, pad_h), (pad_w + extra_w, pad_w))
        else:
            padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        x = layers.ZeroPadding2D(padding=padding_config, name=f"{name}_padding")(x)
        padding = "valid"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=keras.config.image_data_format(),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_5b_block(inputs, name="mixed_5b"):
    """Stem-end Mixed-5b inception block: feeds the Inception-ResNet-A stack.

    Four parallel branches concatenated along the channel axis:

    1. **1×1**: 96 channels.
    2. **1×1 → 5×5**: 48 → 64 channels.
    3. **1×1 → 3×3 → 3×3**: 64 → 96 → 96 channels.
    4. **Avg-pool → 1×1**: 64 channels.

    Spatial size is preserved. Output has 96 + 64 + 96 + 64 = 320
    channels: the input width expected by every :func:`block35`
    Inception-ResNet-A block downstream.

    Args:
        inputs: Input feature map.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with 320 channels (concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 96, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 48, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 64, 5, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 64, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_2")

    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        padding="same",
        data_format=keras.config.image_data_format(),
    )(inputs)
    branch_pool = conv_block(branch_pool, 64, name=f"{name}_branch3_1")

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block35(inputs, scale=1.0, name="repeat_0"):
    """Inception-ResNet-A residual block (35×35 stage; 10× in the model).

    Three parallel inception branches are concatenated, projected to
    320 channels by a 1×1 conv, scaled by ``scale``, added back into
    ``inputs`` as a residual, then ReLU. Branches:

    1. **1×1**: 32 channels.
    2. **1×1 → 3×3**: 32 → 32 channels.
    3. **1×1 → 3×3 → 3×3**: 32 → 48 → 64 channels.

    The ``scale=0.17`` factor (used throughout the model) keeps the
    residual contribution small enough for the training to remain
    stable: without it the deeper Inception-ResNet variants diverge.
    Spatial size is preserved; channels stay at 320.

    Args:
        inputs: Input feature map (320 channels).
        scale: Scalar multiplied with the residual branch before the add.
            Defaults to ``1.0``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``inputs``.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 32, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 32, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 32, 3, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 32, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 48, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 64, 3, padding="same", name=f"{name}_branch2_2")

    branches = [branch0, branch1, branch2]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(320, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_6a_block(inputs, name="mixed_6a"):
    """Reduction-A: halves spatial size between Inception-ResNet-A and -B.

    Three parallel branches concatenated along the channel axis:

    1. **Strided 3×3**: 384 channels, stride-2, valid padding.
    2. **1×1 → 3×3 → strided 3×3**: 256 → 256 → 384 channels, last
       conv stride-2.
    3. **3×3 max-pool, stride-2**: passes through input channels (320).

    Output has 384 + 384 + 320 = 1088 channels: the input width
    expected by every :func:`block17` Inception-ResNet-B block
    downstream.

    Args:
        inputs: Input feature map.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with 1088 channels and spatial size halved.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(
        inputs, 384, 3, strides=2, padding="valid", name=f"{name}_branch0"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 256, 3, padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(
        branch1, 384, 3, strides=2, padding="valid", name=f"{name}_branch1_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)([branch0, branch1, branch_pool])


def block17(inputs, scale=1.0, name="repeat_1_0"):
    """Inception-ResNet-B residual block (17×17 stage; 20× in the model).

    Two parallel inception branches with factorized 7×7 convs are
    concatenated, projected to 1088 channels by a 1×1 conv, scaled by
    ``scale``, added back into ``inputs`` as a residual, then ReLU.
    Branches:

    1. **1×1**: 192 channels.
    2. **1×1 → 1×7 → 7×1**: 128 → 160 → 192 channels (factorized 7×7).

    Uses ``scale=0.10`` throughout the model for training stability.
    Spatial size is preserved; channels stay at 1088.

    Args:
        inputs: Input feature map (1088 channels).
        scale: Scalar multiplied with the residual branch before the add.
            Defaults to ``1.0``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``inputs``.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 128, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 160, (1, 7), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 192, (7, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(1088, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_7a_block(inputs, name="mixed_7a"):
    """Reduction-B: halves spatial size between Inception-ResNet-B and -C.

    Four parallel branches concatenated along the channel axis:

    1. **1×1 → strided 3×3**: 256 → 384 channels, last conv stride-2.
    2. **1×1 → strided 3×3**: 256 → 288 channels, last conv stride-2.
    3. **1×1 → 3×3 → strided 3×3**: 256 → 288 → 320 channels, last
       conv stride-2.
    4. **3×3 max-pool, stride-2**: passes through input channels (1088).

    Output has 384 + 288 + 320 + 1088 = 2080 channels: the input width
    expected by every :func:`block8` Inception-ResNet-C block
    downstream.

    Args:
        inputs: Input feature map.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with 2080 channels and spatial size halved.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 256, 1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 384, 3, strides=2, padding="valid", name=f"{name}_branch0_1"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 288, 3, strides=2, padding="valid", name=f"{name}_branch1_1"
    )

    branch2 = conv_block(inputs, 256, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 288, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(
        branch2, 320, 3, strides=2, padding="valid", name=f"{name}_branch2_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block8(inputs, scale=1.0, activation=True, name="repeat_2_0"):
    """Inception-ResNet-C residual block (8×8 stage; 10× in the model).

    Two parallel inception branches with factorized 3×3 convs are
    concatenated, projected to 2080 channels by a 1×1 conv, scaled by
    ``scale``, added back into ``inputs`` as a residual, then ReLU
    (skipped on the final block). Branches:

    1. **1×1**: 192 channels.
    2. **1×1 → 1×3 → 3×1**: 192 → 224 → 256 channels (factorized 3×3).

    Uses ``scale=0.20`` throughout the model for training stability.
    The very last block in the C-stack passes ``activation=False`` so
    the trailing 1×1 head conv sees the pre-ReLU residual sum (matches
    timm's reference). Spatial size is preserved; channels stay at 2080.

    Args:
        inputs: Input feature map (2080 channels).
        scale: Scalar multiplied with the residual branch before the add.
            Defaults to ``1.0``.
        activation: If ``True``, apply ReLU after the residual add. Set to
            ``False`` for the final block before the 1x1 projection.
            Defaults to ``True``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``inputs``.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 192, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 224, (1, 3), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 256, (3, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(2080, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    if activation:
        x = layers.Activation("relu", name=name)(x)
    return x


def inception_resnet_v2_backbone_feature(inputs, *, data_format, return_stages=False):
    """InceptionResNetV2 full backbone (stem + 3 inception-residual stages + head).

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If ``True``, return a list of per-stage feature maps
            collected at natural reduction boundaries: after the Inception-
            ResNet-A stack (post Mixed_5b + block35 stack), after the
            Inception-ResNet-B stack (post Mixed_6a + block17 stack), and
            after the Inception-ResNet-C stack (post Mixed_7a + block8 stack,
            BEFORE the trailing 1x1 head conv). Defaults to ``False``.

    Returns:
        Final feature map with 1536 channels at spatial size ``H/32`` when
        ``return_stages=False``. When ``return_stages=True``, a list of 3
        per-stage feature maps.
    """
    x = conv_block(inputs, 32, 3, strides=2, padding="valid", name="conv2d_1a")
    x = conv_block(x, 32, 3, padding="valid", name="conv2d_2a")
    x = conv_block(x, 64, 3, padding="same", name="conv2d_2b")
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv_block(x, 80, 1, name="conv2d_3b")
    x = conv_block(x, 192, 3, padding="valid", name="conv2d_4a")
    x = layers.MaxPooling2D(3, strides=2)(x)

    stages = []

    x = mixed_5b_block(x, name="mixed_5b")
    for i in range(10):
        x = block35(x, scale=0.17, name=f"repeat_{i}")
    stages.append(x)

    x = mixed_6a_block(x, name="mixed_6a")
    for i in range(20):
        x = block17(x, scale=0.10, name=f"repeats_1_{i}")
    stages.append(x)

    x = mixed_7a_block(x, name="mixed_7a")
    for i in range(9):
        x = block8(x, scale=0.20, name=f"repeats_2_{i}")
    x = block8(x, activation=False, name="block8")
    stages.append(x)

    if return_stages:
        return stages

    x = conv_block(x, 1536, 1, name="conv2d_7b")
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionResNetV2Model(BaseModel):
    """Instantiates the Inception-ResNet-v2 backbone.

    Inception-ResNet-v2 combines Inception modules: parallel branches of
    1x1, factorized n x 1 / 1 x n, and pooled convs concatenated along
    the channel axis: with ResNet-style identity skip connections that
    add a scaled residual back into the input of each block. The network
    has a deep conv stem followed by three inception-residual stacks
    (A/B/C, operating on 35x35, 17x17, and 8x8 feature maps) separated
    by Reduction-A and Reduction-B blocks, and a final 1x1 projection to
    1536 channels.

    Output is the last layer output before the classifier head:
    the final feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first) with 1536 channels at spatial
    resolution ``H/32``, unpooled and head-free.
    :class:`InceptionResNetV2ImageClassify` composes this model and appends
    GAP + Dense.

    References:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of 3
            per-stage feature maps taken at the natural reduction
            boundaries (after the A-stack, after the B-stack, and after
            the C-stack, before the trailing 1x1 head conv).
            Defaults to `False`.
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
        name: String, the name of the model.
            Defaults to `"InceptionResNetV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionResNetV2Config
    HUB_REPO_SIBLINGS = INCEPTION_RESNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with InceptionResNetV2ImageClassify (which
        # the kf_config declares); build from kf_config, then copy backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = InceptionResNetV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionresnetv2_timm_to_keras import (
            transfer_inception_resnet_v2_weights,
        )

        transfer_inception_resnet_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="InceptionResNetV2Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

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
        x = inception_resnet_v2_backbone_feature(
            x, data_format=data_format, return_stages=as_backbone
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
class InceptionResNetV2ImageClassify(BaseModel):
    """Instantiates the Inception-ResNet-v2 classifier.

    This classifier wraps an :class:`InceptionResNetV2Model` backbone
    and attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`InceptionResNetV2Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

    Args:
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
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`.
            Defaults to `"InceptionResNetV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionResNetV2Config
    HUB_REPO_SIBLINGS = INCEPTION_RESNETV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionresnetv2_timm_to_keras import (
            transfer_inception_resnet_v2_weights,
        )

        transfer_inception_resnet_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionResNetV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = InceptionResNetV2Model(
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
