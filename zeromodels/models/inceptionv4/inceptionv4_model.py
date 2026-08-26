import keras
from keras import layers, utils
from keras.src.utils.argument_validation import standardize_tuple

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .inceptionv4_config import InceptionV4Config

# The backbone (InceptionV4Model) and classifier (InceptionV4ImageClassify) share the
# variant's repo, whose zm_config.json declares InceptionV4ImageClassify.
INCEPTIONV4_HUB_SIBLINGS = frozenset({"InceptionV4Model", "InceptionV4ImageClassify"})


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

    Args:
        inputs: Input Keras tensor.
        filters: Number of output filters for the convolution.
        kernel_size: Int or 2-tuple kernel size. Scalars are expanded to
            ``(k, k)``.
        strides: Stride of the convolution.
        bn_momentum: Momentum for the BatchNormalization layer.
        bn_epsilon: Epsilon for the BatchNormalization layer.
        padding: Padding mode. ``None`` triggers explicit asymmetric
            zero-padding to match the timm reference layout.
        name: Name prefix used for the conv / bn / activation layers.

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
        name=f"{name}_bn",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def stem_blocks(x, conv_block):
    """InceptionV4 initial stem: three Conv→BN→ReLU layers.

    First three layers of the InceptionV4 architecture before the
    Mixed-3a / Mixed-4a / Mixed-5a reduction sequence: stride-2 3×3 →
    valid 3×3 → same-padded 3×3 (32 → 32 → 64 channels). Brings the
    input from full resolution to roughly ``H/2`` at 64 channels before
    the Mixed blocks start downsampling further.

    Args:
        x: Input image tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.

    Returns:
        Tensor after the three stem convolutions (64 channels, spatial
        size ≈ ``H/2``).
    """
    x = conv_block(x, 32, kernel_size=3, strides=2, name="features_0")
    x = conv_block(x, 32, kernel_size=3, name="features_1")
    x = conv_block(x, 64, kernel_size=3, padding=None, name="features_2")
    return x


def mixed3a(x, conv_block, name="features_3"):
    """Mixed-3a stem-reduction block: first 2× downsample after the stem.

    Two parallel branches concatenated along the channel axis:

    1. **3×3 max-pool, stride-2**: passes through input channels (64).
    2. **Strided 3×3 conv**: 96 channels, stride-2.

    Output has 64 + 96 = 160 channels at half the input spatial size.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the maxpool and conv branches.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    conv = conv_block(x, 96, kernel_size=3, strides=2, name=f"{name}_conv")
    return layers.Concatenate(axis=channels_axis, name=name)([maxpool, conv])


def mixed4a(x, conv_block, name="features_4"):
    """Mixed-4a stem block: two parallel paths with factorized 7×7 convs.

    Spatial size is preserved. Two branches concatenated along the
    channel axis:

    1. **1×1 → 3×3**: 64 → 96 channels.
    2. **1×1 → 1×7 → 7×1 → 3×3**: 64 → 64 → 64 → 96 channels
       (factorized 7×7 sandwiched between 1×1 reductions and a final
       3×3).

    Output has 96 + 96 = 192 channels.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the two branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 96, kernel_size=3, strides=1, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, name=f"{name}_branch1_3"
    )

    return layers.Concatenate(axis=channels_axis, name=name)([branch0, branch1])


def mixed5a(x, conv_block, name="features_5"):
    """Mixed-5a stem-reduction block: second 2× downsample.

    Two parallel branches concatenated along the channel axis:

    1. **Strided 3×3 conv**: 192 channels, stride-2.
    2. **3×3 max-pool, stride-2**: passes through input channels (192).

    Output has 192 + 192 = 384 channels at half the input spatial size,
    ready to feed the Inception-A stack.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    conv = conv_block(x, 192, kernel_size=3, strides=2, name=f"{name}_conv")
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    return layers.Concatenate(axis=channels_axis, name=name)([conv, maxpool])


def inception_a(x, conv_block, block_idx):
    """Inception-A block (uniform InceptionV4 module, 4× in stage A).

    Four parallel branches concatenated along the channel axis:

    1. **1×1**: 96 channels.
    2. **1×1 → 3×3**: 64 → 96 channels.
    3. **1×1 → 3×3 → 3×3**: 64 → 96 → 96 channels (replaces 5×5 with
       a stack of 3×3s).
    4. **Avg-pool → 1×1**: same spatial → 96 channels.

    Spatial size is preserved. Output has 96 × 4 = 384 channels.
    Used four times consecutively in stage A (after Mixed-5a).

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 96, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )

    branch2 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_1"
    )
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_2"
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 96, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_a(x, conv_block, name="features_10"):
    """Reduction-A: halves spatial size between Inception-A and -B stages.

    Three parallel branches concatenated along the channel axis:

    1. **Strided 3×3**: 384 channels, stride-2.
    2. **1×1 → 3×3 → strided 3×3**: 192 → 224 → 256 channels, last
       conv stride-2.
    3. **Strided max-pool**: 3×3, stride-2 (passes through input
       channels, 384).

    Used once between the Inception-A and Inception-B stacks. Output
    has 384 + 256 + 384 = 1024 channels at half the input spatial size.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 384, kernel_size=3, strides=2, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 224, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )
    branch1 = conv_block(
        branch1, 256, kernel_size=3, strides=2, name=f"{name}_branch1_2"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_b(x, conv_block, block_idx):
    """Inception-B block (factorized 7×7 module, 7× in stage B).

    Four parallel branches with factorized 7×7 convolutions
    (``7×7 = 1×7 → 7×1``), concatenated along the channel axis:

    1. **1×1**: 384 channels.
    2. **1×1 → 1×7 → 7×1**: 192 → 224 → 256 channels (single 7×7
       factorized).
    3. **1×1 → 7×1 → 1×7 → 7×1 → 1×7**: 192 → 192 → 224 → 224 → 256
       channels (double 7×7 factorized).
    4. **Avg-pool → 1×1**: 128 channels.

    Spatial size is preserved. Output has 384 + 256 + 256 + 128 = 1024
    channels. Used seven times consecutively in stage B (after
    Reduction-A).

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )

    branch2 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        192,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3",
    )
    branch2 = conv_block(
        branch2,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_4",
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 128, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_b(x, conv_block, name="features_18"):
    """Reduction-B: halves spatial size between Inception-B and -C stages.

    Three parallel branches concatenated along the channel axis:

    1. **1×1 → strided 3×3**: 192 → 192 channels, last conv stride-2.
    2. **1×1 → 1×7 → 7×1 → strided 3×3**: 256 → 256 → 320 → 320
       channels (factorized 7×7 + stride-2 3×3).
    3. **Strided max-pool**: 3×3, stride-2 (passes through input
       channels, 1024).

    Used once between the Inception-B and Inception-C stacks. Output
    has 192 + 320 + 1024 = 1536 channels at half the input spatial
    size.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 192, kernel_size=3, strides=2, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        320,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 320, kernel_size=3, strides=2, name=f"{name}_branch1_3"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_c(x, conv_block, block_idx):
    """Inception-C block (parallel 3×3 split, 3× in stage C).

    Four parallel branches with the **parallel** (not stacked) 3×3
    factorization: each ``3×3`` is split into ``1×3 || 3×1`` branches
    whose outputs are concatenated, producing a wider feature map per
    branch instead of a deeper stack. Concatenated along the channel
    axis:

    1. **1×1**: 256 channels.
    2. **1×1 → (1×3 ∥ 3×1)**: 384 → 256 || 256 channels (concat = 512).
    3. **1×1 → 3×1 → 1×3 → (1×3 ∥ 3×1)**: 384 → 448 → 512 → 256 || 256
       channels (concat = 512).
    4. **Avg-pool → 1×1**: 256 channels.

    Output has 256 + 512 + 512 + 256 = 1536 channels: the final
    InceptionV4 feature width. Used three times consecutively in stage
    C (after Reduction-B).

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv → BN → ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four expanded branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1_1a = conv_block(
        branch1,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1a",
    )
    branch1_1b = conv_block(
        branch1,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1b",
    )
    branch1 = layers.Concatenate(axis=channels_axis)([branch1_1a, branch1_1b])

    branch2 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        448,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        512,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2_3a = conv_block(
        branch2,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3a",
    )
    branch2_3b = conv_block(
        branch2,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3b",
    )
    branch2 = layers.Concatenate(axis=channels_axis)([branch2_3a, branch2_3b])

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 256, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def inceptionv4_backbone_feature(inputs, *, data_format, return_stages=False):
    """InceptionV4 full backbone, returns the final stage feature map.

    Args:
        inputs: Input image tensor.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps
            taken at natural downsample boundaries (after Mixed3a, after
            Inception-A group, after Inception-B group, after Inception-C
            group). If False (default), return only the final stage map.

    Returns:
        Final stage feature tensor (after the last Inception-C block), or
        a list of per-stage feature maps when ``return_stages=True``.
    """
    stages = []
    x = stem_blocks(inputs, conv_block)

    x = mixed3a(x, conv_block)
    stages.append(x)  # stage 1: after Mixed3a (stride 4)

    x = mixed4a(x, conv_block)
    x = mixed5a(x, conv_block)

    for i in range(4):
        x = inception_a(x, conv_block, block_idx=6 + i)
    stages.append(x)  # stage 2: after Inception-A group (stride 8)

    x = reduction_a(x, conv_block)
    for i in range(7):
        x = inception_b(x, conv_block, block_idx=11 + i)
    stages.append(x)  # stage 3: after Inception-B group (stride 16)

    x = reduction_b(x, conv_block)
    for i in range(3):
        x = inception_c(x, conv_block, block_idx=19 + i)
    stages.append(x)  # stage 4: after Inception-C group (stride 32)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionV4Model(BaseModel):
    """Instantiates the Inception V4 backbone.

    Inception V4 streamlines the Inception family with three uniform
    Inception-A / B / C module designs interleaved by dedicated
    Reduction-A and Reduction-B blocks that handle all spatial
    downsampling; the network is deeper than InceptionV3 and drops the
    auxiliary classifier branches. The output tensor is the last layer
    output before the classifier head: the final-stage feature map
    ``(B, H, W, C)`` (after the last Inception-C block), unpooled and
    head-free. :class:`InceptionV4ImageClassify` composes this model and
    applies a GlobalAveragePooling2D + Dense head to produce logits.

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
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps at natural downsample boundaries
            (after Mixed3a, after the Inception-A group, after the
            Inception-B group, and after the Inception-C group).
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"InceptionV4Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionV4Config
    HUB_REPO_SIBLINGS = INCEPTIONV4_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with InceptionV4ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = InceptionV4ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionv4_timm_to_keras import transfer_inceptionv4_weights

        transfer_inceptionv4_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="InceptionV4Model",
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
        x = inceptionv4_backbone_feature(
            x, data_format=data_format, return_stages=as_backbone
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
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
class InceptionV4ImageClassify(BaseModel):
    """Instantiates the Inception V4 classifier.

    This classifier wraps an :class:`InceptionV4Model` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`InceptionV4Model`; only
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
            named `f"{name}_backbone"`. Defaults to `"InceptionV4ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionV4Config
    HUB_REPO_SIBLINGS = INCEPTIONV4_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionv4_timm_to_keras import transfer_inceptionv4_weights

        transfer_inceptionv4_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionV4ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionV4Model(
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
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
