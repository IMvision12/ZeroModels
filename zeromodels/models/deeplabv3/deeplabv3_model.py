import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.utils import standardize_input_shape

from .deeplabv3_config import DeepLabV3Config


def deeplabv3_dilated_resnet_backbone(
    inputs,
    backbone_variant,
):
    """Build a dilated ResNet backbone for DeepLabV3.

    Constructs a ResNet-50 or ResNet-101 backbone with dilated
    (atrous) convolutions in the last two stages, matching the
    torchvision DeepLabV3 backbone configuration
    (``output_stride=8``).

    Args:
        inputs: Input Keras tensor.
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.

    Returns:
        C5 feature tensor at 1/8 of the input spatial resolution.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    depths = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = inputs

    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name="backbone_bn1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="valid", data_format=data_format
    )(x)

    filters_list = [64, 128, 256, 512]
    dilate_stages = [False, False, True, True]
    current_dilation = 1

    for stage_idx, depths in enumerate(depths):
        filters = filters_list[stage_idx]
        original_stride = 2 if stage_idx > 0 else 1

        if dilate_stages[stage_idx] and stage_idx > 0:
            current_dilation *= original_stride
            stage_stride = 1
        else:
            stage_stride = original_stride

        previous_dilation = current_dilation // (
            original_stride if dilate_stages[stage_idx] and stage_idx > 0 else 1
        )

        for block_idx in range(depths):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"

            if block_idx == 0:
                block_stride = stage_stride
                block_dilation = previous_dilation
            else:
                block_stride = 1
                block_dilation = current_dilation

            residual = x

            x = layers.Conv2D(
                filters,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv1",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn1"
            )(x)
            x = layers.ReLU()(x)

            if block_stride > 1:
                pad_size = block_dilation
                x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=block_stride,
                    padding="valid",
                    dilation_rate=block_dilation,
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            else:
                if block_dilation > 1:
                    pad_size = block_dilation
                    x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(
                        x
                    )
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="valid",
                        dilation_rate=block_dilation,
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)
                else:
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)

            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn2"
            )(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(
                filters * 4,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv3",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn3"
            )(x)

            in_channels = residual.shape[channels_axis]
            out_channels = filters * 4
            if block_stride != 1 or in_channels != out_channels:
                if block_stride > 1:
                    residual = layers.ZeroPadding2D(padding=0, data_format=data_format)(
                        residual
                    )
                residual = layers.Conv2D(
                    out_channels,
                    1,
                    strides=block_stride,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

    return x


def deeplabv3_aspp(x, name="classifier_0"):
    """Atrous Spatial Pyramid Pooling module.

    Standard DeepLabV3 ASPP head:

    * one 1×1 conv branch,
    * three 3×3 atrous conv branches with dilation rates ``12``, ``24``,
      and ``36``,
    * one image-level branch (global average pool → 1×1 conv → bilinear
      upsample back to ``HxW``).

    The five branches are concatenated along the channel axis, projected
    back to 256 channels via a 1×1 conv, and passed through dropout.
    Atrous rates match torchvision's DeepLabV3 head.

    Args:
        x: Input feature tensor (typically the dilated backbone's C5
            output) of shape ``(B, H, W, C)`` for ``channels_last`` (or
            ``(B, C, H, W)`` for ``channels_first``).
        name: Prefix used for every layer name inside this block.

    Returns:
        Tensor of shape ``(B, H, W, 256)``: ASPP-fused features at the
        same spatial resolution as ``x``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    branches = []

    b0 = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_convs_0_0"
    )(x)
    b0 = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_convs_0_1"
    )(b0)
    b0 = layers.ReLU()(b0)
    branches.append(b0)

    for i, rate in enumerate([12, 24, 36], start=1):
        b = layers.Conv2D(
            256,
            3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_convs_{i}_0",
        )(x)
        b = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"{name}_convs_{i}_1",
        )(b)
        b = layers.ReLU()(b)
        branches.append(b)

    input_shape = ops.shape(x)
    if data_format == "channels_last":
        target_h, target_w = input_shape[1], input_shape[2]
    else:
        target_h, target_w = input_shape[2], input_shape[3]

    b4 = layers.GlobalAveragePooling2D(
        data_format=data_format, keepdims=True, name=f"{name}_convs_4_0"
    )(x)
    b4 = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_convs_4_1"
    )(b4)
    b4 = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_convs_4_2"
    )(b4)
    b4 = layers.ReLU()(b4)
    b4 = layers.Resizing(
        height=target_h,
        width=target_w,
        interpolation="bilinear",
        data_format=data_format,
        name=f"{name}_convs_4_upsample",
    )(b4)
    branches.append(b4)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")(branches)

    x = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_project_0"
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_project_1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    return x


def deeplabv3_classifier_head(x, num_classes, name="classifier"):
    """DeepLabV3 classifier head: 3×3 conv → BN → ReLU → 1×1 conv to logits.

    Final per-pixel projection applied after :func:`deeplabv3_aspp`.
    Output retains the spatial resolution of ``x`` (typically the
    backbone's ``output_stride=8`` resolution); pair with a bilinear
    upsample back to the input size for the final segmentation mask.

    Args:
        x: Feature tensor from :func:`deeplabv3_aspp` of shape
            ``(B, H, W, 256)``.
        num_classes: Number of segmentation classes (output channels).
        name: Prefix used for every layer name inside this block.

    Returns:
        Tensor of shape ``(B, H, W, num_classes)``: per-pixel class
        logits at the same spatial resolution as ``x``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_2"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_classes, 1, data_format=data_format, name=f"{name}_4")(x)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepLabV3Model(BaseModel):
    """DeepLabV3 dilated ResNet backbone (no segmentation head).

    Builds the dilated ResNet-50 or ResNet-101 backbone used by
    DeepLabV3 with ``output_stride=8`` and exposes the final
    2048-channel feature map. Pair with :class:`DeepLabV3SemanticSegment`
    to get the full segmentation outputs.

    Reference:
        - `Rethinking Atrous Convolution for Semantic Image
          Segmentation <https://arxiv.org/abs/1706.05587>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `520`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = None

    def __init__(
        self,
        backbone_variant="ResNet50",
        image_size=520,
        input_tensor=None,
        name="DeepLabV3Model",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        backbone_features = deeplabv3_dilated_resnet_backbone(
            img_input,
            backbone_variant=backbone_variant,
        )

        super().__init__(
            inputs=img_input, outputs=backbone_features, name=name, **kwargs
        )

        self.backbone_variant = backbone_variant
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepLabV3SemanticSegment(BaseModel):
    """DeepLabV3 full semantic segmentation model (backbone + ASPP + head).

    Composes :class:`DeepLabV3Model`, adds the ASPP module, the
    classifier head, and a bilinear upsample back to the input
    resolution. Output shape is ``(B, H, W, num_classes)`` in
    ``channels_last``.

    Reference:
        - `Rethinking Atrous Convolution for Semantic Image
          Segmentation <https://arxiv.org/abs/1706.05587>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        num_classes: Number of segmentation classes.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `520`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    config_class = DeepLabV3Config
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = None

    def __init__(
        self,
        backbone_variant="ResNet50",
        num_classes=21,
        image_size=520,
        input_tensor=None,
        name="DeepLabV3SemanticSegment",
        **kwargs,
    ):
        base = DeepLabV3Model(
            backbone_variant=backbone_variant,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )

        x = deeplabv3_aspp(base.output, name="classifier_0")
        x = deeplabv3_classifier_head(x, num_classes, name="classifier")

        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            upsample_h, upsample_w = (
                base.image_size[1],
                base.image_size[2],
            )
        else:
            upsample_h, upsample_w = (
                base.image_size[0],
                base.image_size[1],
            )
        x = layers.Resizing(
            height=upsample_h,
            width=upsample_w,
            interpolation="bilinear",
            name="final_upsampling",
        )(x)

        super().__init__(inputs=base.input, outputs=x, name=name, **kwargs)

        self.backbone_variant = backbone_variant
        self.num_classes = num_classes
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
