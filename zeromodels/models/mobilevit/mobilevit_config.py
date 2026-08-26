from zeromodels.base import BaseConfig


class MobileViTConfig(BaseConfig):
    r"""Configuration for MobileViT: [`MobileViTModel`], [`MobileViTImageClassify`]
    and [`MobileViTSemanticSegment`].

    MobileViT interleaves MobileNetV2 blocks with lightweight transformer blocks that
    apply global attention over unfolded patches, giving a compact conv/transformer
    hybrid. A single config serves the whole family: the backbone/classifier read the
    architecture fields, and the DeepLabV3 segmentation head additionally reads the
    `output_stride` / `atrous_rates` / `aspp_*` fields (the classifier ignores them).
    One `kf_config.json` sits on each variant's repo and fields serialize flat.

    Args:
        initial_dims (`int`, *optional*, defaults to 16):
            Channel count of the stem convolution.
        head_dims (`int`, *optional*, defaults to 640):
            Channel count of the final 1x1 head convolution.
        block_dims (`tuple`, *optional*, defaults to `(32, 64, 96, 128, 160)`):
            Output channel count per stage.
        expansion_ratio (`tuple`, *optional*, defaults to `(4.0, 4.0, 4.0, 4.0, 4.0)`):
            Inverted-residual expansion ratio per stage.
        attention_dims (`tuple`, *optional*, defaults to `(None, None, 144, 192, 240)`):
            Transformer hidden size per stage; `None` marks the purely convolutional
            early stages.
        image_size (`int`, *optional*, defaults to 256):
            Square input resolution the weights were trained at (512 for the
            segmentation checkpoints).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of output classes (backbone ignores it; 21 for the PASCAL VOC
            segmentation checkpoints).
        output_stride (`int`, *optional*, defaults to 32):
            Segmentation only. Ratio of input to backbone-output spatial resolution;
            the last stage uses atrous convolutions to hold this stride (16 for the
            DeepLabV3 heads). Ignored by the classifier, which always uses 32.
        atrous_rates (`tuple`, *optional*, defaults to `(6, 12, 18)`):
            Segmentation only. Dilation rates of the ASPP parallel branches.
        aspp_out_channels (`int`, *optional*, defaults to 256):
            Segmentation only. Channel count of each ASPP branch and the fused
            projection.
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            Segmentation only. Dropout probability before the final classifier conv.

    Examples:

    ```python
    >>> from zeromodels.models.mobilevit import (
    ...     MobileViTConfig,
    ...     MobileViTImageClassify,
    ... )

    >>> configuration = MobileViTConfig()
    >>> model = MobileViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilevit"

    initial_dims: int = 16
    head_dims: int = 640
    block_dims: tuple = (32, 64, 96, 128, 160)
    expansion_ratio: tuple = (4.0, 4.0, 4.0, 4.0, 4.0)
    attention_dims: tuple = (None, None, 144, 192, 240)
    image_size: int = 256
    num_classes: int = 1000
    # DeepLabV3 segmentation head (ignored by the backbone / classifier).
    output_stride: int = 32
    atrous_rates: tuple = (6, 12, 18)
    aspp_out_channels: int = 256
    aspp_dropout_prob: float = 0.1
