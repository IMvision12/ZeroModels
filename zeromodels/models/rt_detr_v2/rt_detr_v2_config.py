from zeromodels.base import BaseConfig


class RTDetrV2Config(BaseConfig):
    r"""Configuration for [`RTDETRV2Detect`], the RT-DETRv2 real-time detector.

    Same architecture as [`RTDetrConfig`] with the RT-DETRv2 discrete-sampling
    deformable attention in the decoder. The defaults describe the ResNet-50
    (r50vd) configuration; other variants override the backbone / encoder fields.

    Args:
        backbone_hidden_sizes (`tuple`, *optional*, defaults to `(256, 512, 1024, 2048)`):
            Output channels of each ResNet backbone stage.
        backbone_block_repeats (`tuple`, *optional*, defaults to `(3, 4, 6, 3)`):
            Number of residual blocks per backbone stage.
        backbone_embedding_size (`int`, *optional*, defaults to 64):
            Channels of the backbone stem embedding.
        backbone_layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            Residual block type, `"basic"` (r18/r34) or `"bottleneck"` (r50/r101).
        encoder_in_channels (`tuple`, *optional*, defaults to `(512, 1024, 2048)`):
            Backbone feature-map channels fed into the hybrid encoder.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the hybrid encoder.
        encoder_num_layers (`int`, *optional*, defaults to 1):
            Number of AIFI self-attention encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward dimension of the encoder layers.
        encoder_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the encoder.
        encode_proj_layers (`tuple`, *optional*, defaults to `(2,)`):
            Backbone feature levels the AIFI self-attention is applied to.
        encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation used inside the encoder layers.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            Activation used in the convolutional (backbone / CCFM) blocks.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Channel expansion ratio in the CCFM RepBlocks.
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the decoder.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward dimension of the decoder layers.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the decoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            Deformable-attention sampling points per feature level.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation used inside the decoder layers.
        num_feature_levels (`int`, *optional*, defaults to 3):
            Number of multi-scale feature levels used by the decoder.
        feat_strides (`tuple`, *optional*, defaults to `(8, 16, 32)`):
            Strides of the feature levels relative to the input image.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots.
        num_classes (`int`, *optional*, defaults to 80):
            Number of object classes (COCO detection).
        image_size (`int`, *optional*, defaults to 640):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.rt_detr_v2 import RTDetrV2Config, RTDETRV2Detect

    >>> configuration = RTDetrV2Config()
    >>> model = RTDETRV2Detect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rt_detr_v2"

    backbone_hidden_sizes: tuple = (256, 512, 1024, 2048)
    backbone_block_repeats: tuple = (3, 4, 6, 3)
    backbone_embedding_size: int = 64
    backbone_layer_type: str = "bottleneck"
    encoder_in_channels: tuple = (512, 1024, 2048)
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_num_heads: int = 8
    encode_proj_layers: tuple = (2,)
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    hidden_expansion: float = 1.0
    hidden_dim: int = 256
    decoder_num_layers: int = 6
    decoder_ffn_dim: int = 1024
    decoder_num_heads: int = 8
    decoder_n_points: int = 4
    decoder_activation_function: str = "relu"
    num_feature_levels: int = 3
    feat_strides: tuple = (8, 16, 32)
    num_queries: int = 300
    num_classes: int = 80
    image_size: int = 640
