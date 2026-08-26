from zeromodels.base import BaseConfig


class DFineConfig(BaseConfig):
    r"""Configuration for [`DFineDetect`], the D-FINE real-time detector.

    The defaults describe the largest (x-large) HGNetV2 + hybrid-encoder
    configuration; smaller variants override the channel / depth fields. Fields
    serialize flat to a repo's `kf_config.json`.

    Args:
        stem_channels (`tuple`, *optional*, defaults to `(3, 16, 16)`):
            Channels of the HGNetV2 backbone stem convolutions.
        stage_in_channels (`tuple`, *optional*, defaults to `(16, 64, 256, 512)`):
            Input channels of each backbone stage.
        stage_mid_channels (`tuple`, *optional*, defaults to `(16, 32, 64, 128)`):
            Middle (bottleneck) channels of each backbone stage.
        stage_out_channels (`tuple`, *optional*, defaults to `(64, 256, 512, 1024)`):
            Output channels of each backbone stage.
        stage_num_blocks (`tuple`, *optional*, defaults to `(1, 1, 2, 1)`):
            Number of HG blocks per backbone stage.
        stage_numb_of_layers (`tuple`, *optional*, defaults to `(3, 3, 3, 3)`):
            Number of layers inside each HG block per stage.
        use_lab (`bool`, *optional*, defaults to `True`):
            Whether to use the Learnable Affine Block (LAB) in the backbone.
        encoder_in_channels (`tuple`, *optional*, defaults to `(256, 512, 1024)`):
            Backbone feature-map channels fed into the hybrid encoder.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the hybrid encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward dimension of the encoder (AIFI) layers.
        encode_proj_layers (`tuple`, *optional*, defaults to `(2,)`):
            Backbone feature levels the AIFI self-attention is applied to.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Channel expansion ratio in the CCFM RepBlocks.
        ccfm_num_blocks (`int`, *optional*, defaults to 1):
            Number of blocks in the cross-scale feature-fusion module (CCFM).
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the decoder.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Feed-forward dimension of the decoder layers.
        decoder_n_points (`list`, *optional*, defaults to `None`):
            Deformable-attention sampling points per feature level (per variant).
        num_feature_levels (`int`, *optional*, defaults to 3):
            Number of multi-scale feature levels used by the decoder.
        feat_strides (`tuple`, *optional*, defaults to `(8, 16, 32)`):
            Strides of the feature levels relative to the input image.
        num_classes (`int`, *optional*, defaults to 80):
            Number of object classes (COCO detection).
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots.
        image_size (`int`, *optional*, defaults to 640):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dfine import DFineConfig, DFineDetect

    >>> configuration = DFineConfig()
    >>> model = DFineDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "d_fine"

    stem_channels: tuple = (3, 16, 16)
    stage_in_channels: tuple = (16, 64, 256, 512)
    stage_mid_channels: tuple = (16, 32, 64, 128)
    stage_out_channels: tuple = (64, 256, 512, 1024)
    stage_num_blocks: tuple = (1, 1, 2, 1)
    stage_numb_of_layers: tuple = (3, 3, 3, 3)
    use_lab: bool = True
    encoder_in_channels: tuple = (256, 512, 1024)
    encoder_hidden_dim: int = 256
    encoder_ffn_dim: int = 1024
    encode_proj_layers: tuple = (2,)
    hidden_expansion: float = 1.0
    ccfm_num_blocks: int = 1
    hidden_dim: int = 256
    decoder_num_layers: int = 6
    decoder_ffn_dim: int = 1024
    decoder_n_points: list = None
    num_feature_levels: int = 3
    feat_strides: tuple = (8, 16, 32)
    num_classes: int = 80
    num_queries: int = 300
    image_size: int = 640
