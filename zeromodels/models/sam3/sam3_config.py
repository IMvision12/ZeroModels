from zeromodels.base import BaseConfig


class Sam3Config(BaseConfig):
    r"""Configuration for [`SAM3Model`], SAM 3 (Segment Anything with Concepts).

    A ViT image encoder + FPN neck feeding a DETR-style encoder-decoder and a mask
    decoder, with a CLIP-style text branch for concept prompts. There is a single
    released variant; the defaults describe it. Fields serialize flat to a repo's
    `zm_config.json`.

    Args:
        vit_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the ViT image encoder.
        vit_intermediate_size (`int`, *optional*, defaults to 4736):
            Feed-forward dimension of the ViT image encoder.
        vit_num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers in the image encoder.
        vit_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the image encoder.
        vit_image_size (`int`, *optional*, defaults to 1008):
            Image resolution the ViT encoder operates at.
        vit_patch_size (`int`, *optional*, defaults to 14):
            Patch size of the ViT encoder.
        vit_window_size (`int`, *optional*, defaults to 24):
            Windowed-attention window size.
        vit_global_attn_indexes (`tuple`, *optional*, defaults to `(7, 15, 23, 31)`):
            Encoder layer indices that use global attention.
        vit_rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary-position-embedding base frequency.
        vit_pretrain_image_size (`int`, *optional*, defaults to 336):
            Resolution the position embeddings were pretrained at (interpolated).
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            Channel width of the FPN neck.
        fpn_scale_factors (`tuple`, *optional*, defaults to `(4.0, 2.0, 1.0, 0.5)`):
            Upsampling / downsampling factors for the FPN levels.
        detr_encoder_hidden_size (`int`, *optional*, defaults to 256):
            Hidden size of the DETR encoder.
        detr_encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of DETR encoder layers.
        detr_encoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the DETR encoder.
        detr_encoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the DETR encoder.
        detr_encoder_dropout (`float`, *optional*, defaults to 0.1):
            Dropout in the DETR encoder.
        detr_decoder_hidden_size (`int`, *optional*, defaults to 256):
            Hidden size of the DETR decoder.
        detr_decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of DETR decoder layers.
        detr_decoder_num_queries (`int`, *optional*, defaults to 200):
            Number of object queries.
        detr_decoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the DETR decoder.
        detr_decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the DETR decoder.
        detr_decoder_dropout (`float`, *optional*, defaults to 0.1):
            Dropout in the DETR decoder.
        mask_decoder_hidden_size (`int`, *optional*, defaults to 256):
            Hidden size of the mask decoder.
        mask_decoder_num_upsampling_stages (`int`, *optional*, defaults to 3):
            Number of upsampling stages in the mask decoder.
        mask_decoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the mask decoder.
        text_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the text (concept) encoder.
        text_projection_dim (`int`, *optional*, defaults to 512):
            Dimension of the shared image-text projection space.
        image_size (`int`, *optional*, defaults to 1008):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.sam3 import Sam3Config, SAM3Model

    >>> configuration = Sam3Config()
    >>> model = SAM3Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "sam3"

    vit_hidden_size: int = 1024
    vit_intermediate_size: int = 4736
    vit_num_hidden_layers: int = 32
    vit_num_attention_heads: int = 16
    vit_image_size: int = 1008
    vit_patch_size: int = 14
    vit_window_size: int = 24
    vit_global_attn_indexes: tuple = (7, 15, 23, 31)
    vit_rope_theta: float = 10000.0
    vit_pretrain_image_size: int = 336
    fpn_hidden_size: int = 256
    fpn_scale_factors: tuple = (4.0, 2.0, 1.0, 0.5)
    detr_encoder_hidden_size: int = 256
    detr_encoder_num_layers: int = 6
    detr_encoder_num_attention_heads: int = 8
    detr_encoder_intermediate_size: int = 2048
    detr_encoder_dropout: float = 0.1
    detr_decoder_hidden_size: int = 256
    detr_decoder_num_layers: int = 6
    detr_decoder_num_queries: int = 200
    detr_decoder_num_attention_heads: int = 8
    detr_decoder_intermediate_size: int = 2048
    detr_decoder_dropout: float = 0.1
    mask_decoder_hidden_size: int = 256
    mask_decoder_num_upsampling_stages: int = 3
    mask_decoder_num_attention_heads: int = 8
    text_hidden_size: int = 1024
    text_projection_dim: int = 512
    image_size: int = 1008
