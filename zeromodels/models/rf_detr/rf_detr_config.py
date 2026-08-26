from zeromodels.base import BaseConfig


class RFDetrConfig(BaseConfig):
    r"""Configuration for [`RFDETRDetect`], the RF-DETR detector (DINOv2 backbone).

    The defaults describe the rfdetr-base configuration; other variants override
    the patch / window / resolution / depth fields. Fields serialize flat to a
    repo's `kf_config.json`.

    Args:
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the transformer decoder.
        backbone_hidden_size (`int`, *optional*, defaults to 384):
            Hidden size of the DINOv2 vision backbone.
        backbone_num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads in the backbone.
        backbone_num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the backbone.
        backbone_mlp_ratio (`int`, *optional*, defaults to 4):
            MLP expansion ratio in the backbone.
        backbone_use_swiglu (`bool`, *optional*, defaults to `False`):
            Whether the backbone MLP uses a SwiGLU activation.
        num_register_tokens (`int`, *optional*, defaults to 0):
            Number of DINOv2 register tokens.
        out_feature_indexes (`list`, *optional*, defaults to `None`):
            Backbone layer indices whose features feed the detector (per variant).
        patch_size (`int`, *optional*, defaults to 14):
            Patch size of the DINOv2 backbone.
        num_windows (`int`, *optional*, defaults to 4):
            Number of windows for the backbone's windowed attention.
        positional_encoding_size (`int`, *optional*, defaults to 37):
            Grid size the backbone positional encoding is interpolated to.
        resolution (`int`, *optional*, defaults to 560):
            Backbone input resolution the variant was trained at.
        dec_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers.
        sa_nheads (`int`, *optional*, defaults to 8):
            Number of decoder self-attention heads.
        ca_nheads (`int`, *optional*, defaults to 16):
            Number of decoder cross-attention heads.
        dec_n_points (`int`, *optional*, defaults to 2):
            Deformable-attention sampling points per feature level.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots.
        num_classes (`int`, *optional*, defaults to 91):
            Number of object classes (COCO detection).
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to use two-stage decoding (encoder region proposals).
        bbox_reparam (`bool`, *optional*, defaults to `True`):
            Whether to reparameterize box predictions relative to reference points.
        lite_refpoint_refine (`bool`, *optional*, defaults to `True`):
            Whether to use the lightweight reference-point refinement.
        group_detr (`int`, *optional*, defaults to 13):
            Number of query groups for Group-DETR training-time assignment.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the decoder layers.
        image_size (`int`, *optional*, defaults to `None`):
            Square input resolution to build for; `None` uses `resolution`.

    Examples:

    ```python
    >>> from zeromodels.models.rf_detr import RFDetrConfig, RFDETRDetect

    >>> configuration = RFDetrConfig()
    >>> model = RFDETRDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"

    hidden_dim: int = 256
    backbone_hidden_size: int = 384
    backbone_num_heads: int = 6
    backbone_num_layers: int = 12
    backbone_mlp_ratio: int = 4
    backbone_use_swiglu: bool = False
    num_register_tokens: int = 0
    out_feature_indexes: list = None
    patch_size: int = 14
    num_windows: int = 4
    positional_encoding_size: int = 37
    resolution: int = 560
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_classes: int = 91
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    group_detr: int = 13
    dim_feedforward: int = 2048
    image_size: int = None


class RFDetrSegmentConfig(BaseConfig):
    r"""Configuration for [`RFDETRInstanceSegment`], RF-DETR instance segmentation.

    Same DINOv2 + decoder backbone as [`RFDetrConfig`] with a mask head on top.
    The defaults describe the rfdetr-seg-small configuration; other variants
    override the window / resolution / depth / query fields.

    Args:
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension of the transformer decoder.
        backbone_hidden_size (`int`, *optional*, defaults to 384):
            Hidden size of the DINOv2 vision backbone.
        backbone_num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads in the backbone.
        backbone_num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the backbone.
        backbone_mlp_ratio (`int`, *optional*, defaults to 4):
            MLP expansion ratio in the backbone.
        backbone_use_swiglu (`bool`, *optional*, defaults to `False`):
            Whether the backbone MLP uses a SwiGLU activation.
        num_register_tokens (`int`, *optional*, defaults to 0):
            Number of DINOv2 register tokens.
        out_feature_indexes (`list`, *optional*, defaults to `None`):
            Backbone layer indices whose features feed the detector (per variant).
        patch_size (`int`, *optional*, defaults to 12):
            Patch size of the DINOv2 backbone.
        num_windows (`int`, *optional*, defaults to 2):
            Number of windows for the backbone's windowed attention.
        positional_encoding_size (`int`, *optional*, defaults to 32):
            Grid size the backbone positional encoding is interpolated to.
        resolution (`int`, *optional*, defaults to 384):
            Backbone input resolution the variant was trained at.
        dec_layers (`int`, *optional*, defaults to 4):
            Number of decoder layers.
        sa_nheads (`int`, *optional*, defaults to 8):
            Number of decoder self-attention heads.
        ca_nheads (`int`, *optional*, defaults to 16):
            Number of decoder cross-attention heads.
        dec_n_points (`int`, *optional*, defaults to 2):
            Deformable-attention sampling points per feature level.
        num_queries (`int`, *optional*, defaults to 100):
            Number of object queries, i.e. detection slots.
        num_classes (`int`, *optional*, defaults to 91):
            Number of object classes (COCO).
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to use two-stage decoding (encoder region proposals).
        bbox_reparam (`bool`, *optional*, defaults to `True`):
            Whether to reparameterize box predictions relative to reference points.
        lite_refpoint_refine (`bool`, *optional*, defaults to `True`):
            Whether to use the lightweight reference-point refinement.
        group_detr (`int`, *optional*, defaults to 13):
            Number of query groups for Group-DETR training-time assignment.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the decoder layers.
        mask_downsample_ratio (`int`, *optional*, defaults to 4):
            Downsampling ratio of the predicted instance masks.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Hidden size of the mask-head feed-forward layers.
        seg_activation (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the segmentation mask head.
        image_size (`int`, *optional*, defaults to `None`):
            Square input resolution to build for; `None` uses `resolution`.

    Examples:

    ```python
    >>> from zeromodels.models.rf_detr import RFDetrSegmentConfig, RFDETRInstanceSegment

    >>> configuration = RFDetrSegmentConfig()
    >>> model = RFDETRInstanceSegment(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"

    hidden_dim: int = 256
    backbone_hidden_size: int = 384
    backbone_num_heads: int = 6
    backbone_num_layers: int = 12
    backbone_mlp_ratio: int = 4
    backbone_use_swiglu: bool = False
    num_register_tokens: int = 0
    out_feature_indexes: list = None
    patch_size: int = 12
    num_windows: int = 2
    positional_encoding_size: int = 32
    resolution: int = 384
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 100
    num_classes: int = 91
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    group_detr: int = 13
    dim_feedforward: int = 2048
    mask_downsample_ratio: int = 4
    intermediate_size: int = 1024
    seg_activation: str = "gelu"
    image_size: int = None
