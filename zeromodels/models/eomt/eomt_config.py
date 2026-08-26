from zeromodels.base import BaseConfig


class EoMTConfig(BaseConfig):
    r"""Configuration for [`EoMTUniversalSegment`], the EoMT segmenter.

    Instantiating it with the defaults yields a configuration close to the
    eomt_large_coco_panoptic_640 style. Fields mirror the model constructor and
    serialize flat to a repo's `kf_config.json`.

    Args:
        hidden_dim (`int`, *optional*, defaults to 1024):
            Transformer hidden dimension.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Total number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads per layer.
        depths (`int`, *optional*, defaults to 4):
            Number of final encoder blocks that receive the injected object
            queries.
        num_queries (`int`, *optional*, defaults to 200):
            Number of object queries.
        num_classes (`int`, *optional*, defaults to 133):
            Number of segmentation classes.
        layerscale_value (`float`, *optional*, defaults to 1e-5):
            Initial LayerScale value.
        patch_size (`int`, *optional*, defaults to 16):
            ViT patch edge length.
        num_register_tokens (`int`, *optional*, defaults to 4):
            Number of DINOv2 register tokens.
        num_upscale_blocks (`int`, *optional*, defaults to 2):
            Number of 2x upscaling layers applied to patch features before mask
            prediction.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Feed-forward expansion ratio.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic-depth drop-path rate.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether the FFN uses a SwiGLU activation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for the layer-normalization layers.
        image_size (`int`, *optional*, defaults to 640):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.eomt import EoMTConfig, EoMTUniversalSegment

    >>> # Initializing a zeromodels/eomt_large_coco_panoptic_640 style configuration
    >>> configuration = EoMTConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = EoMTUniversalSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "eomt"

    hidden_dim: int = 1024
    num_hidden_layers: int = 24
    num_heads: int = 16
    depths: int = 4
    num_queries: int = 200
    num_classes: int = 133
    layerscale_value: float = 1e-5
    patch_size: int = 16
    num_register_tokens: int = 4
    num_upscale_blocks: int = 2
    mlp_ratio: int = 4
    drop_path_rate: float = 0.0
    attention_dropout: float = 0.0
    use_swiglu_ffn: bool = False
    layer_norm_eps: float = 1e-6
    image_size: int = 640
