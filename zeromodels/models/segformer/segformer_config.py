from zeromodels.base import BaseConfig


class SegformerConfig(BaseConfig):
    r"""Configuration for [`SegFormerSemanticSegment`], the SegFormer segmenter.

    Instantiating it with the defaults yields a configuration close to the
    SegFormer-B0 (ADE20K) style. Fields mirror the model constructor and
    serialize flat to a repo's `kf_config.json`.

    Args:
        embed_dim (`tuple`, *optional*, defaults to `(32, 64, 160, 256)`):
            Per-stage hidden dimensions for the four MiT backbone stages.
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Per-stage transformer-block counts.
        decode_head_dim (`int`, *optional*, defaults to 256):
            Channel width of the all-MLP decode head. `256` for B0/B1,
            `768` for B2-B5.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout applied before the final classifier.
        num_classes (`int`, *optional*, defaults to 19):
            Number of segmentation classes (Cityscapes: 19, ADE20K: 150).
        image_size (`int`, *optional*, defaults to 512):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.segformer import SegformerConfig, SegFormerSemanticSegment

    >>> # Initializing a SegFormer zeromodels/segformer_b0_ade_512 style configuration
    >>> configuration = SegformerConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = SegFormerSemanticSegment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "segformer"

    embed_dim: tuple = (32, 64, 160, 256)
    depths: tuple = (2, 2, 2, 2)
    decode_head_dim: int = 256
    dropout_rate: float = 0.1
    num_classes: int = 19
    image_size: int = 512
