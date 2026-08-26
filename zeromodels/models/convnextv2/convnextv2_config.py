from zeromodels.base import BaseConfig


class ConvNeXtV2Config(BaseConfig):
    r"""Configuration for [`ConvNeXtV2Model`] / [`ConvNeXtV2ImageClassify`].

    ConvNeXtV2 augments ConvNeXt with Global Response Normalization (GRN) inside each
    block and is pre-trained with the Fully Convolutional Masked Autoencoder (FCMAE)
    recipe. LayerScale is disabled (`layer_scale_init=None`). One `kf_config.json`
    (declaring the canonical [`ConvNeXtV2ImageClassify`]) sits on each variant's repo,
    and both the backbone and classifier load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 3, 9, 3)`):
            Number of ConvNeXt blocks per stage.
        projection_dim (`tuple`, *optional*, defaults to `(96, 192, 384, 768)`):
            Channel width per stage.
        use_conv (`bool`, *optional*, defaults to `False`):
            Use 1x1 Conv2D layers inside each block's MLP instead of Dense (the small
            atto/femto/pico/nano variants set this to `True`).
        use_grn (`bool`, *optional*, defaults to `True`):
            Apply Global Response Normalization inside each block (the V2 recipe).
        layer_scale_init (`float`, *optional*, defaults to `None`):
            Initial per-channel LayerScale value; `None` disables LayerScale (V2 default).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.convnextv2 import (
    ...     ConvNeXtV2Config,
    ...     ConvNeXtV2ImageClassify,
    ... )

    >>> configuration = ConvNeXtV2Config()
    >>> model = ConvNeXtV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "convnextv2"

    depths: tuple = (3, 3, 9, 3)
    projection_dim: tuple = (96, 192, 384, 768)
    use_conv: bool = False
    use_grn: bool = True
    layer_scale_init: float = None
    image_size: int = 224
    num_classes: int = 1000
