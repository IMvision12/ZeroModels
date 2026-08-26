from zeromodels.base import BaseConfig


class ResNetV2Config(BaseConfig):
    r"""Configuration for [`ResNetV2Model`] / [`ResNetV2ImageClassify`].

    ResNetV2 (BiT) applies pre-activation group-normalized residual blocks with a
    width multiplier. The hosted variants override `depths` / `width_factor` /
    `image_size` / `num_classes` (the in21k variants classify 21843 classes at 224
    px; the in1k fine-tunes classify 1000 at 448/480 px). One `kf_config.json`
    (declaring the canonical [`ResNetV2ImageClassify`]) sits on each variant's repo,
    and both the backbone and the classifier load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 4, 6, 3)`):
            Number of pre-activation blocks per stage.
        filters (`tuple`, *optional*, defaults to `(256, 512, 1024, 2048)`):
            Output filter counts per stage (before the width multiplier).
        width_factor (`int`, *optional*, defaults to 1):
            Width multiplier applied to every stage.
        stem_width (`int`, *optional*, defaults to 64):
            Base width of the stem convolution.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.resnetv2 import ResNetV2Config, ResNetV2ImageClassify

    >>> configuration = ResNetV2Config()
    >>> model = ResNetV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "resnetv2"

    depths: tuple = (3, 4, 6, 3)
    filters: tuple = (256, 512, 1024, 2048)
    width_factor: int = 1
    stem_width: int = 64
    image_size: int = 224
    num_classes: int = 1000
