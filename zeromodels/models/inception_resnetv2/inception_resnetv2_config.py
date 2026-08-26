from zeromodels.base import BaseConfig


class InceptionResNetV2Config(BaseConfig):
    r"""Configuration for [`InceptionResNetV2Model`] / [`InceptionResNetV2ImageClassify`].

    Inception-ResNet-V2 combines Inception modules with residual connections for faster
    training and strong accuracy at 299x299. One `kf_config.json` (declaring the
    canonical [`InceptionResNetV2ImageClassify`]) sits on each variant's repo, and both
    the backbone and classifier load from it. Fields mirror the model constructor and
    serialize flat.

    Args:
        image_size (`int`, *optional*, defaults to 299):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.inception_resnetv2 import (
    ...     InceptionResNetV2Config,
    ...     InceptionResNetV2ImageClassify,
    ... )

    >>> configuration = InceptionResNetV2Config()
    >>> model = InceptionResNetV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "inception_resnetv2"

    image_size: int = 299
    num_classes: int = 1000
