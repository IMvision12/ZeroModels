from zeromodels.base import BaseConfig


class InceptionV4Config(BaseConfig):
    r"""Configuration for [`InceptionV4Model`] / [`InceptionV4ImageClassify`].

    Inception V4 is a deeper, more uniform Inception network built from stem, Inception-
    A/B/C and reduction blocks. One `zm_config.json` (declaring the canonical
    [`InceptionV4ImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        image_size (`int`, *optional*, defaults to 299):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.inceptionv4 import (
    ...     InceptionV4Config,
    ...     InceptionV4ImageClassify,
    ... )

    >>> configuration = InceptionV4Config()
    >>> model = InceptionV4ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "inceptionv4"

    image_size: int = 299
    num_classes: int = 1000
