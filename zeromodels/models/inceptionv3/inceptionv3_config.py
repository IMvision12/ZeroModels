from zeromodels.base import BaseConfig


class InceptionV3Config(BaseConfig):
    r"""Configuration for [`InceptionV3Model`] / [`InceptionV3ImageClassify`].

    Inception V3 factorizes larger convolutions into smaller and asymmetric ones and
    uses grid-reduction blocks for an efficient deep network at 299x299. One
    `kf_config.json` (declaring the canonical [`InceptionV3ImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        image_size (`int`, *optional*, defaults to 299):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.inceptionv3 import (
    ...     InceptionV3Config,
    ...     InceptionV3ImageClassify,
    ... )

    >>> configuration = InceptionV3Config()
    >>> model = InceptionV3ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "inceptionv3"

    image_size: int = 299
    num_classes: int = 1000
