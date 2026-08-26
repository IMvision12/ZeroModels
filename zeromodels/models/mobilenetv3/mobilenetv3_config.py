from zeromodels.base import BaseConfig


class MobileNetV3Config(BaseConfig):
    r"""Configuration for [`MobileNetV3Model`] / [`MobileNetV3ImageClassify`].

    MobileNetV3 combines inverted-residual blocks with squeeze-and-excitation and
    hard-swish activations, tuned by NAS into `"small"` and `"large"` variants. One
    `kf_config.json` (declaring the canonical [`MobileNetV3ImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        width_multiplier (`float`, *optional*, defaults to 1.0):
            Channel-width scaling factor.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            Block-repeat (depth) scaling factor.
        config (`str`, *optional*, defaults to `"large"`):
            Block schedule preset, `"small"` or `"large"`.
        minimal (`bool`, *optional*, defaults to `False`):
            Use the minimalistic variant (ReLU, no SE, 3x3 kernels).
        block_count_multiplier (`float`, *optional*, defaults to 1.0):
            Extra per-stage block-count scaling (the `150d` variant uses 1.2).
        head_count_multiplier (`int`, *optional*, defaults to 1):
            Head channel-width multiplier (the `150d` variant uses 2).
        first_block_noskip (`bool`, *optional*, defaults to `False`):
            Disable the residual skip in the first block (the `rw` variant).
        se_round_divisor (`int`, *optional*, defaults to 8):
            Divisor for rounding squeeze-and-excitation channels; `None` matches the
            `rw` variant's un-rounded behavior.
        se_use_block_act (`bool`, *optional*, defaults to `False`):
            Use the block's activation (not ReLU) inside the SE gate (the `rw` variant).
        bn_epsilon (`float`, *optional*, defaults to 1e-5):
            BatchNorm epsilon.
        head_use_bias (`bool`, *optional*, defaults to `True`):
            Whether the classifier head convolution uses a bias (backbone ignores it).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.mobilenetv3 import (
    ...     MobileNetV3Config,
    ...     MobileNetV3ImageClassify,
    ... )

    >>> configuration = MobileNetV3Config()
    >>> model = MobileNetV3ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilenetv3"

    width_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    config: str = "large"
    minimal: bool = False
    block_count_multiplier: float = 1.0
    head_count_multiplier: int = 1
    first_block_noskip: bool = False
    se_round_divisor: int = 8
    se_use_block_act: bool = False
    bn_epsilon: float = 1e-5
    head_use_bias: bool = True
    image_size: int = 224
    num_classes: int = 1000
