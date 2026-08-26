from zeromodels.base import BaseConfig


class SwinV2Config(BaseConfig):
    r"""Configuration for [`SwinV2Model`] / [`SwinV2ImageClassify`].

    Swin Transformer V2 scales Swin with cosine attention, log-spaced continuous
    relative-position bias, and residual post-normalization, enabling training at
    higher resolutions. One `kf_config.json` (declaring the canonical
    [`SwinV2ImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        embed_dim (`int`, *optional*, defaults to 96):
            Channel width of the first stage (doubles each stage).
        depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Number of Swin blocks per stage.
        num_heads (`tuple`, *optional*, defaults to `(3, 6, 12, 24)`):
            Number of attention heads per stage.
        window_size (`int`, *optional*, defaults to 8):
            Side length of the local attention window.
        pretrain_size (`int`, *optional*, defaults to 256):
            Resolution the relative-position bias was trained at.
        pretrained_window_size (`int`, *optional*, defaults to 0):
            Window size used during pretraining (0 when not fine-tuned across sizes).
        image_size (`int`, *optional*, defaults to 256):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.swinv2 import SwinV2Config, SwinV2ImageClassify

    >>> configuration = SwinV2Config()
    >>> model = SwinV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "swinv2"

    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    window_size: int = 8
    pretrain_size: int = 256
    pretrained_window_size: int = 0
    image_size: int = 256
    num_classes: int = 1000
