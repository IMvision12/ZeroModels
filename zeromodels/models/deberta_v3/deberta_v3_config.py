from zeromodels.models.deberta_v2.deberta_v2_config import DebertaV2Config


class DebertaV3Config(DebertaV2Config):
    r"""Configuration for the DeBERTa-v3 encoder ([`DebertaV3Model`]) and its heads.

    DeBERTa-v3 keeps the DeBERTa-v2 architecture but is pre-trained ELECTRA-style
    (replaced-token detection with gradient-disentangled embedding sharing); the shipped
    variants use no input convolution (`conv_kernel_size=0`). One `zm_config.json`
    (declaring the canonical [`DebertaV3Model`]) sits on each variant's repo. Defaults
    below are the `deberta-v3-base` values; see [`DebertaV2Config`] for field docs.

    Examples:

    ```python
    >>> from zeromodels.models.deberta_v3 import DebertaV3Config, DebertaV3Model

    >>> configuration = DebertaV3Config()
    >>> model = DebertaV3Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deberta_v3"

    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    conv_kernel_size: int = 0
