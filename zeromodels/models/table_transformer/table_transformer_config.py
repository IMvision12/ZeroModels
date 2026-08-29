from zeromodels.base import BaseConfig


class TableTransformerConfig(BaseConfig):
    r"""Configuration for [`TableTransformerDetect`], the Table Transformer detector.

    Table Transformer (TATR) is the DETR architecture applied to table
    detection and table-structure recognition (from Microsoft's PubTables-1M).
    It reuses DETR's ResNet backbone plus transformer encoder / decoder, with
    two differences: the encoder and decoder layers are pre-normalized (the
    LayerNorm is applied before each attention / feed-forward sub-layer) and
    the encoder has an extra final LayerNorm. The backbone is a ResNet-18
    (basic blocks, 512-channel last stage), so the 1x1 input projection reduces
    512 channels to `hidden_dim`. Instantiating with the defaults yields a
    configuration close to `microsoft/table-transformer-detection`. Fields
    mirror the model constructor and serialize flat to a repo's `zm_config.json`.

    Args:
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the transformer encoder and decoder layers.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the transformer.
        num_encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Dimension of the feed-forward ("intermediate") layer in the transformer.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability in the transformer layers.
        num_queries (`int`, *optional*, defaults to 15):
            Number of object queries, i.e. detection slots. The maximal number of
            objects [`TableTransformerDetect`] can detect in a single image. The
            detection checkpoint uses 15, the structure-recognition checkpoints 125.
        num_classes (`int`, *optional*, defaults to 3):
            Number of object classes, including the no-object class (table detection:
            2 + 1; table-structure recognition: 6 + 1).
        image_size (`int`, *optional*, defaults to 800):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.table_transformer import (
    ...     TableTransformerConfig,
    ...     TableTransformerDetect,
    ... )

    >>> # Initializing a microsoft/table-transformer-detection style configuration
    >>> configuration = TableTransformerConfig()

    >>> # Initializing a model (with random weights) from that configuration
    >>> model = TableTransformerDetect(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "table-transformer"

    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout_rate: float = 0.1
    num_queries: int = 15
    num_classes: int = 3
    image_size: int = 800
