"""TIPSv2 model configuration."""

from kerasformers.base import BaseConfig


class Tipsv2VisionConfig(BaseConfig):
    r"""Configuration for the TIPSv2 vision tower (the `vision_config` sub-config).

    The vision tower is a DINOv2-style ViT: a Conv2d patch embedding, a learned CLS
    token, learned register tokens, learned position embeddings (bilinearly
    interpolated for off-size inputs), and transformer blocks with LayerScale and an
    MLP or SwiGLU feed-forward.

    Args:
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the vision encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the vision encoder.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Feed-forward expansion ratio (hidden feed-forward = `hidden_dim * mlp_ratio`).
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the MLP feed-forward (ignored for SwiGLU).
        image_size (`int`, *optional*, defaults to 448):
            Square input resolution the vision tower is built for.
        patch_size (`int`, *optional*, defaults to 14):
            Patch size of the vision encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether the query/key/value projections use a bias.
        layerscale_value (`float`, *optional*, defaults to 1.0):
            Initial value for the LayerScale parameters.
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feed-forward instead of a standard MLP.
        num_register_tokens (`int`, *optional*, defaults to 1):
            Number of register tokens inserted after the CLS token.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for the layer-normalization layers.

    Example:

    ```python
    >>> from kerasformers.models.tipsv2 import Tipsv2VisionConfig

    >>> configuration = Tipsv2VisionConfig()
    ```"""

    model_type = "tipsv2_vision"

    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 1
    layer_norm_eps: float = 1e-6


class Tipsv2TextConfig(BaseConfig):
    r"""Configuration for the TIPSv2 text tower (the `text_config` sub-config).

    The text tower is a bidirectional transformer with token embeddings scaled by
    `sqrt(hidden_dim)`, fixed sinusoidal position embeddings, and masked-mean pooling.

    Args:
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the text encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the text encoder.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the text encoder.
        vocab_size (`int`, *optional*, defaults to 32000):
            Text tokenizer vocabulary size.
        max_seq_len (`int`, *optional*, defaults to 64):
            Maximum text sequence length (positions).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation used in the text MLP.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for the layer-normalization layers.
        scale_sqrt_depth (`bool`, *optional*, defaults to `True`):
            Whether to scale token embeddings by `sqrt(hidden_dim)`.
        pooling_epsilon (`float`, *optional*, defaults to 1e-8):
            Epsilon added to the token count in masked-mean pooling.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.

    Example:

    ```python
    >>> from kerasformers.models.tipsv2 import Tipsv2TextConfig

    >>> configuration = Tipsv2TextConfig()
    ```"""

    model_type = "tipsv2_text"

    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 64
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    scale_sqrt_depth: bool = True
    pooling_epsilon: float = 1e-8
    pad_token_id: int = 0


class Tipsv2Config(BaseConfig):
    r"""Configuration for TIPSv2: the composite holding each tower's sub-config.

    TIPSv2 is a CLIP/SigLIP-style dual encoder. The L2-normalized vision and text
    embeddings are compared with a temperature-scaled cosine similarity
    (`logits = text @ image.T / temperature`).

    Args:
        text_config (`Tipsv2TextConfig` or `dict`, *optional*):
            Configuration of the TIPSv2 text encoder.
        vision_config (`Tipsv2VisionConfig` or `dict`, *optional*):
            Configuration of the TIPSv2 vision tower.
        temperature_init_value (`float`, *optional*, defaults to 0.005065968260169029):
            Temperature dividing the cosine-similarity logits. TIPSv2 checkpoints do
            not store the learned temperature, so this config value is used at
            inference.

    Example:

    ```python
    >>> from kerasformers.models.tipsv2 import Tipsv2Config, Tipsv2Model

    >>> configuration = Tipsv2Config()
    >>> model = Tipsv2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "tipsv2"

    sub_configs = {"text_config": Tipsv2TextConfig, "vision_config": Tipsv2VisionConfig}
    sub_config_prefixes = {"text_config": "text_", "vision_config": "vision_"}
    group_extras = {
        "text_config": ("vocab_size", "max_seq_len"),
        "vision_config": ("image_size", "patch_size", "num_register_tokens"),
    }

    text_config: Tipsv2TextConfig | dict | None = None
    vision_config: Tipsv2VisionConfig | dict | None = None
    temperature_init_value: float = 0.005065968260169029
