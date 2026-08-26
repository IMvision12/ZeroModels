from zeromodels.base import BaseConfig


class GptConfig(BaseConfig):
    r"""Configuration for the original GPT backbone ([`GptModel`]) and its
    generative head ([`GptTextGenerate`]).

    GPT (Radford et al. 2018) is a decoder-only transformer with learned token and
    absolute-position embeddings and post-LayerNorm blocks (no final norm). One
    `zm_config.json` (declaring the canonical [`GptModel`]) sits on the variant's
    repo; both the backbone and the generative head load from it. Fields mirror the
    model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 40478):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Model / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward hidden width per block.
        num_layers (`int`, *optional*, defaults to 12):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 12):
            Attention heads per block.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            Size of the learned position table.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            LayerNorm epsilon.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`GptTextGenerate`] ties the LM head to the token embeddings.

    Examples:

    ```python
    >>> from zeromodels.models.gpt import GptConfig, GptModel

    >>> configuration = GptConfig()
    >>> model = GptModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "openai-gpt"

    vocab_size: int = 40478
    embed_dim: int = 768
    mlp_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 512
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
