from zeromodels.models.roberta.roberta_config import RobertaConfig


class XLMRobertaConfig(RobertaConfig):
    r"""Configuration for the XLM-RoBERTa encoder ([`XLMRobertaModel`]) and heads.

    XLM-RoBERTa is RoBERTa trained on 100 languages of CommonCrawl; architecturally it
    is RoBERTa with a much larger multilingual vocabulary. One `zm_config.json`
    (declaring the canonical [`XLMRobertaModel`]) sits on each variant's repo. Fields
    mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Multilingual token vocabulary size.

    See [`RobertaConfig`] for the remaining fields (defaults are identical apart from
    `vocab_size`).

    Examples:

    ```python
    >>> from zeromodels.models.xlm_roberta import XLMRobertaConfig, XLMRobertaModel

    >>> configuration = XLMRobertaConfig()
    >>> model = XLMRobertaModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "xlm_roberta"

    vocab_size: int = 250002
