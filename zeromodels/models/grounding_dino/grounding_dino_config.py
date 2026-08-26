from zeromodels.base import BaseConfig


class GroundingDinoConfig(BaseConfig):
    r"""Configuration for [`GroundingDinoDetect`], the Grounding DINO detector.

    Open-set / text-grounded detection: a Swin backbone + BERT text encoder + a
    deformable cross-modality encoder-decoder. The defaults describe the
    grounding_dino_tiny style; grounding_dino_base overrides the Swin backbone
    fields. Fields serialize flat to a repo's `zm_config.json`.

    Args:
        d_model (`int`, *optional*, defaults to 256):
            Hidden dimension of the encoder-decoder transformer.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of deformable encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the decoder.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots.
        num_feature_levels (`int`, *optional*, defaults to 4):
            Number of multi-scale feature levels from the backbone.
        encoder_n_points (`int`, *optional*, defaults to 4):
            Deformable-attention sampling points per level in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            Deformable-attention sampling points per level in the decoder.
        max_text_len (`int`, *optional*, defaults to 256):
            Maximum length of the tokenized text prompt.
        query_dim (`int`, *optional*, defaults to 4):
            Dimensionality of the reference points (box coordinates).
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to use two-stage decoding (encoder region proposals).
        positional_embedding_temperature (`float`, *optional*, defaults to 20.0):
            Temperature of the sine positional embeddings.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of the transformer layer-norms.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation used in the encoder-decoder feed-forward layers.
        backbone_embed_dim (`int`, *optional*, defaults to 96):
            Embedding dimension of the Swin backbone (96 tiny, 128 base).
        backbone_depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Number of Swin blocks per backbone stage.
        backbone_num_heads (`tuple`, *optional*, defaults to `(3, 6, 12, 24)`):
            Number of attention heads per Swin backbone stage.
        backbone_window_size (`int`, *optional*, defaults to 7):
            Window size of the Swin backbone attention (7 tiny, 12 base).
        backbone_out_indices (`tuple`, *optional*, defaults to `(2, 3, 4)`):
            Swin stages whose feature maps feed the detector.
        text_vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT text encoder.
        text_hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the BERT text encoder.
        text_num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the text encoder.
        text_num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the text encoder.
        text_intermediate_size (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the text encoder.
        text_max_position_embeddings (`int`, *optional*, defaults to 512):
            Maximum position embeddings of the text encoder.
        text_layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Epsilon of the text-encoder layer-norms.

    Examples:

    ```python
    >>> from zeromodels.models.grounding_dino import GroundingDinoConfig, GroundingDinoDetect

    >>> configuration = GroundingDinoConfig()
    >>> model = GroundingDinoDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "grounding-dino"

    d_model: int = 256
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    num_queries: int = 900
    num_feature_levels: int = 4
    encoder_n_points: int = 4
    decoder_n_points: int = 4
    max_text_len: int = 256
    query_dim: int = 4
    two_stage: bool = True
    positional_embedding_temperature: float = 20.0
    layer_norm_eps: float = 1e-05
    activation_function: str = "relu"
    backbone_embed_dim: int = 96
    backbone_depths: tuple = (2, 2, 6, 2)
    backbone_num_heads: tuple = (3, 6, 12, 24)
    backbone_window_size: int = 7
    backbone_out_indices: tuple = (2, 3, 4)
    text_vocab_size: int = 30522
    text_hidden_size: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    text_intermediate_size: int = 3072
    text_max_position_embeddings: int = 512
    text_layer_norm_eps: float = 1e-12
