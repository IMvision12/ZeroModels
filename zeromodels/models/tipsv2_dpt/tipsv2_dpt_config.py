"""TIPSv2-DPT model configuration."""

from zeromodels.base import BaseConfig


class Tipsv2DptConfig(BaseConfig):
    r"""Configuration for the TIPSv2-DPT models.

    TIPSv2-DPT stacks independent DPT (Dense Prediction Transformer) heads (depth,
    semantic segmentation) on a shared TIPSv2 vision backbone. Each head reassembles
    the backbone hidden states at ``out_indices`` into a feature pyramid, fuses them
    (RefineNet-style), and decodes a dense map.

    Args:
        image_size (`int`, *optional*, defaults to 448):
            Square input resolution.
        patch_size (`int`, *optional*, defaults to 14):
            Backbone patch size.
        num_register_tokens (`int`, *optional*, defaults to 1):
            Backbone register-token count.
        vision_hidden_dim (`int`, *optional*, defaults to 1152):
            Backbone hidden size.
        vision_num_layers (`int`, *optional*, defaults to 27):
            Backbone depth.
        vision_num_heads (`int`, *optional*, defaults to 16):
            Backbone attention heads.
        vision_mlp_ratio (`float`, *optional*, defaults to 3.7361111111111112):
            Backbone MLP expansion ratio.
        vision_use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Backbone SwiGLU feed-forward toggle.
        vision_layerscale_value (`float`, *optional*, defaults to 1.0):
            Backbone LayerScale init.
        vision_layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Backbone LayerNorm epsilon.
        out_indices (`list[int]`, *optional*, defaults to `[7, 14, 21, 27]`):
            1-indexed backbone stages (block outputs) fed to the DPT necks.
        neck_hidden_sizes (`list[int]`, *optional*, defaults to `[144, 288, 576, 1152]`):
            Reassemble projection channels per pyramid level.
        reassemble_factors (`list[float]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            Spatial up/down-sampling factors of the reassemble layers.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            Channel width inside the fusion stage and decoders.
        readout_activation (`str`, *optional*, defaults to `"gelu_tanh"`):
            Activation after the readout projection.
        num_depth_bins (`int`, *optional*, defaults to 256):
            Number of depth bins for the depth head's soft-argmax.
        min_depth (`float`, *optional*, defaults to 0.001):
            Minimum depth (meters).
        max_depth (`float`, *optional*, defaults to 10.0):
            Maximum depth (meters).
        depth_decoder_activation (`str`, *optional*, defaults to `"relu"`):
            Activation inside the depth decoder.
        num_labels (`int`, *optional*, defaults to 150):
            Number of semantic-segmentation classes.

    Example:

    ```python
    >>> from zeromodels.models.tipsv2_dpt import (
    ...     Tipsv2DptConfig,
    ...     Tipsv2DptDensePredict,
    ... )

    >>> configuration = Tipsv2DptConfig()
    >>> model = Tipsv2DptDensePredict(configuration)
    ```"""

    model_type = "tipsv2_dpt"

    image_size: int = 448
    patch_size: int = 14
    num_register_tokens: int = 1
    vision_hidden_dim: int = 1152
    vision_num_layers: int = 27
    vision_num_heads: int = 16
    vision_mlp_ratio: float = 3.7361111111111112
    vision_use_swiglu_ffn: bool = False
    vision_layerscale_value: float = 1.0
    vision_layer_norm_eps: float = 1e-6
    out_indices: list = None
    neck_hidden_sizes: list = None
    reassemble_factors: list = None
    fusion_hidden_size: int = 256
    readout_activation: str = "gelu_tanh"
    num_depth_bins: int = 256
    min_depth: float = 0.001
    max_depth: float = 10.0
    depth_decoder_activation: str = "relu"
    num_labels: int = 150
