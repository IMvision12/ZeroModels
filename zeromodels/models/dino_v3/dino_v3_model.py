import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.models.convnext.convnext_model import convnext_backbone_feature
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .dino_v3_config import DinoV3ConvNeXtConfig, DinoV3ViTConfig
from .dino_v3_layers import (
    DinoV3Attention,
    DinoV3CLSToken,
    DinoV3LayerScale,
    DinoV3RegisterTokens,
    build_rope_2d_cache,
)


def dinov3_swiglu_ffn(
    x, embed_dim, hidden_dim, block_idx, hidden_act="gelu", mlp_bias=True
):
    """Gated feed-forward network used in DINOv3 ViT blocks (GeGLU / SwiGLU).

    Three Dense projections: a gate branch (activated) and a value /
    up branch: are multiplied elementwise, then projected back to
    ``embed_dim`` by a third Dense. With ``hidden_act="silu"`` this is the
    standard SwiGLU; with ``"gelu"`` it is GeGLU. DINOv3 selects the
    variant via the ``use_swiglu`` flag on the encoder block.

    Args:
        x: Input token sequence of shape ``(B, N, embed_dim)``.
        embed_dim: Output / residual channel dimension.
        hidden_dim: Width of the gate / up branches.
        block_idx: Block index, used to name the inner Dense layers
            (``blocks_{block_idx}_swiglu_*``).
        hidden_act: Activation applied to the gate branch (``"silu"``
            for SwiGLU, ``"gelu"`` for GeGLU).
        mlp_bias: Whether the three Dense layers carry biases.

    Returns:
        Tensor of shape ``(B, N, embed_dim)``.
    """
    gate = layers.Dense(
        hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_gate"
    )(x)
    gate = layers.Activation(hidden_act)(gate)
    up = layers.Dense(
        hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_up"
    )(x)
    x = layers.Multiply()([gate, up])
    x = layers.Dense(
        embed_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_down"
    )(x)
    return x


def dinov3_mlp_block(
    x, embed_dim, hidden_dim, block_idx, hidden_act="gelu", mlp_bias=True
):
    """Standard two-layer transformer MLP: Dense → activation → Dense.

    Used by DINOv3 blocks when ``use_swiglu=False``. Layer names follow
    ``blocks_{block_idx}_dense_{1,2}`` so the PyTorch state-dict can be
    transferred by suffix.

    Args:
        x: Input token sequence of shape ``(B, N, embed_dim)``.
        embed_dim: Output / residual channel dimension.
        hidden_dim: Intermediate Dense width.
        block_idx: Block index used in layer names.
        hidden_act: Activation name (typically ``"gelu"``).
        mlp_bias: Whether both Dense layers carry biases.

    Returns:
        Tensor of shape ``(B, N, embed_dim)``.
    """
    x = layers.Dense(hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_dense_1")(
        x
    )
    x = layers.Activation(hidden_act, name=f"blocks_{block_idx}_{hidden_act}")(x)
    x = layers.Dense(embed_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_dense_2")(
        x
    )
    return x


def dinov3_transformer_block(
    inputs,
    embed_dim,
    num_heads,
    mlp_hidden_dim,
    num_prefix_tokens,
    rope_theta,
    use_swiglu,
    layer_scale_init,
    block_idx,
    rope_cos,
    rope_sin,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    hidden_act="gelu",
    mlp_bias=True,
    layer_norm_eps=1e-5,
):
    """One pre-LN DINOv3 transformer block (2D-RoPE attention + MLP).

    Structure:

    1. Pre-norm → :class:`DinoV3Attention` (with 2D RoPE applied to
       Q/K of the patch tokens; prefix tokens, CLS + registers, skip
       RoPE) → optional :class:`DinoV3LayerScale` → residual.
    2. Pre-norm → :func:`dinov3_mlp_block` *or* :func:`dinov3_swiglu_ffn`
       (selected by ``use_swiglu``) → optional :class:`DinoV3LayerScale` →
       residual.

    All sublayer names are deterministic (``blocks_{block_idx}_*``) so
    the source DINOv3 state-dict can be transferred by name. The shared
    RoPE cache (``rope_cos``, ``rope_sin``) is set once per block via
    :meth:`DinoV3Attention.set_rope_cache`, avoiding re-computation per
    forward pass.

    Reference:
        - `DINOv3 <https://arxiv.org/abs/2508.10104>`_

    Args:
        inputs: Token sequence ``(B, N, embed_dim)``, typically
            ``[CLS, registers..., patch_tokens...]``.
        embed_dim: Hidden / model dimension.
        num_heads: Number of attention heads.
        mlp_hidden_dim: Hidden width of the MLP / SwiGLU branch.
        num_prefix_tokens: Number of leading tokens that bypass RoPE
            (CLS + registers).
        rope_theta: RoPE base frequency.
        use_swiglu: If ``True`` use :func:`dinov3_swiglu_ffn`; otherwise
            :func:`dinov3_mlp_block`.
        layer_scale_init: Initial DinoV3LayerScale value. Pass ``None`` to disable
            DinoV3LayerScale on both residual branches.
        block_idx: Block index used in every layer name.
        rope_cos: Pre-computed cosine cache for 2D RoPE.
        rope_sin: Pre-computed sine cache for 2D RoPE.
        query_bias: Whether the query projection carries a bias.
        key_bias: Whether the key projection carries a bias.
        value_bias: Whether the value projection carries a bias.
        hidden_act: MLP / SwiGLU activation name.
        mlp_bias: Whether MLP / SwiGLU Dense layers carry biases.
        layer_norm_eps: Epsilon for both pre-norm LayerNorms.

    Returns:
        Tensor of shape ``(B, N, embed_dim)``: the block's output sequence.
    """
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    attn = DinoV3Attention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_prefix_tokens=num_prefix_tokens,
        rope_theta=rope_theta,
        query_bias=query_bias,
        key_bias=key_bias,
        value_bias=value_bias,
        block_prefix=f"blocks_{block_idx}",
    )
    attn.set_rope_cache(rope_cos, rope_sin)
    x = attn(x)
    if layer_scale_init is not None:
        x = DinoV3LayerScale(
            layer_scale_init=layer_scale_init, name=f"blocks_{block_idx}_layerscale_1"
        )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    y = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    if use_swiglu:
        y = dinov3_swiglu_ffn(
            y, embed_dim, mlp_hidden_dim, block_idx, hidden_act, mlp_bias
        )
    else:
        y = dinov3_mlp_block(
            y, embed_dim, mlp_hidden_dim, block_idx, hidden_act, mlp_bias
        )
    if layer_scale_init is not None:
        y = DinoV3LayerScale(
            layer_scale_init=layer_scale_init, name=f"blocks_{block_idx}_layerscale_2"
        )(y)
    out = layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return out


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoV3ViTModel(BaseModel):
    """DINOv3 Vision Transformer model with 2D RoPE and register tokens.

    When ``as_backbone=False`` (default), returns the final
    LayerNorm-normalized token sequence ``(B, num_tokens, embed_dim)`` (CLS +
    register tokens followed by patch tokens). When ``as_backbone=True``,
    returns the list of intermediate feature maps (initial embedding +
    each transformer block, last LayerNorm-normalized), suitable for
    feeding into detection / segmentation / depth necks.

    Weights are gated on the model Hub: see
    https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m for
    license acceptance. Accept the license and set ``HF_TOKEN`` env var
    before calling :meth:`from_weights`.

    Reference:
        - `DINOv3 <https://arxiv.org/abs/2508.10104>`_

    Args:
        as_backbone: If ``True``, output the list of intermediate
            features (last LayerNorm-normalized) for use as a backbone.
            If ``False`` (default), output only the final
            LayerNorm-normalized token sequence.
        patch_size: ViT patch size. DINOv3 uses 16.
        embed_dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        use_swiglu: Whether to use a gated MLP (GeGLU / SwiGLU)
            instead of the standard two-layer MLP. Defaults to ``False``.
        num_register_tokens: Number of register tokens. Defaults to ``4``.
        layer_scale_init: DinoV3LayerScale init value. Defaults to ``1.0``.
        rope_theta: 2D-RoPE frequency base. Defaults to ``100.0``.
        query_bias: Whether the attention Q projection uses bias.
            Defaults to ``True`` (canonical DINOv3 setting).
        key_bias: Whether the attention K projection uses bias.
            Defaults to ``False`` (canonical DINOv3 setting).
        value_bias: Whether the attention V projection uses bias.
            Defaults to ``True`` (canonical DINOv3 setting).
        hidden_act: MLP activation name (``"gelu"`` or ``"silu"``).
            Defaults to ``"gelu"``.
        mlp_bias: Whether MLP Dense layers use bias. Defaults to ``True``.
        layer_norm_eps: Epsilon for LayerNorm layers. Defaults to ``1e-5``.
        include_normalization: Whether to prepend
            image normalization.
        normalization_mode: Normalization preset.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DinoV3ViTConfig
    HF_MODEL_TYPE = "dinov3_vit"

    @classmethod
    def config_from_hf(cls, hf_config):
        intermediate = hf_config.get("intermediate_size")
        hidden = hf_config["hidden_size"]
        mlp_ratio = intermediate / hidden if intermediate else 4.0
        return {
            "patch_size": hf_config.get("patch_size", 16),
            "embed_dim": hidden,
            "depth": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_ratio": mlp_ratio,
            "use_swiglu": hf_config.get("use_gated_mlp", False),
            "num_register_tokens": hf_config.get("num_register_tokens", 4),
            "layer_scale_init": hf_config.get("layerscale_value", 1.0),
            "rope_theta": hf_config.get("rope_theta", 100.0),
            "query_bias": hf_config.get("query_bias", True),
            "key_bias": hf_config.get("key_bias", False),
            "value_bias": hf_config.get("value_bias", True),
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "mlp_bias": hf_config.get("mlp_bias", True),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.dino_v3.convert_dino_v3_hf_to_keras import (
            transfer_dinov3_vit_weights,
        )

        transfer_dinov3_vit_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_swiglu=False,
        num_register_tokens=4,
        layer_scale_init=1.0,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        hidden_act="gelu",
        mlp_bias=True,
        layer_norm_eps=1e-5,
        include_normalization=True,
        normalization_mode="imagenet",
        image_size=224,
        input_tensor=None,
        name="DinoV3ViTModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        if data_format == "channels_first":
            height, width = image_size[1], image_size[2]
        else:
            height, width = image_size[0], image_size[1]

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        rope_cos_np, rope_sin_np = build_rope_2d_cache(
            grid_h, grid_w, embed_dim // num_heads, theta=rope_theta
        )
        rope_cos = ops.convert_to_tensor(rope_cos_np)
        rope_sin = ops.convert_to_tensor(rope_sin_np)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="patch_embed",
        )(x)
        if data_format == "channels_first":
            x = keras.ops.transpose(x, (0, 2, 3, 1))
        x = layers.Reshape((-1, embed_dim))(x)
        x = DinoV3CLSToken(name="cls_token")(x)
        if num_register_tokens > 0:
            x = DinoV3RegisterTokens(
                num_tokens=num_register_tokens, name="register_tokens"
            )(x)

        features = [x]
        for i in range(depth):
            x = dinov3_transformer_block(
                x,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                num_prefix_tokens=num_prefix_tokens,
                rope_theta=rope_theta,
                use_swiglu=use_swiglu,
                layer_scale_init=layer_scale_init,
                block_idx=i,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                query_bias=query_bias,
                key_bias=key_bias,
                value_bias=value_bias,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                layer_norm_eps=layer_norm_eps,
            )
            features.append(x)
        # Final LayerNorm applied to last feature
        features[-1] = layers.LayerNormalization(
            epsilon=layer_norm_eps, axis=-1, name="final_layernorm"
        )(features[-1])

        outputs = features if as_backbone else features[-1]
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.num_register_tokens = num_register_tokens
        self.layer_scale_init = layer_scale_init
        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias
        self.layer_norm_eps = layer_norm_eps
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_swiglu": self.use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "layer_scale_init": self.layer_scale_init,
                "rope_theta": self.rope_theta,
                "query_bias": self.query_bias,
                "key_bias": self.key_bias,
                "value_bias": self.value_bias,
                "hidden_act": self.hidden_act,
                "mlp_bias": self.mlp_bias,
                "layer_norm_eps": self.layer_norm_eps,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoV3ConvNeXtModel(BaseModel):
    """DINOv3 ConvNeXt model.

    When ``as_backbone=False`` (default), returns the final-stage
    feature map ``(B, H/32, W/32, C)``. When ``as_backbone=True``,
    returns the list of per-stage feature maps, suitable for feeding
    into detection / segmentation / depth necks.

    Weights are gated on the model Hub: see
    https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    for license acceptance. Accept the license and set ``HF_TOKEN`` env
    var before calling :meth:`from_weights`.

    Reference:
        - `DINOv3 <https://arxiv.org/abs/2508.10104>`_

    Args:
        as_backbone: If ``True``, output the list of per-stage feature
            maps for use as a backbone. If ``False`` (default), output
            only the final-stage feature map.
        depths: Per-stage block counts.
        projection_dim: Per-stage channel counts.
        include_normalization: Whether to prepend
            image normalization.
        normalization_mode: Normalization preset.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DinoV3ConvNeXtConfig
    HF_MODEL_TYPE = "dinov3_convnext"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "depths": list(hf_config["depths"]),
            "projection_dim": list(hf_config["hidden_sizes"]),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.dino_v3.convert_dino_v3_hf_to_keras import (
            transfer_dinov3_convnext_weights,
        )

        transfer_dinov3_convnext_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        as_backbone=False,
        depths=None,
        projection_dim=None,
        include_normalization=True,
        normalization_mode="imagenet",
        image_size=224,
        input_tensor=None,
        name="DinoV3ConvNeXtModel",
        **kwargs,
    ):
        if depths is None:
            depths = [3, 3, 9, 3]
        if projection_dim is None:
            projection_dim = [96, 192, 384, 768]

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)
        channels_axis = -1 if data_format == "channels_last" else 1

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        features = convnext_backbone_feature(
            x,
            depths=depths,
            projection_dim=projection_dim,
            drop_path_rate=0.0,
            layer_scale_init=1e-6,
            use_conv=True,
            use_grn=False,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=True,
        )

        outputs = features if as_backbone else features[-1]
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.depths = list(depths)
        self.projection_dim = list(projection_dim)
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "depths": self.depths,
                "projection_dim": self.projection_dim,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
