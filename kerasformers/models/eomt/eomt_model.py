import keras
from keras import layers, ops, utils

from kerasformers.base import FunctionalBaseModel
from kerasformers.base.base_model import hf_num_classes
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape

from .eomt_config import EoMTConfig
from .eomt_layers import (
    EoMTAttention,
    EoMTEmbeddings,
    EoMTLayerScale,
    EoMTQueryInjection,
    EoMTStochasticDepth,
)


def eomt_mlp(x, hidden_dim, mlp_ratio=4, block_prefix="layers_0"):
    """Standard two-layer transformer MLP: Dense → GELU → Dense.

    Used in EoMT encoder layers when ``use_swiglu_ffn=False`` (matches the
    standard DINOv2 MLP).

    Args:
        x: Input token sequence of shape ``(B, N, hidden_dim)``.
        hidden_dim: Token / model dimension.
        mlp_ratio: Hidden expansion factor, the intermediate Dense
            width is ``int(hidden_dim * mlp_ratio)``.
        block_prefix: Prefix used to name the inner Dense / Activation
            layers.

    Returns:
        Tensor of shape ``(B, N, hidden_dim)``.
    """
    hidden_features = int(hidden_dim * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{block_prefix}_mlp_fc1")(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_mlp_gelu")(x)
    x = layers.Dense(hidden_dim, name=f"{block_prefix}_mlp_fc2")(x)
    return x


def eomt_swiglu_ffn(x, hidden_dim, mlp_ratio=4, block_prefix="layers_0"):
    """SwiGLU gated feed-forward network.

    Single fused ``Dense`` projects up to ``2 * hidden_features`` and
    the result is split into a value branch and a gate branch: the
    gate is passed through SiLU then multiplied with the value, and a
    final ``Dense`` projects back to ``hidden_dim``. Matches the
    DINOv2-style SwiGLU used by the SwiGLU EoMT variants.

    The hidden width is computed as ``int(hidden_dim * mlp_ratio)``,
    then scaled by ``2/3`` and rounded up to the nearest multiple of 8
    (the canonical DINOv2 / Llama recipe: keeps total params close to
    the standard MLP while being kernel-friendly).

    Args:
        x: Input token sequence of shape ``(B, N, hidden_dim)``.
        hidden_dim: Token / model dimension.
        mlp_ratio: Pre-scaling expansion factor before the ``2/3``
            SwiGLU correction.
        block_prefix: Prefix used to name the inner layers.

    Returns:
        Tensor of shape ``(B, N, hidden_dim)``.
    """
    hidden_features = int(hidden_dim * mlp_ratio)
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    x = layers.Dense(2 * hidden_features, name=f"{block_prefix}_mlp_weights_in")(x)
    x1 = x[..., :hidden_features]
    x2 = x[..., hidden_features:]
    hidden = layers.Activation("silu", name=f"{block_prefix}_mlp_silu")(x1)
    hidden = layers.Multiply(name=f"{block_prefix}_mlp_gate")([hidden, x2])
    return layers.Dense(hidden_dim, name=f"{block_prefix}_mlp_weights_out")(hidden)


def eomt_encoder_layer(
    hidden_states,
    hidden_dim,
    num_heads,
    mlp_ratio=4,
    layerscale_value=1.0,
    drop_path_rate=0.0,
    attention_dropout=0.0,
    use_swiglu_ffn=False,
    layer_norm_eps=1e-6,
    block_prefix="layers_0",
):
    """One pre-LN DINOv2-style encoder block with LayerScale + stochastic depth.

    Structure (matches the EoMT / DINOv2 reference):

    1. Pre-norm → :class:`EoMTAttention` → :class:`EoMTLayerScale` →
       :class:`EoMTStochasticDepth` (if ``drop_path_rate > 0``) → residual.
    2. Pre-norm → :func:`eomt_mlp` *or* :func:`eomt_swiglu_ffn`
       (selected by ``use_swiglu_ffn``) → :class:`EoMTLayerScale` →
       :class:`EoMTStochasticDepth` → residual.

    All sublayer names are deterministic (``{block_prefix}_*``) so the
    source EoMT state-dict can be transferred by name.

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        hidden_states: Input token sequence of shape
            ``(B, N, hidden_dim)``.
        hidden_dim: Token / model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP / SwiGLU expansion ratio.
        layerscale_value: Initial value for the per-channel LayerScale
            gammas on both residual branches.
        drop_path_rate: Stochastic-depth drop probability. ``0`` disables it.
        attention_dropout: Dropout applied inside the attention layer.
        use_swiglu_ffn: If ``True``, use :func:`eomt_swiglu_ffn`;
            otherwise use the standard GELU :func:`eomt_mlp`.
        layer_norm_eps: Epsilon for both pre-norm LayerNorms.
        block_prefix: Prefix used to name every sublayer in this block.

    Returns:
        Tensor of shape ``(B, N, hidden_dim)``.
    """
    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm1"
    )(hidden_states)
    hidden_states = EoMTAttention(
        hidden_dim, num_heads, attention_dropout, name=f"{block_prefix}_attention"
    )(hidden_states)
    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale1"
    )(hidden_states)
    drop_path = (
        EoMTStochasticDepth(drop_path_rate, name=f"{block_prefix}_drop_path")
        if drop_path_rate > 0.0
        else layers.Identity(name=f"{block_prefix}_identity")
    )
    hidden_states = layers.Add(name=f"{block_prefix}_attn_residual")(
        [drop_path(hidden_states), residual]
    )

    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm2"
    )(hidden_states)

    if use_swiglu_ffn:
        hidden_states = eomt_swiglu_ffn(
            hidden_states, hidden_dim, mlp_ratio, block_prefix
        )
    else:
        hidden_states = eomt_mlp(hidden_states, hidden_dim, mlp_ratio, block_prefix)

    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale2"
    )(hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_mlp_residual")(
        [drop_path(hidden_states), residual]
    )
    return hidden_states


def eomt_scale_layer(x, hidden_dim, data_format, block_prefix="upscale_block_0"):
    """One 2× spatial-upscale block for the mask-feature pyramid.

    ``Conv2DTranspose(stride=2)`` → GELU → depthwise 3×3 conv →
    channels-last LayerNorm. The LayerNorm is always run in
    ``channels_last`` (with explicit permutes for the
    ``channels_first`` path) because Keras' :class:`LayerNormalization`
    normalizes the last axis. Each call doubles the spatial resolution.

    Args:
        x: Input feature map. Shape ``(B, H, W, C)`` for
            ``channels_last`` or ``(B, C, H, W)`` for ``channels_first``.
        hidden_dim: Output channel dimension produced by the transposed
            conv.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        block_prefix: Prefix used to name every sublayer.

    Returns:
        Tensor with the same channel layout as ``x``, with spatial
        dimensions doubled (``H, W → 2H, 2W``).
    """
    x = layers.Conv2DTranspose(
        hidden_dim,
        kernel_size=2,
        strides=2,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name=f"{block_prefix}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_gelu")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_conv2",
    )(x)
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm")(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)
    return x


def eomt_scale_block(x, hidden_dim, num_upscale_blocks, data_format):
    """Stack ``num_upscale_blocks`` consecutive :func:`eomt_scale_layer` blocks.

    Each block doubles the spatial resolution, so total upscale factor
    is ``2 ** num_upscale_blocks``. Each block is named
    ``upscale_block_{i}`` so the source EoMT state-dict maps directly.

    Args:
        x: Input feature map from the reshaped patch tokens.
        hidden_dim: Output channel dimension carried through every
            upscale block.
        num_upscale_blocks: How many 2× upscale layers to chain.
        data_format: ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Upscaled feature map with spatial dims multiplied by
        ``2 ** num_upscale_blocks``.
    """
    for i in range(num_upscale_blocks):
        x = eomt_scale_layer(
            x, hidden_dim, data_format=data_format, block_prefix=f"upscale_block_{i}"
        )
    return x


def eomt_mask_head(x, hidden_dim):
    """Per-query mask-token MLP: Dense → GELU → Dense → GELU → Dense.

    Applied to each object-query embedding to produce a ``hidden_dim``
    "mask token". The mask token is then bilinearly broadcast against
    the upscaled spatial features (via einsum) to produce per-query
    mask logits.

    Args:
        x: Query embeddings of shape ``(B, num_queries, hidden_dim)``.
        hidden_dim: Dimension used for every Dense layer in this head.

    Returns:
        Tensor of shape ``(B, num_queries, hidden_dim)``: per-query
        mask tokens.
    """
    x = layers.Dense(hidden_dim, name="mask_head_fc1")(x)
    x = layers.Activation("gelu", name="mask_head_gelu1")(x)
    x = layers.Dense(hidden_dim, name="mask_head_fc2")(x)
    x = layers.Activation("gelu", name="mask_head_gelu2")(x)
    x = layers.Dense(hidden_dim, name="mask_head_fc3")(x)
    return x


def eomt_functional(
    inputs,
    hidden_dim,
    num_hidden_layers,
    num_heads,
    depths,
    num_queries,
    layerscale_value,
    patch_size,
    num_register_tokens,
    mlp_ratio,
    drop_path_rate,
    attention_dropout,
    use_swiglu_ffn,
    layer_norm_eps,
):
    """Build the EoMT encoder graph (no task heads).

    Pipeline: :class:`EoMTEmbeddings` (patch + register + CLS) →
    ``num_hidden_layers`` :func:`eomt_encoder_layer` blocks → final
    ``LayerNormalization``. The learned object queries are injected via
    :class:`EoMTQueryInjection` right before the last ``depths``
    encoder layers, so only those blocks attend over the
    ``(queries, prefix, patches)`` joint sequence.

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        inputs: Image tensor, ``(B, H, W, 3)`` for ``channels_last`` or
            ``(B, 3, H, W)`` for ``channels_first``.
        hidden_dim: Transformer hidden dimension.
        num_hidden_layers: Total number of encoder blocks.
        num_heads: Attention heads per block.
        depths: Number of final encoder blocks that receive the
            object-query injection (queries are prepended just before
            block ``num_hidden_layers - depths``).
        num_queries: Number of learned object queries.
        layerscale_value: Initial value for the per-block LayerScale
            gammas.
        patch_size: ViT patch size.
        num_register_tokens: Number of DINOv2-style register tokens
            inserted between CLS and the patch tokens.
        mlp_ratio: MLP / SwiGLU expansion ratio inside each block.
        drop_path_rate: Stochastic-depth drop rate.
        attention_dropout: Dropout applied inside each attention layer.
        use_swiglu_ffn: If ``True``, blocks use :func:`eomt_swiglu_ffn`;
            otherwise :func:`eomt_mlp`.
        layer_norm_eps: Epsilon for every LayerNorm.

    Returns:
        Sequence-output tensor of shape
        ``(B, num_queries + 1 + num_register_tokens + num_patches,
        hidden_dim)`` after the final LayerNorm. Queries occupy
        ``[:, :num_queries]``, the prefix (CLS + registers) lives at
        ``[:, num_queries : num_queries + 1 + num_register_tokens]``,
        and patches fill the tail.
    """
    data_format = keras.config.image_data_format()
    image_size = inputs.shape[2] if data_format == "channels_first" else inputs.shape[1]

    hidden_states = EoMTEmbeddings(
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        image_size=image_size,
        num_register_tokens=num_register_tokens,
        name="embeddings",
    )(inputs)

    query_injection = EoMTQueryInjection(num_queries, hidden_dim, name="query")
    query_injection_idx = num_hidden_layers - depths

    for i in range(num_hidden_layers):
        if i == query_injection_idx:
            hidden_states = query_injection(hidden_states)

        hidden_states = eomt_encoder_layer(
            hidden_states,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
            block_prefix=f"layers_{i}",
        )

    sequence_output = layers.LayerNormalization(
        epsilon=layer_norm_eps, name="layernorm"
    )(hidden_states)
    return sequence_output


@keras.saving.register_keras_serializable(package="kerasformers")
class EoMTModel(FunctionalBaseModel):
    """EoMT encoder backbone with query injection (no task heads).

    Builds the plain DINOv2-style ViT encoder used by EoMT, including
    the learned object-query injection at the boundary of the final
    ``depths`` encoder layers. Returns the post-LayerNorm sequence
    output of shape
    ``(batch, num_queries + num_prefix + num_patches, hidden_dim)``.
    Pair with :class:`EoMTUniversalSegment` to get the full universal-
    segmentation outputs (class logits + mask logits).

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        hidden_dim: Transformer hidden dimension.
        num_hidden_layers: Total number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        depths: Number of final encoder blocks that receive the
            injected object queries.
        num_queries: Number of learned object queries.
        layerscale_value: Initial value for LayerScale parameters.
        patch_size: Image patch size.
        num_register_tokens: Number of DINOv2-style register tokens.
        mlp_ratio: Expansion ratio for the feedforward network.
        drop_path_rate: Stochastic depth rate.
        attention_dropout: Dropout rate for the attention weights.
        use_swiglu_ffn: Whether to use SwiGLU instead of the GELU MLP.
        layer_norm_eps: Epsilon for layer normalization.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `640`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "eomt"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_dim": hf_config["hidden_size"],
            "num_hidden_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "depths": hf_config["num_blocks"],
            "num_queries": hf_config["num_queries"],
            "layerscale_value": hf_config.get("layerscale_value", 1.0),
            "patch_size": hf_config.get("patch_size", 16),
            "num_register_tokens": hf_config.get("num_register_tokens", 4),
            "mlp_ratio": hf_config.get("mlp_ratio", 4),
            "use_swiglu_ffn": hf_config.get("use_swiglu_ffn", False),
            "image_size": hf_config.get("image_size", 640),
        }

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = EoMTUniversalSegment.from_hf(hf_id, skip_mismatch=skip_mismatch)
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model

    def __init__(
        self,
        hidden_dim=1024,
        num_hidden_layers=24,
        num_heads=16,
        depths=4,
        num_queries=200,
        layerscale_value=1e-5,
        patch_size=16,
        num_register_tokens=4,
        mlp_ratio=4,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        image_size=640,
        input_tensor=None,
        name="EoMTModel",
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

        sequence_output = eomt_functional(
            img_input,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            depths=depths,
            num_queries=num_queries,
            layerscale_value=layerscale_value,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
        )

        super().__init__(inputs=img_input, outputs=sequence_output, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.depths = depths
        self.num_queries = num_queries
        self.layerscale_value = layerscale_value
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "depths": self.depths,
                "num_queries": self.num_queries,
                "layerscale_value": self.layerscale_value,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "mlp_ratio": self.mlp_ratio,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class EoMTUniversalSegment(FunctionalBaseModel):
    """EoMT full universal-segmentation model (encoder + class + mask heads).

    Composes :class:`EoMTModel` and adds the class-prediction head, the
    mask-feature upscale stack, the per-query mask head, and the
    bilinear einsum that produces per-query mask logits. Output is a
    dict with:

    - ``"class_logits"``: ``(batch, num_queries, num_classes + 1)``
    - ``"mask_logits"``: ``(batch, num_queries, H_up, W_up)`` where
      ``H_up = image_size // patch_size * 2^num_upscale_blocks``.

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        num_classes: Number of segmentation classes.
        num_upscale_blocks: Number of 2x upscaling layers applied to
            patch features before mask prediction.
        See :class:`EoMTModel` for the remaining args.
    """

    BASE_MODEL_CONFIG = None
    config_class = EoMTConfig
    # Weights load by Hub repo id, e.g.
    # from_weights("kerasformers/eomt_large_coco_panoptic_640"), via kf_config.json
    # on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "eomt"

    @classmethod
    def config_from_hf(cls, hf_config):
        image_size = hf_config.get("image_size", 640)
        return {
            "hidden_dim": hf_config["hidden_size"],
            "num_hidden_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "depths": hf_config["num_blocks"],
            "num_queries": hf_config["num_queries"],
            "layerscale_value": hf_config.get("layerscale_value", 1.0),
            "patch_size": hf_config.get("patch_size", 16),
            "num_register_tokens": hf_config.get("num_register_tokens", 4),
            "mlp_ratio": hf_config.get("mlp_ratio", 4),
            "use_swiglu_ffn": hf_config.get("use_swiglu_ffn", False),
            "num_upscale_blocks": hf_config.get("num_upscale_blocks", 2),
            "num_classes": hf_num_classes(hf_config),
            "image_size": image_size,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.eomt.convert_eomt_hf_to_keras import (
            transfer_eomt_weights,
        )

        transfer_eomt_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=1024,
        num_hidden_layers=24,
        num_heads=16,
        depths=4,
        num_queries=200,
        num_classes=133,
        layerscale_value=1e-5,
        patch_size=16,
        num_register_tokens=4,
        num_upscale_blocks=2,
        mlp_ratio=4,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        image_size=640,
        input_tensor=None,
        name="EoMTUniversalSegment",
        **kwargs,
    ):
        base = EoMTModel(
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            depths=depths,
            num_queries=num_queries,
            layerscale_value=layerscale_value,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        sequence_output = base.output

        data_format = keras.config.image_data_format()
        image_size = base.image_size
        image_size = image_size[1] if data_format == "channels_first" else image_size[0]
        grid_h = grid_w = image_size // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        query_output = sequence_output[:, :num_queries, :]
        patch_output = sequence_output[:, num_queries + num_prefix_tokens :, :]

        class_logits = layers.Dense(num_classes + 1, name="class_predictor")(
            query_output
        )

        query_mask_tokens = eomt_mask_head(query_output, hidden_dim)
        patch_spatial = ops.reshape(patch_output, (-1, grid_h, grid_w, hidden_dim))
        if data_format == "channels_first":
            patch_spatial = ops.transpose(patch_spatial, (0, 3, 1, 2))

        upscaled_features = eomt_scale_block(
            patch_spatial, hidden_dim, num_upscale_blocks, data_format=data_format
        )

        if data_format == "channels_first":
            mask_logits = ops.einsum(
                "bqc,bchw->bqhw", query_mask_tokens, upscaled_features
            )
        else:
            mask_logits = ops.einsum(
                "bqc,bhwc->bqhw", query_mask_tokens, upscaled_features
            )

        super().__init__(
            inputs=base.input,
            outputs={"class_logits": class_logits, "mask_logits": mask_logits},
            name=name,
            **kwargs,
        )

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.depths = depths
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.layerscale_value = layerscale_value
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.num_upscale_blocks = num_upscale_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "depths": self.depths,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "layerscale_value": self.layerscale_value,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "num_upscale_blocks": self.num_upscale_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
