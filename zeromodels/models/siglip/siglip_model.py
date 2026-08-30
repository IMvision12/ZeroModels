import keras
from keras import initializers, layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_model import strip_functional_graph_keys
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .siglip_config import SigLIPConfig
from .siglip_layers import (
    SigLIPAttention,
    SigLIPLogitScaleBias,
    SigLIPPositionEmbedding,
    SigLIPPositionIDs,
    SigLIPProbe,
)

# The full SigLIPModel plus its task heads (vision / text / zero-shot / classify)
# all load from one repo per variant, whose zm_config.json declares the canonical
# SigLIPZeroShotClassify. Listing them as siblings lets any head load that repo.
SIGLIP_HUB_SIBLINGS = frozenset(
    {
        "SigLIPModel",
        "SigLIPVisionModel",
        "SigLIPTextModel",
        "SigLIPZeroShotClassify",
        "SigLIPImageClassify",
    }
)


def siglip_encoder(
    inputs,
    hidden_dim,
    num_heads,
    mlp_dim,
    layer_norm_epsilon=1e-6,
    name="encoder_layer",
):
    """One pre-LN SigLIP transformer block (LN → MHSA → Add → LN → MLP → Add).

    Shared building block for both the vision and text encoders. All
    sublayer names are deterministic, ``{name}_*``, so the
    corresponding pretrained weights can be transferred by name during
    checkpoint conversion.

    Args:
        inputs: Input token sequence of shape ``(B, L, hidden_dim)``.
        hidden_dim: Hidden / model dimension. Must be divisible by
            ``num_heads``.
        num_heads: Attention head count.
        mlp_dim: MLP hidden dimension (typically
            ``4 * hidden_dim``).
        layer_norm_epsilon: Epsilon for both pre-norm LayerNorms.
            Defaults to ``1e-6``.
        name: Prefix used for every sublayer name (e.g.
            ``"vision_model_encoder_layers_3"``).

    Returns:
        Output tensor of shape ``(B, L, hidden_dim)``.

    Raises:
        ValueError: If ``hidden_dim`` is not divisible by ``num_heads``.
    """

    if hidden_dim % num_heads != 0:
        raise ValueError(
            "`hidden_dim` must be divisible by `num_heads`. "
            f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
        )

    residual1 = inputs
    x = layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name=f"{name}_layernorm_1"
    )(inputs)

    x = SigLIPAttention(
        num_heads,
        hidden_dim // num_heads,
        combined_qkv=False,
        block_prefix=f"{name}_self_attn",
    )(x)

    x = layers.Add(name=f"{name}_add_1")([residual1, x])

    residual2 = x
    x = layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name=f"{name}_layernorm_2"
    )(x)

    x = layers.Dense(
        mlp_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_1",
    )(x)
    x = keras.activations.gelu(x, approximate=True)

    x = layers.Dense(
        hidden_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_2",
    )(x)

    outputs = layers.Add(name=f"{name}_add_2")([residual2, x])

    return outputs


def siglip_attention_pooling(
    inputs,
    hidden_dim,
    mlp_dim,
    num_heads,
    layer_norm_epsilon=1e-6,
    name="attention_pooling",
):
    """SigLIP attention-pool head: learnable probe ↔ sequence cross-attention.

    Pipeline: build learnable probe tokens → cross-attention (probes as
    Q, input as K/V) → LN → MLP → residual → take the first probe.
    Replaces CLS-token pooling on the vision side of SigLIP.

    Args:
        inputs: Token sequence of shape ``(B, L, hidden_dim)``.
        hidden_dim: Hidden / model dimension.
        mlp_dim: MLP hidden dimension.
        num_heads: Cross-attention head count.
        layer_norm_epsilon: Epsilon for the LayerNorm. Defaults to ``1e-6``.
        name: Prefix used for every sublayer name.

    Returns:
        Pooled tensor of shape ``(B, hidden_dim)``.
    """
    probe_layer = SigLIPProbe(hidden_dim, name=f"{name}_probe")
    probes = probe_layer(inputs)

    hidden_states = SigLIPAttention(
        num_heads,
        hidden_dim // num_heads,
        combined_qkv=True,
        block_prefix=f"{name}_attention",
    )(probes, key=inputs, value=inputs)

    residuals = hidden_states
    x = layers.LayerNormalization(epsilon=layer_norm_epsilon, name=f"{name}_layernorm")(
        hidden_states
    )

    x = layers.Dense(
        mlp_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_1",
    )(x)
    x = keras.activations.gelu(x, approximate=True)

    x = layers.Dense(
        hidden_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_2",
    )(x)

    x = layers.Add(name=f"{name}_add")([residuals, x])

    outputs = x[:, 0]
    return outputs


def siglip_vision_embedding(
    inputs,
    hidden_dim,
    patch_size,
    image_size,
    data_format=None,
    name="vision_embedding",
):
    """Patch-embed + learned positional embeddings for the SigLIP vision tower.

    Pipeline: patch ``Conv2D`` (stride = ``patch_size``) → flatten to
    a token sequence → add 1-D learned positional embeddings. No CLS
    token is prepended (SigLIP uses attention pooling instead).

    Args:
        inputs: Image tensor of shape ``(B, H, W, C)`` for
            ``channels_last`` or ``(B, C, H, W)`` for ``channels_first``.
        hidden_dim: Per-patch embedding dimension.
        patch_size: Side length of each square patch.
        image_size: Side length of the (square) input image. Must be
            divisible by ``patch_size``.
        data_format: ``"channels_last"`` / ``"channels_first"``.
            ``None`` uses the global default.
        name: Prefix used for every sublayer name.

    Returns:
        Tensor of shape ``(B, (image_size // patch_size)**2, hidden_dim)``.
    """

    num_positions = (image_size // patch_size) ** 2
    num_patches_per_side = image_size // patch_size

    patch_embeddings = layers.Conv2D(
        hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        kernel_initializer=initializers.LecunNormal(),
        data_format=data_format,
        name=f"{name}_patch_embedding_conv",
    )(inputs)

    if data_format == "channels_last":
        patch_embeddings = layers.Reshape(
            (-1, hidden_dim),
        )(patch_embeddings)
    else:
        patch_embeddings = layers.Reshape(
            (hidden_dim, -1),
        )(patch_embeddings)
        patch_embeddings = layers.Permute(
            (2, 1),
        )(patch_embeddings)

    position_ids = SigLIPPositionIDs(
        grid_h=num_patches_per_side,
        grid_w=num_patches_per_side,
        use_2d_positions=False,
        name=f"{name}_position_ids",
    )(inputs)

    position_embeddings = SigLIPPositionEmbedding(
        max_positions=num_positions,
        embed_dim=hidden_dim,
        embeddings_initializer=initializers.RandomNormal(stddev=hidden_dim**-0.5),
        name=f"{name}_position_embedding",
    )(position_ids)

    outputs = layers.Add(name=f"{name}_add_embeddings")(
        [patch_embeddings, position_embeddings]
    )

    return outputs


def siglip_vision_features(
    inputs,
    patch_size,
    hidden_dim,
    num_layers,
    num_heads,
    mlp_dim,
    layer_norm_epsilon=1e-6,
    data_format=None,
):
    """Pre-pool SigLIP vision encoder output (patch embed + N encoders + LN).

    Args:
        inputs: Image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
            Height and width must be equal.
        patch_size: Edge length of each square patch.
        hidden_dim: Vision-side hidden dimension (must be divisible by
            ``num_heads``).
        num_layers: Number of stacked transformer encoder layers.
        num_heads: Number of attention heads per encoder layer.
        mlp_dim: Per-encoder feed-forward hidden dimension.
        layer_norm_epsilon: Epsilon for every LayerNorm. Defaults to 1e-6.
        data_format: ``"channels_last"`` or ``"channels_first"``. ``None``
            uses the global default.

    Returns:
        Full token sequence ``(B, num_patches, hidden_dim)`` after the final
        LayerNorm: equivalent to the reference vision encoder's last hidden state.
    """
    input_shape = inputs.shape
    if data_format == "channels_last":
        height, width = input_shape[1], input_shape[2]
    else:
        height, width = input_shape[2], input_shape[3]

    if height != width:
        raise ValueError(
            "`siglip_vision_features` expects the height and width to be the "
            f"same in input shape. Received: input_shape={input_shape}"
        )

    x = siglip_vision_embedding(
        inputs,
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        image_size=height,
        data_format=data_format,
        name="vision_model_embeddings",
    )
    for i in range(num_layers):
        x = siglip_encoder(
            x,
            hidden_dim,
            num_heads,
            mlp_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            name=f"vision_model_encoder_layers_{i}",
        )
    return layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name="vision_model_final_layernorm"
    )(x)


def siglip_vision_backbone(
    inputs,
    patch_size,
    hidden_dim,
    num_layers,
    num_heads,
    mlp_dim,
    layer_norm_epsilon=1e-6,
    data_format=None,
):
    """SigLIP vision encoder: features + attention pooling, no projection.

    Vision-encoder forward pass. Pipeline:
    :func:`siglip_vision_features` → :func:`siglip_attention_pooling`.

    Args:
        inputs: Image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        patch_size: Edge length of each square patch.
        hidden_dim: Vision-side hidden dimension.
        num_layers: Number of stacked transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_dim: Per-encoder feed-forward hidden dimension.
        layer_norm_epsilon: Epsilon for every LayerNorm. Defaults to 1e-6.
        data_format: ``"channels_last"`` or ``"channels_first"``. ``None``
            uses the global default.

    Returns:
        Tuple ``(last_hidden_state, pooler_output)`` of shapes
        ``(B, num_patches, hidden_dim)`` and ``(B, hidden_dim)``.
    """
    last_hidden_state = siglip_vision_features(
        inputs,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        layer_norm_epsilon=layer_norm_epsilon,
        data_format=data_format,
    )
    pooler_output = siglip_attention_pooling(
        last_hidden_state,
        hidden_dim,
        mlp_dim,
        num_heads,
        layer_norm_epsilon,
        name="vision_model_head",
    )
    return last_hidden_state, pooler_output


def siglip_text_embedding(
    inputs,
    vocab_size,
    sequence_length,
    embed_dim,
    embeddings_initializer="normal",
    mask_zero=False,
    name="text_embedding",
):
    """Token + learned positional embeddings for the SigLIP text tower.

    Looks up token IDs in an :class:`Embedding`, adds a learned
    positional embedding for each position, and returns their sum.

    Args:
        inputs: Integer token-id tensor of shape ``(B, sequence_length)``.
        vocab_size: Size of the token vocabulary.
        sequence_length: Maximum sequence length / positional-table size.
        embed_dim: Token / positional embedding dimension.
        embeddings_initializer: Initializer for both embedding tables.
            Defaults to ``"normal"``.
        mask_zero: Whether the token embedding should treat ``0`` as a
            padding ID and emit a mask. Defaults to ``False``.
        name: Prefix used for every sublayer name.

    Returns:
        Tensor of shape ``(B, sequence_length, embed_dim)``.
    """
    embedded_tokens = layers.Embedding(
        vocab_size,
        embed_dim,
        embeddings_initializer=embeddings_initializer,
        mask_zero=mask_zero,
        name=f"{name}_token_embedding",
    )(inputs)

    position_ids = SigLIPPositionIDs(
        grid_h=1,
        grid_w=sequence_length,
        use_2d_positions=False,
        name=f"{name}_position_ids",
    )(inputs)

    embedded_positions = SigLIPPositionEmbedding(
        max_positions=sequence_length,
        embed_dim=embed_dim,
        embeddings_initializer=embeddings_initializer,
        name=f"{name}_position_embedding",
    )(position_ids)

    outputs = layers.Add(name=f"{name}_add_embeddings")(
        [embedded_tokens, embedded_positions]
    )

    return outputs


def siglip_text_backbone(
    inputs,
    vocab_size,
    embed_dim,
    hidden_dim,
    num_layers,
    num_heads,
    mlp_dim,
    layer_norm_epsilon=1e-6,
    max_seq_len=64,
    projection_dim=None,
):
    """SigLIP text encoder: embeddings + encoder stack + final LN + head.

    Text-encoder forward pass. Returns the post-LN encoder
    output as ``last_hidden_state`` and the last-token projection through
    the ``text_model_head`` Dense as ``pooler_output``.

    Args:
        inputs: Token tensor of shape ``(B, sequence_length)``.
        vocab_size: Size of the token vocabulary.
        embed_dim: Dimension of the input token embeddings.
        hidden_dim: Hidden dimension of the transformer (must be divisible
            by ``num_heads``).
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads per encoder.
        mlp_dim: Feed-forward hidden dimension.
        layer_norm_epsilon: Epsilon for every LayerNorm. Defaults to 1e-6.
        max_seq_len: Positional-embedding table length. Defaults to 64.
        projection_dim: Output dim of the head Dense. Defaults to
            ``hidden_dim``.

    Returns:
        Tuple ``(last_hidden_state, pooler_output)`` of shapes
        ``(B, sequence_length, hidden_dim)`` and ``(B, projection_dim)``.
    """
    projection_dim = projection_dim or hidden_dim

    x = siglip_text_embedding(
        inputs,
        vocab_size=vocab_size,
        sequence_length=max_seq_len,
        embed_dim=embed_dim,
        name="text_model_embeddings",
    )

    for i in range(num_layers):
        x = siglip_encoder(
            x,
            hidden_dim,
            num_heads,
            mlp_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            name=f"text_model_encoder_layers_{i}",
        )

    last_hidden_state = layers.LayerNormalization(
        epsilon=layer_norm_epsilon,
        name="text_model_final_layernorm",
    )(x)

    last_token = last_hidden_state[:, -1, :]
    pooler_output = layers.Dense(
        projection_dim,
        kernel_initializer=initializers.LecunNormal(),
        name="text_model_head",
    )(last_token)

    return last_hidden_state, pooler_output


def siglip_head(vision_embedding, text_embedding):
    """L2-normalize embeddings and produce scaled+biased similarity logits.

    Standard SigLIP sigmoid head. L2-normalize both sides, compute the
    pairwise cosine similarity matrix, then apply the learnable
    :class:`SigLIPLogitScaleBias` (``scale * sim + bias``). Returns the
    ``(B, B)`` image-vs-text logit matrix together with its transpose.

    Args:
        vision_embedding: Image embedding tensor ``(B, embed_dim)``.
        text_embedding: Text embedding tensor ``(B, embed_dim)``.

    Returns:
        Tuple ``(image_logits, text_logits)``, each of shape ``(B, B)``.
        ``image_logits[i, j]`` is the temperature-scaled cosine
        similarity between image ``i`` and text ``j``; ``text_logits``
        is its transpose.
    """
    vision_norms = ops.sqrt(
        ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
    )
    text_norms = ops.sqrt(ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True))
    norm_vision = ops.divide(vision_embedding, vision_norms)
    norm_text = ops.divide(text_embedding, text_norms)

    similarity_matrix = ops.matmul(norm_text, ops.transpose(norm_vision))

    text_logits = SigLIPLogitScaleBias()(similarity_matrix)
    image_logits = ops.transpose(text_logits)

    return image_logits, text_logits


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPVisionModel(BaseModel):
    """SigLIP vision tower as a standalone model.

    Patch embedding +
    transformer stack + final LayerNorm, followed by the attention-
    pooling head. Use this when you only need image features and don't
    want to instantiate the text tower.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["last_hidden_state"]   # (B, num_patches, vision_hidden_dim)
        out["pooler_output"]       # (B, vision_hidden_dim): attention-pooled

    Construction:

    >>> SigLIPVisionModel.from_weights("siglip_base_p16_224")
    >>> SigLIPVisionModel.from_weights("hf:google/siglip-base-patch16-224")

    Loading from a full SigLIP checkpoint silently ignores the
    text-tower and ``logit_scale`` / ``logit_bias`` entries.

    Reference:
        - `Sigmoid Loss for Language Image Pre-Training
          <https://arxiv.org/abs/2303.15343>`_

    Args:
        image_size: Input image specification. Accepts an
            integer ``N`` (builds an ``N x N x 3`` square input), a
            2-tuple ``(H, W)``, or a 3-tuple in the active data format's
            order. Defaults to ``224``.
        patch_size: ViT patch edge in pixels. Defaults to ``16``.
        vision_hidden_dim: Vision encoder hidden dimension.
            Defaults to ``768``.
        vision_num_layers: Number of transformer encoder layers.
            Defaults to ``12``.
        vision_num_heads: Number of self-attention heads per encoder.
            Defaults to ``12``.
        vision_mlp_dim: MLP hidden dimension inside each
            encoder. Defaults to ``3072``.
        input_tensor: Optional pre-existing Keras tensor to use as the
            ``images`` input.
        name: Model name. Defaults to ``"SigLIPVisionModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = SigLIPConfig
    HUB_REPO_SIBLINGS = SIGLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIPModel

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full model; build it
        # from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_dim=3072,
        input_tensor=None,
        name="SigLIPVisionModel",
        **kwargs,
    ):
        for k in (
            "vocab_size",
            "embed_dim",
            "text_hidden_dim",
            "text_num_layers",
            "text_num_heads",
            "text_mlp_dim",
            "max_seq_len",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            images_input = layers.Input(shape=image_size, name="images")
        else:
            images_input = input_tensor

        last_hidden_state, pooler_output = siglip_vision_backbone(
            images_input,
            patch_size=patch_size,
            hidden_dim=vision_hidden_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            mlp_dim=vision_mlp_dim,
            data_format=data_format,
        )

        super().__init__(
            inputs=images_input,
            outputs={
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
            },
            name=name,
            **kwargs,
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.input_tensor = input_tensor

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPTextModel(BaseModel):
    """SigLIP text tower as a standalone model.

    Token + positional
    embedding, transformer stack, final LayerNorm, and last-token
    projection through the ``text_model_head`` Dense. Use this when
    you only need text features and don't want to instantiate the
    vision tower.

    Output dict:

    .. code-block:: python

        out = model(token_ids)
        out["last_hidden_state"]   # (B, sequence_length, text_hidden_dim)
        out["pooler_output"]       # (B, embed_dim): last-token + Dense head

    Construction:

    >>> SigLIPTextModel.from_weights("siglip_base_p16_224")
    >>> SigLIPTextModel.from_weights("hf:google/siglip-base-patch16-224")

    Loading from a full SigLIP checkpoint silently ignores the
    vision-tower and ``logit_scale`` / ``logit_bias`` entries.

    Reference:
        - `Sigmoid Loss for Language Image Pre-Training
          <https://arxiv.org/abs/2303.15343>`_

    Args:
        vocab_size: Token vocabulary size. Defaults to ``32000``
            for SigLIP v1 (BERT-style); SigLIP 2 uses ``256000``.
        embed_dim: Output dim of the ``text_model_head`` projection
            (i.e. shared joint embedding dim). Defaults to ``768``.
        text_hidden_dim: Text encoder hidden dimension. Defaults to ``768``.
        text_num_layers: Number of transformer encoder layers.
            Defaults to ``12``.
        text_num_heads: Number of self-attention heads per encoder.
            Defaults to ``12``.
        text_mlp_dim: MLP hidden dimension inside each
            encoder. Defaults to ``3072``.
        max_seq_len: Positional-embedding table length / max
            input length. Defaults to ``64``.
        input_tensor: Optional pre-existing Keras tensor to use as the
            ``token_ids`` input.
        name: Model name. Defaults to ``"SigLIPTextModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = SigLIPConfig
    HUB_REPO_SIBLINGS = SIGLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIPModel

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full model; build it
        # from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        vocab_size=32000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        max_seq_len=64,
        input_tensor=None,
        name="SigLIPTextModel",
        **kwargs,
    ):
        for k in (
            "image_size",
            "patch_size",
            "vision_hidden_dim",
            "vision_num_layers",
            "vision_num_heads",
            "vision_mlp_dim",
        ):
            kwargs.pop(k, None)

        if input_tensor is None:
            token_ids_input = layers.Input(shape=(max_seq_len,), name="token_ids")
        else:
            token_ids_input = input_tensor

        last_hidden_state, pooler_output = siglip_text_backbone(
            token_ids_input,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            mlp_dim=text_mlp_dim,
            max_seq_len=max_seq_len,
        )

        super().__init__(
            inputs=token_ids_input,
            outputs={
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
            },
            name=name,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.max_seq_len = max_seq_len
        self.input_tensor = input_tensor

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "max_seq_len": self.max_seq_len,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPModel(BaseModel):
    """SigLIP dual encoder (no contrastive head).

    Composes :class:`SigLIPVisionModel` and :class:`SigLIPTextModel`
    around a shared input pair, and returns the towers'
    ``pooler_output`` as ``image_embeddings`` / ``text_embeddings``. No
    L2-norm or ``logit_scale`` / ``logit_bias`` is applied: for the
    full zero-shot head use :class:`SigLIPZeroShotClassify`. For
    supervised classification use :class:`SigLIPImageClassify`.

    The two sub-models are exposed as ``model.vision_model`` and
    ``model.text_model`` so the towers can be re-used directly.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ...})
        out["image_embeddings"]   # (B, vision_hidden_dim)
        out["text_embeddings"]    # (B, embed_dim)

    Construction:

    >>> SigLIPModel.from_weights("siglip_base_p16_224")
    >>> SigLIPModel.from_weights("hf:google/siglip-base-patch16-224")

    Reference:
        - `Sigmoid Loss for Language Image Pre-Training
          <https://arxiv.org/abs/2303.15343>`_

    Args:
        image_size: Input image specification. Accepts an
            integer ``N`` (builds an ``N x N x 3`` square input), a
            2-tuple ``(H, W)``, or a 3-tuple in the active data format's
            order. Defaults to ``224``.
        patch_size: ViT patch edge in pixels. Defaults to ``16``.
        vision_hidden_dim: Vision encoder hidden dimension.
            Defaults to ``768``.
        vision_num_layers: Vision encoder depth. Defaults to ``12``.
        vision_num_heads: Vision encoder attention heads. Defaults to ``12``.
        vision_mlp_dim: Vision encoder MLP hidden dim.
            Defaults to ``3072``.
        vocab_size: Tokenizer vocabulary size. Defaults to ``32000``.
        embed_dim: Shared joint embedding dim (= output dim of the
            text head). Defaults to ``768``.
        text_hidden_dim: Text encoder hidden dimension. Defaults to ``768``.
        text_num_layers: Text encoder depth. Defaults to ``12``.
        text_num_heads: Text encoder attention heads. Defaults to ``12``.
        text_mlp_dim: Text encoder MLP hidden dim.
            Defaults to ``3072``.
        max_seq_len: Positional-embedding table length / max
            text input length. Defaults to ``64``.
        input_tensor: Optional dict of pre-existing Keras tensors with
            keys ``"images"`` and ``"token_ids"``.
        name: Model name. Defaults to ``"SigLIPModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = SigLIPConfig
    HUB_REPO_SIBLINGS = SIGLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "image_size": vc.get("image_size", 224),
            "patch_size": vc.get("patch_size", 16),
            "vision_hidden_dim": vc.get("hidden_size", 768),
            "vision_num_layers": vc.get("num_hidden_layers", 12),
            "vision_num_heads": vc.get("num_attention_heads", 12),
            "vision_mlp_dim": vc.get("intermediate_size", 3072),
            "vocab_size": tc.get("vocab_size", 32000),
            "embed_dim": tc.get("hidden_size", 768),
            "text_hidden_dim": tc.get("hidden_size", 768),
            "text_num_layers": tc.get("num_hidden_layers", 12),
            "text_num_heads": tc.get("num_attention_heads", 12),
            "text_mlp_dim": tc.get("intermediate_size", 3072),
            "max_seq_len": tc.get("max_position_embeddings", 64),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_dim=3072,
        vocab_size=32000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        max_seq_len=64,
        input_tensor=None,
        name="SigLIPModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images")
            if images_input is None:
                images_input = layers.Input(shape=image_size, name="images")
            token_ids_input = input_tensor.get("token_ids")
            if token_ids_input is None:
                token_ids_input = layers.Input(shape=(max_seq_len,), name="token_ids")
        else:
            images_input = layers.Input(shape=image_size, name="images")
            token_ids_input = layers.Input(shape=(max_seq_len,), name="token_ids")

        vision_model = SigLIPVisionModel(
            image_size=image_size,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_mlp_dim=vision_mlp_dim,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )
        text_model = SigLIPTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_mlp_dim=text_mlp_dim,
            max_seq_len=max_seq_len,
            input_tensor=token_ids_input,
            name=f"{name}_text_tower",
        )

        outputs = {
            "image_embeddings": vision_model.output["pooler_output"],
            "text_embeddings": text_model.output["pooler_output"],
        }
        inputs = {"images": images_input, "token_ids": token_ids_input}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vision_model = vision_model
        self.text_model = text_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.max_seq_len = max_seq_len
        self.input_tensor = input_tensor

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "max_seq_len": self.max_seq_len,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPZeroShotClassify(BaseModel):
    """SigLIP + sigmoid-similarity head for zero-shot classification / retrieval.

    Composes :class:`SigLIPModel` and adds the standard SigLIP head:
    L2-normalize both sides, compute the pairwise cosine-similarity
    matrix, then apply a learnable ``logit_scale`` and ``logit_bias``
    (see :class:`SigLIPLogitScaleBias`). Output is the ``(B, B)``
    image-vs-text similarity logits, which sigmoid to per-pair
    matching probabilities when ``text_*`` inputs are class-name
    prompts.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ...})
        out["image_logits"]   # (B, B): image[i] vs text[j], scaled+biased
        out["text_logits"]    # (B, B): transpose of image_logits

    Construction:

    >>> SigLIPZeroShotClassify.from_weights("siglip_base_p16_224")
    >>> SigLIPZeroShotClassify.from_weights("hf:google/siglip-base-patch16-224")

    Reference:
        - `Sigmoid Loss for Language Image Pre-Training
          <https://arxiv.org/abs/2303.15343>`_

    Args (identical to :class:`SigLIPModel`):
        image_size, patch_size, vision_hidden_dim,
        vision_num_layers, vision_num_heads, vision_mlp_dim,
        vocab_size, embed_dim, text_hidden_dim, text_num_layers,
        text_num_heads, text_mlp_dim, max_seq_len,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = SigLIPConfig
    HUB_REPO_SIBLINGS = SIGLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_dim=3072,
        vocab_size=32000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        max_seq_len=64,
        input_tensor=None,
        name="SigLIPZeroShotClassify",
        **kwargs,
    ):
        base = SigLIPModel(
            image_size=image_size,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_mlp_dim=vision_mlp_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_mlp_dim=text_mlp_dim,
            max_seq_len=max_seq_len,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = siglip_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.image_size = base.image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.max_seq_len = max_seq_len
        self.input_tensor = input_tensor

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "max_seq_len": self.max_seq_len,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPImageClassify(BaseModel):
    """SigLIP vision tower + linear image-classification head.

    Composes :class:`SigLIPVisionModel`, mean-pools the
    ``last_hidden_state`` patch tokens (the attention-pooling head is
    bypassed for classification), and applies a single linear
    classifier producing ``num_classes`` logits.

    .. code-block:: python

        model = SigLIPImageClassify.from_weights(
            "hf:<user>/siglip-finetune-imagenet"
        )
        logits = model(images)              # (B, num_classes)

    The vision tower is exposed as ``model.vision_model`` so it can be
    re-used directly (for feature extraction).

    Reference:
        - `Sigmoid Loss for Language Image Pre-Training
          <https://arxiv.org/abs/2303.15343>`_

    Args:
        num_classes: Number of output classes. Defaults to ``1000``.
        image_size: Input image specification. Accepts an
            integer ``N`` (builds an ``N x N x 3`` square input), a
            2-tuple ``(H, W)``, or a 3-tuple in the active data format's
            order. Defaults to ``224``.
        patch_size: ViT patch edge in pixels. Defaults to ``16``.
        vision_hidden_dim: Vision encoder hidden dimension.
            Defaults to ``768``.
        vision_num_layers: Vision encoder depth. Defaults to ``12``.
        vision_num_heads: Vision encoder attention heads. Defaults to ``12``.
        vision_mlp_dim: Vision encoder MLP hidden dim.
            Defaults to ``3072``.
        input_tensor: Optional pre-existing Keras tensor to use as the
            ``images`` input.
        name: Model name. Defaults to ``"SigLIPImageClassify"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = SigLIPConfig
    HUB_REPO_SIBLINGS = SIGLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        """Base model class to warm-start the vision encoder from.

        Subclasses (e.g. :class:`SigLIP2ImageClassify`) override this to
        point at their matching encoder-only model.
        """
        return SigLIPModel

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full model; build it
        # from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from zeromodels.base.base_model import hf_num_classes

        config = SigLIPModel.config_from_hf(hf_config)
        try:
            config["num_classes"] = hf_num_classes(hf_config)
        except KeyError:
            pass
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_image_classify_weights,
        )

        transfer_siglip_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_classes=1000,
        image_size=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_dim=3072,
        input_tensor=None,
        name="SigLIPImageClassify",
        **kwargs,
    ):
        for k in (
            "vocab_size",
            "embed_dim",
            "text_hidden_dim",
            "text_num_layers",
            "text_num_heads",
            "text_mlp_dim",
            "max_seq_len",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            images_input = layers.Input(shape=image_size, name="images")
        else:
            images_input = input_tensor

        vision_model = SigLIPVisionModel(
            image_size=image_size,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_mlp_dim=vision_mlp_dim,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )
        encoded = vision_model.output["last_hidden_state"]

        pooled = ops.mean(encoded, axis=1)
        logits = layers.Dense(num_classes, name="classifier")(pooled)

        super().__init__(inputs=images_input, outputs=logits, name=name, **kwargs)

        self.vision_model = vision_model
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.input_tensor = input_tensor

    def get_config(self):
        config = strip_functional_graph_keys(super().get_config())
        config.update(
            {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
