import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .clip_config import CLIPConfig
from .clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    CLIPTextModelEmbedding,
    CLIPVisionModelEmbedding,
)

# The full CLIPModel plus its task heads (vision / text / embed / zero-shot /
# classify) all load from one repo per variant, whose zm_config.json declares the
# canonical CLIPModel. Listing them as siblings lets any head load that repo.
CLIP_HUB_SIBLINGS = frozenset(
    {
        "CLIPModel",
        "CLIPVisionModel",
        "CLIPTextModel",
        "CLIPImageEmbed",
        "CLIPTextEmbed",
        "CLIPZeroShotClassify",
        "CLIPImageClassify",
    }
)


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


def activation_layer(hidden_act):
    if hidden_act == "quick_gelu":
        return keras.layers.Lambda(quick_gelu)
    return keras.layers.Activation(hidden_act)


def residual_attention_block(
    x,
    proj_dim,
    num_heads,
    layer_name_prefix,
    layer_idx,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    layer_prefix = f"{layer_name_prefix}_{layer_idx}"

    ln_1_output = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{layer_prefix}_layernorm_1"
    )(x)

    mask = None
    if causal_attention_mask is not None:
        mask = ops.cast(causal_attention_mask, dtype=x.dtype)
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, dtype=x.dtype)
        mask = (
            ops.add(mask, attention_mask)
            if causal_attention_mask is not None
            else attention_mask
        )

    attention_output = CLIPAttention(
        proj_dim=proj_dim,
        num_heads=num_heads,
        name_prefix=f"{layer_prefix}_attn",
    )(ln_1_output, attention_mask=mask)[0]

    residual_1 = keras.layers.Add()([x, attention_output])
    ln_2_output = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{layer_prefix}_layernorm_2"
    )(residual_1)

    mlp_intermediate_size = int(proj_dim * mlp_ratio)
    mlp_output = keras.layers.Dense(
        mlp_intermediate_size, name=f"{layer_prefix}_dense_1"
    )(ln_2_output)
    mlp_output = activation_layer(hidden_act)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    return keras.layers.Add()([residual_1, mlp_output])


def clip_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    x = inputs
    for i in range(num_layers):
        x = residual_attention_block(
            x,
            proj_dim=width,
            num_heads=heads,
            layer_name_prefix=layer_prefix,
            layer_idx=i,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
    return x


def clip_vision_features(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    data_format="channels_last",
):
    patch_embeddings = keras.layers.Conv2D(
        filters=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="vision_model_conv",
    )(inputs)

    embeddings = CLIPVisionModelEmbedding(
        width, input_resolution, patch_size, data_format, name="vision_model_embeddings"
    )(patch_embeddings)

    x = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="vision_model_layernorm_1"
    )(embeddings)
    return clip_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
    )


def clip_vision_backbone(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    data_format="channels_last",
):
    last_hidden_state = clip_vision_features(
        inputs,
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        num_layers=num_layers,
        heads=heads,
        vision_mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        data_format=data_format,
    )
    class_token = keras.layers.Lambda(lambda x: x[:, 0, :], name="extract_token")(
        last_hidden_state
    )
    pooler_output = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="vision_model_layernorm_2"
    )(class_token)
    return last_hidden_state, pooler_output


def clip_text_backbone(
    inputs,
    attention_mask,
    text_hidden_dim,
    text_num_layers,
    text_num_heads,
    vocab_size,
    max_seq_len,
    text_mlp_ratio,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    x = CLIPTextModelEmbedding(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=text_hidden_dim,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.cast(
        ops.triu(ops.ones((max_seq_len, max_seq_len)), k=1), "float32"
    ) * (-1e8)

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, max_seq_len))
    expanded_mask = ops.repeat(expanded_mask, max_seq_len, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = clip_encoder(
        x,
        width=text_hidden_dim,
        num_layers=text_num_layers,
        heads=text_num_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        layer_prefix="text_model_encoder",
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
    )

    last_hidden_state = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="text_model_layernorm"
    )(encoded_output)

    indices = ops.argmax(inputs, axis=-1)
    one_hot_indices = ops.one_hot(indices, max_seq_len)
    pooler_output = ops.einsum("bi,bij->bj", one_hot_indices, last_hidden_state)

    return last_hidden_state, pooler_output


def clip_head(image_embeddings, text_embeddings):
    image_embeddings = image_embeddings / ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), axis=-1, keepdims=True)
    )
    text_embeddings = text_embeddings / ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), axis=-1, keepdims=True)
    )
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    return logit_scale_layer([image_embeddings, text_embeddings])


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPVisionModel(BaseModel):
    """CLIP vision tower as a standalone model: no text encoder, no projection.

    The patch-embedding +
    transformer stack from CLIP, ending at the post-encoder LayerNorm.
    Use this when you only need image features and don't want to
    instantiate the text tower or carry the ``visual_projection`` Dense.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["last_hidden_state"]   # (B, num_patches + 1, vision_hidden_dim)
        out["pooler_output"]       # (B, vision_hidden_dim): post-LN CLS token

    Construction:

    >>> CLIPVisionModel.from_weights("clip_vit_base_16")
    >>> CLIPVisionModel.from_weights("hf:openai/clip-vit-base-patch16")

    Loading from a full CLIP checkpoint silently ignores the text-tower,
    ``visual_projection``, and ``logit_scale`` entries.

    Args:
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)``, or a 3-tuple in the active data format's order.
        vision_num_layers: ViT encoder depth.
        vision_hidden_dim: ViT hidden dim.
        vision_patch_size: ViT patch size.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
        hidden_act: MLP activation name.
        layer_norm_eps: Epsilon for every LayerNorm.
        input_tensor: Optional pre-existing input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full CLIPModel; build
        # it from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        vision_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPVisionModel",
        **kwargs,
    ):
        for k in (
            "embed_dim",
            "max_seq_len",
            "vocab_size",
            "text_hidden_dim",
            "text_num_heads",
            "text_num_layers",
            "text_mlp_ratio",
        ):
            kwargs.pop(k, None)

        vision_num_heads = vision_hidden_dim // 64
        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)
        if data_format == "channels_first":
            image_size = input_shape[1]
        else:
            image_size = input_shape[0]

        if input_tensor is None:
            images_input = layers.Input(shape=input_shape, name="images")
        else:
            images_input = input_tensor

        last_hidden_state, pooler_output = clip_vision_backbone(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_hidden_dim,
            num_layers=vision_num_layers,
            heads=vision_num_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
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
        self.vision_num_layers = vision_num_layers
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_patch_size = vision_patch_size
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPTextModel(BaseModel):
    """CLIP text tower as a standalone model: no vision encoder, no projection.

    Token + positional
    embedding, causal-masked transformer stack, post-encoder LayerNorm,
    and EOT-position pluck. Use this when you only need text features
    and don't want to instantiate the vision tower or carry the
    ``text_projection`` Dense.

    Output dict:

    .. code-block:: python

        out = model({"token_ids": ..., "padding_mask": ...})
        out["last_hidden_state"]   # (B, max_seq_len, text_hidden_dim)
        out["pooler_output"]       # (B, text_hidden_dim): EOT-position hidden state

    Construction:

    >>> CLIPTextModel.from_weights("clip_vit_base_16")
    >>> CLIPTextModel.from_weights("hf:openai/clip-vit-base-patch16")

    Loading from a full CLIP checkpoint silently ignores the
    vision-tower, ``text_projection``, and ``logit_scale`` entries.

    Args:
        max_seq_len: Text input length.
        vocab_size: Tokenizer vocab size.
        text_hidden_dim: Text encoder hidden dim.
        text_num_heads: Text encoder head count.
        text_num_layers: Text encoder depth.
        text_mlp_ratio: MLP expansion ratio in text blocks.
        hidden_act: MLP activation name.
        layer_norm_eps: Epsilon for every LayerNorm.
        input_tensor: Optional dict of pre-existing input tensors with
            keys ``"token_ids"`` and ``"padding_mask"``.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full CLIPModel; build
        # it from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        max_seq_len=77,
        vocab_size=49408,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPTextModel",
        **kwargs,
    ):
        for k in (
            "embed_dim",
            "image_size",
            "vision_num_layers",
            "vision_hidden_dim",
            "vision_patch_size",
            "vision_mlp_ratio",
        ):
            kwargs.pop(k, None)

        if isinstance(input_tensor, dict):
            token_ids_input = input_tensor.get("token_ids")
            if token_ids_input is None:
                token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = input_tensor.get("padding_mask")
            if padding_mask_input is None:
                padding_mask_input = layers.Input(
                    shape=[max_seq_len], name="padding_mask"
                )
        else:
            token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = layers.Input(shape=[max_seq_len], name="padding_mask")

        last_hidden_state, pooler_output = clip_text_backbone(
            token_ids_input,
            attention_mask=padding_mask_input,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

        super().__init__(
            inputs={
                "token_ids": token_ids_input,
                "padding_mask": padding_mask_input,
            },
            outputs={
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
            },
            name=name,
            **kwargs,
        )

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPImageEmbed(BaseModel):
    """CLIP vision tower + ``visual_projection``: joint-space image embeddings.

    Composes
    :class:`CLIPVisionModel` and applies the bias-free
    ``visual_projection`` Dense, producing the same image side as
    :class:`CLIPModel` but without instantiating the text tower or the
    ``logit_scale``. The projection weights are **pretrained**: loaded
    from the same CLIP checkpoint as :class:`CLIPModel`'s
    ``visual_projection``, so the output already lives in the joint
    image/text space used by the contrastive head.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["image_embeds"]        # (B, embed_dim): joint-space, unnormalized
        out["last_hidden_state"]   # (B, num_patches + 1, vision_hidden_dim)

    Construction:

    >>> CLIPImageEmbed.from_weights("clip_vit_base_16")
    >>> CLIPImageEmbed.from_weights("hf:openai/clip-vit-base-patch16")

    The text tower and ``logit_scale`` entries in the source checkpoint
    are silently ignored.

    Args:
        embed_dim: Shared joint embedding dim (= ``projection_dim``).
            Defaults to ``512``.
        image_size: Input image specification. Defaults to ``224``.
        vision_num_layers: ViT encoder depth. Defaults to ``12``.
        vision_hidden_dim: ViT hidden dim. Defaults to ``768``.
        vision_patch_size: ViT patch size. Defaults to ``32``.
        vision_mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        hidden_act: MLP activation name. Defaults to ``"quick_gelu"``.
        layer_norm_eps: LayerNorm epsilon. Defaults to ``1e-5``.
        input_tensor: Optional pre-existing Keras tensor for the
            ``images`` input.
        name: Model name. Defaults to ``"CLIPImageEmbed"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full CLIPModel; build
        # it from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        vision_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPImageEmbed",
        **kwargs,
    ):
        for k in (
            "max_seq_len",
            "vocab_size",
            "text_hidden_dim",
            "text_num_heads",
            "text_num_layers",
            "text_mlp_ratio",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            images_input = layers.Input(shape=input_shape, name="images")
        else:
            images_input = input_tensor

        vision_model = CLIPVisionModel(
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )

        image_embeds = layers.Dense(
            embed_dim, use_bias=False, name="visual_projection"
        )(vision_model.output["pooler_output"])

        super().__init__(
            inputs=images_input,
            outputs={
                "image_embeds": image_embeds,
                "last_hidden_state": vision_model.output["last_hidden_state"],
            },
            name=name,
            **kwargs,
        )

        self.vision_model = vision_model
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.vision_num_layers = vision_num_layers
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_patch_size = vision_patch_size
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPTextEmbed(BaseModel):
    """CLIP text tower + ``text_projection``: joint-space text embeddings.

    Composes
    :class:`CLIPTextModel` and applies the bias-free ``text_projection``
    Dense, producing the same text side as :class:`CLIPModel` but
    without instantiating the vision tower or the ``logit_scale``. The
    projection weights are **pretrained**: loaded from the same CLIP
    checkpoint as :class:`CLIPModel`'s ``text_projection``, so the
    output already lives in the joint image/text space.

    Output dict:

    .. code-block:: python

        out = model({"token_ids": ..., "padding_mask": ...})
        out["text_embeds"]         # (B, embed_dim): joint-space, unnormalized
        out["last_hidden_state"]   # (B, max_seq_len, text_hidden_dim)

    Construction:

    >>> CLIPTextEmbed.from_weights("clip_vit_base_16")
    >>> CLIPTextEmbed.from_weights("hf:openai/clip-vit-base-patch16")

    The vision tower and ``logit_scale`` entries in the source
    checkpoint are silently ignored.

    Args:
        embed_dim: Shared joint embedding dim. Defaults to ``512``.
        max_seq_len: Text input length. Defaults to ``77``.
        vocab_size: Tokenizer vocab size. Defaults to ``49408``.
        text_hidden_dim: Text encoder hidden dim. Defaults to ``512``.
        text_num_heads: Text encoder head count. Defaults to ``8``.
        text_num_layers: Text encoder depth. Defaults to ``12``.
        text_mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        hidden_act: MLP activation. Defaults to ``"quick_gelu"``.
        layer_norm_eps: LayerNorm epsilon. Defaults to ``1e-5``.
        input_tensor: Optional dict of pre-existing Keras tensors with
            keys ``"token_ids"`` and ``"padding_mask"``.
        name: Model name. Defaults to ``"CLIPTextEmbed"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full CLIPModel; build
        # it from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        max_seq_len=77,
        vocab_size=49408,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPTextEmbed",
        **kwargs,
    ):
        for k in (
            "image_size",
            "vision_num_layers",
            "vision_hidden_dim",
            "vision_patch_size",
            "vision_mlp_ratio",
        ):
            kwargs.pop(k, None)

        if isinstance(input_tensor, dict):
            token_ids_input = input_tensor.get("token_ids")
            if token_ids_input is None:
                token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = input_tensor.get("padding_mask")
            if padding_mask_input is None:
                padding_mask_input = layers.Input(
                    shape=[max_seq_len], name="padding_mask"
                )
        else:
            token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = layers.Input(shape=[max_seq_len], name="padding_mask")

        text_model = CLIPTextModel(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor={
                "token_ids": token_ids_input,
                "padding_mask": padding_mask_input,
            },
            name=f"{name}_text_tower",
        )

        text_pooler_3d = ops.expand_dims(text_model.output["pooler_output"], axis=1)
        text_proj = layers.Dense(embed_dim, use_bias=False, name="text_projection")(
            text_pooler_3d
        )
        text_embeds = ops.squeeze(text_proj, axis=1)

        super().__init__(
            inputs={
                "token_ids": token_ids_input,
                "padding_mask": padding_mask_input,
            },
            outputs={
                "text_embeds": text_embeds,
                "last_hidden_state": text_model.output["last_hidden_state"],
            },
            name=name,
            **kwargs,
        )

        self.text_model = text_model
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPModel(BaseModel):
    """Contrastive Language-Image Pre-training (CLIP) dual encoder.

    Joint vision + text encoder pair projecting to a shared embedding
    space. Returns the projected embeddings on each side: *no*
    similarity / logit-scale head is applied; use
    :class:`CLIPZeroShotClassify` for the standard contrastive head,
    or call :meth:`CLIPModel` and compute similarity yourself.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ..., "padding_mask": ...})
        out["image_embeddings"]   # (B, embed_dim)
        out["text_embeddings"]    # (B, embed_dim)

    Construction:

    >>> CLIPModel.from_weights("clip_vit_base_16")             # zeromodels release
    >>> CLIPModel.from_weights("hf:openai/clip-vit-base-patch16")
    >>> CLIPModel.from_weights("hf:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")

    Reference:
        - `Learning Transferable Visual Models From Natural Language
          Supervision <https://arxiv.org/abs/2103.00020>`_

    Args:
        embed_dim: Shared embedding dim (= ``projection_dim``).
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        vision_num_layers: ViT encoder depth.
        vision_hidden_dim: ViT hidden dim.
        vision_patch_size: ViT patch size.
        max_seq_len: Text input length.
        vocab_size: Tokenizer vocab size.
        text_hidden_dim: Text encoder hidden dim.
        text_num_heads: Text encoder head count.
        text_num_layers: Text encoder depth.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
        text_mlp_ratio: MLP expansion ratio in text blocks.
        hidden_act: MLP activation. ``"quick_gelu"`` for canonical
            OpenAI CLIP; ``"gelu"`` / ``"gelu_new"`` for LAION /
            community variants.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        input_tensor: Optional dict of pre-existing input tensors.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "embed_dim": hf_config["projection_dim"],
            "image_size": vc.get("image_size", 224),
            "vision_num_layers": vc["num_hidden_layers"],
            "vision_hidden_dim": vc["hidden_size"],
            "vision_patch_size": vc["patch_size"],
            "max_seq_len": tc.get("max_position_embeddings", 77),
            "vocab_size": tc["vocab_size"],
            "text_hidden_dim": tc["hidden_size"],
            "text_num_heads": tc["num_attention_heads"],
            "text_num_layers": tc["num_hidden_layers"],
            "vision_mlp_ratio": vc["intermediate_size"] / vc["hidden_size"],
            "text_mlp_ratio": tc["intermediate_size"] / tc["hidden_size"],
            "hidden_act": vc.get("hidden_act", "quick_gelu"),
            "layer_norm_eps": vc.get("layer_norm_eps", 1e-5),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        max_seq_len=77,
        vocab_size=49408,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images")
            if images_input is None:
                images_input = layers.Input(shape=input_shape, name="images")
            token_ids_input = input_tensor.get("token_ids")
            if token_ids_input is None:
                token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = input_tensor.get("padding_mask")
            if padding_mask_input is None:
                padding_mask_input = layers.Input(
                    shape=[max_seq_len], name="padding_mask"
                )
        else:
            images_input = layers.Input(shape=input_shape, name="images")
            token_ids_input = layers.Input(shape=[max_seq_len], name="token_ids")
            padding_mask_input = layers.Input(shape=[max_seq_len], name="padding_mask")

        vision_model = CLIPVisionModel(
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )
        text_model = CLIPTextModel(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor={
                "token_ids": token_ids_input,
                "padding_mask": padding_mask_input,
            },
            name=f"{name}_text_tower",
        )

        image_embeddings = layers.Dense(
            embed_dim, use_bias=False, name="visual_projection"
        )(vision_model.output["pooler_output"])

        text_pooler_3d = ops.expand_dims(text_model.output["pooler_output"], axis=1)
        text_proj = layers.Dense(embed_dim, use_bias=False, name="text_projection")(
            text_pooler_3d
        )
        text_embeddings = ops.squeeze(text_proj, axis=1)

        outputs = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
        inputs = {
            "images": images_input,
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vision_model = vision_model
        self.text_model = text_model
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.vision_num_layers = vision_num_layers
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_patch_size = vision_patch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPZeroShotClassify(BaseModel):
    """CLIP + contrastive similarity head for zero-shot classification / retrieval.

    Composes the same vision + text encoders as :class:`CLIPModel` and
    adds the standard CLIP head: L2-normalize both sides, then a
    learnable ``logit_scale`` temperature on the cosine-similarity
    matrix. Output is the ``(B, B)`` image-vs-text similarity logits,
    which softmax to zero-shot class probabilities when ``text_*``
    inputs are class-name prompts.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ..., "padding_mask": ...})
        out["image_logits"]   # (B, B): image[i] vs text[j], scaled
        out["text_logits"]    # (B, B): transpose of image_logits

    Construction:

    >>> CLIPZeroShotClassify.from_weights("clip_vit_base_16")
    >>> CLIPZeroShotClassify.from_weights("hf:openai/clip-vit-base-patch16")

    Args (identical to :class:`CLIPModel`):
        embed_dim, image_size, vision_num_layers, vision_hidden_dim,
        vision_patch_size, max_seq_len, vocab_size, text_hidden_dim,
        text_num_heads, text_num_layers, vision_mlp_ratio,
        text_mlp_ratio, hidden_act, layer_norm_eps, input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        max_seq_len=77,
        vocab_size=49408,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPZeroShotClassify",
        **kwargs,
    ):
        base = CLIPModel(
            embed_dim=embed_dim,
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            vision_mlp_ratio=vision_mlp_ratio,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )

        image_embeddings = base.output["image_embeddings"]
        text_embeddings = base.output["text_embeddings"]
        image_logits, text_logits = clip_head(image_embeddings, text_embeddings)

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.embed_dim = embed_dim
        self.image_size = base.image_size
        self.vision_num_layers = vision_num_layers
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_patch_size = vision_patch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPImageClassify(BaseModel):
    """CLIP vision encoder + linear image-classification head.

    Uses **only the CLIP
    vision encoder** (no text encoder, no visual projection), then
    mean-pools the patch tokens (excluding CLS) and applies a single
    linear classifier producing ``num_classes`` logits.

    .. code-block:: python

        model = CLIPImageClassify.from_weights(
            "hf:<user>/clip-finetune-imagenet"
        )
        logits = model(images)              # (B, num_classes)

    Args:
        num_classes: Number of output classes.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        vision_num_layers: ViT encoder depth.
        vision_hidden_dim: ViT hidden dim.
        vision_patch_size: ViT patch size.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
        hidden_act: MLP activation. ``"quick_gelu"`` for OpenAI,
            ``"gelu"`` / ``"gelu_new"`` for community variants.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        input_tensor: Optional pre-existing input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = CLIPConfig
    HUB_REPO_SIBLINGS = CLIP_HUB_SIBLINGS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full CLIPModel; build
        # it from the repo's zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from zeromodels.base.base_model import hf_num_classes

        config = CLIPModel.config_from_hf(hf_config)
        try:
            config["num_classes"] = hf_num_classes(hf_config)
        except KeyError:
            pass
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_hf_to_keras import transfer_clip_image_classify_weights

        transfer_clip_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_classes=1000,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=16,
        vision_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPImageClassify",
        **kwargs,
    ):
        for k in (
            "embed_dim",
            "max_seq_len",
            "vocab_size",
            "text_hidden_dim",
            "text_num_heads",
            "text_num_layers",
            "text_mlp_ratio",
        ):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            images_input = layers.Input(shape=input_shape, name="images")
        else:
            images_input = input_tensor

        vision_model = CLIPVisionModel(
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )
        encoded = vision_model.output["last_hidden_state"]

        pooled = ops.mean(encoded[:, 1:, :], axis=1)
        logits = layers.Dense(num_classes, name="classifier")(pooled)

        super().__init__(inputs=images_input, outputs=logits, name=name, **kwargs)

        self.vision_model = vision_model
        self.num_classes = num_classes
        self.image_size = image_size
        self.vision_num_layers = vision_num_layers
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_patch_size = vision_patch_size
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
