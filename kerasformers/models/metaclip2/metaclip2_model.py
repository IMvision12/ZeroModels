import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.clip.clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    CLIPTextModelEmbedding,
    CLIPVisionModelEmbedding,
)
from kerasformers.utils import standardize_input_shape

from .metaclip2_config import MetaClip2Config
from .metaclip2_tokenizer import METACLIP2_EOS_TOKEN_ID

# The full MetaClip2Model plus its task heads (vision / text / zero-shot /
# classify) all load from one repo per variant, whose kf_config.json declares the
# canonical MetaClip2ZeroShotClassify. Listing them as siblings lets any head load
# that repo.
METACLIP2_HUB_SIBLINGS = frozenset(
    {
        "MetaClip2Model",
        "MetaClip2VisionModel",
        "MetaClip2TextModel",
        "MetaClip2ZeroShotClassify",
        "MetaClip2ImageClassify",
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
    hidden_act="gelu",
):
    layer_prefix = f"{layer_name_prefix}_{layer_idx}"

    ln_1_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_1"
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
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_2"
    )(residual_1)

    mlp_intermediate_size = int(proj_dim * mlp_ratio)
    mlp_output = keras.layers.Dense(
        mlp_intermediate_size, name=f"{layer_prefix}_dense_1"
    )(ln_2_output)
    mlp_output = activation_layer(hidden_act)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    output = keras.layers.Add()([residual_1, mlp_output])
    return output


def metaclip2_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
    hidden_act="gelu",
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
        )
    return x


def metaclip2_vision_features(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
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

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="vision_model_layernorm_1")(
        embeddings
    )
    return metaclip2_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
    )


def metaclip2_vision_backbone(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    data_format="channels_last",
):
    last_hidden_state = metaclip2_vision_features(
        inputs,
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        num_layers=num_layers,
        heads=heads,
        vision_mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        data_format=data_format,
    )
    class_token = keras.layers.Lambda(lambda x: x[:, 0, :], name="extract_token")(
        last_hidden_state
    )
    pooler_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name="vision_model_layernorm_2"
    )(class_token)
    return last_hidden_state, pooler_output


def metaclip2_text_backbone(
    inputs,
    attention_mask,
    text_hidden_dim,
    text_num_layers,
    text_num_heads,
    vocab_size,
    max_seq_len,
    text_mlp_ratio,
    hidden_act="gelu",
    eos_token_id=METACLIP2_EOS_TOKEN_ID,
):
    x = CLIPTextModelEmbedding(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=text_hidden_dim,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.triu(ops.ones((max_seq_len, max_seq_len)) * (-1e8), k=1)

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, max_seq_len))
    expanded_mask = ops.repeat(expanded_mask, max_seq_len, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = metaclip2_encoder(
        x,
        width=text_hidden_dim,
        num_layers=text_num_layers,
        heads=text_num_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        hidden_act=hidden_act,
        layer_prefix="text_model_encoder",
    )

    last_hidden_state = keras.layers.LayerNormalization(name="text_model_layernorm")(
        encoded_output
    )

    eos_mask = ops.cast(ops.equal(inputs, eos_token_id), "int32")
    indices = ops.argmax(eos_mask, axis=-1)

    one_hot_indices = ops.one_hot(indices, max_seq_len)
    pooler_output = ops.einsum("bi,bij->bj", one_hot_indices, last_hidden_state)

    return last_hidden_state, pooler_output


def metaclip2_head(image_embeddings, text_embeddings):
    normalize_image_features = ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), axis=-1, keepdims=True)
    )
    normalize_text_features = ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), axis=-1, keepdims=True)
    )
    image_embeddings = image_embeddings / normalize_image_features
    text_embeddings = text_embeddings / normalize_text_features
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    image_logits, text_logits = logit_scale_layer([image_embeddings, text_embeddings])
    return image_logits, text_logits


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2VisionModel(BaseModel):
    """MetaCLIP 2 vision tower as a standalone model: no text encoder, no projection.

    The patch-embedding +
    transformer stack from MetaCLIP 2, ending at the post-encoder
    LayerNorm. Use this when you only need image features and don't
    want to instantiate the text tower or carry the
    ``visual_projection`` Dense.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["last_hidden_state"]   # (B, num_patches + 1, vision_hidden_dim)
        out["pooler_output"]       # (B, vision_hidden_dim): post-LN CLS token

    Construction:

    >>> MetaClip2VisionModel.from_weights("metaclip2_worldwide_b32_224")
    >>> MetaClip2VisionModel.from_weights("hf:facebook/metaclip-2-worldwide-b32")

    Loading from a full MetaCLIP 2 checkpoint silently ignores the
    text-tower, ``visual_projection``, and ``logit_scale`` entries.

    Reference:
        - `MetaCLIP 2 <https://arxiv.org/abs/2507.22062>`_

    Args:
        image_size: Input image specification. Accepts an
            integer ``N`` (builds an ``N x N x 3`` square input), a
            2-tuple ``(H, W)``, or a 3-tuple in the active data
            format's order. Defaults to ``224``.
        vision_num_layers: ViT encoder depth. Defaults to ``12``.
        vision_hidden_dim: ViT hidden dim. Defaults to ``768``.
        vision_patch_size: ViT patch size. Defaults to ``32``.
        vision_num_heads: Number of vision attention heads. ``None`` uses
            ``vision_hidden_dim // 64``.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
            Defaults to ``4.0``.
        hidden_act: MLP activation. ``"gelu"`` for MetaCLIP 2;
            ``"quick_gelu"`` for legacy OpenAI-style checkpoints.
        input_tensor: Optional pre-existing Keras tensor to use as the
            ``images`` input.
        name: Model name. Defaults to ``"MetaClip2VisionModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MetaClip2Config
    HUB_REPO_SIBLINGS = METACLIP2_HUB_SIBLINGS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full MetaClip2Model;
        # build it from the repo's kf_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MetaClip2Model.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return MetaClip2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        vision_num_heads=None,
        vision_mlp_ratio=4.0,
        hidden_act="gelu",
        input_tensor=None,
        name="MetaClip2VisionModel",
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
            "eos_token_id",
        ):
            kwargs.pop(k, None)

        if vision_num_heads is None:
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

        last_hidden_state, pooler_output = metaclip2_vision_backbone(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_hidden_dim,
            num_layers=vision_num_layers,
            heads=vision_num_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
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
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2TextModel(BaseModel):
    """MetaCLIP 2 text tower as a standalone model: no vision encoder, no projection.

    Token + positional
    embedding, causal-masked transformer stack, post-encoder LayerNorm,
    and EOS-position pluck. Use this when you only need text features
    and don't want to instantiate the vision tower or carry the
    ``text_projection`` Dense.

    Output dict:

    .. code-block:: python

        out = model({"token_ids": ..., "padding_mask": ...})
        out["last_hidden_state"]   # (B, max_seq_len, text_hidden_dim)
        out["pooler_output"]       # (B, text_hidden_dim): EOS-position hidden state

    Construction:

    >>> MetaClip2TextModel.from_weights("metaclip2_worldwide_b32_224")
    >>> MetaClip2TextModel.from_weights("hf:facebook/metaclip-2-worldwide-b32")

    Loading from a full MetaCLIP 2 checkpoint silently ignores the
    vision-tower, ``text_projection``, and ``logit_scale`` entries.

    Reference:
        - `MetaCLIP 2 <https://arxiv.org/abs/2507.22062>`_

    Args:
        max_seq_len: Text input length. Defaults to ``77``.
        vocab_size: Tokenizer vocab size (XLM-R). Defaults to ``901629``.
        text_hidden_dim: Text encoder hidden dim. Defaults to ``512``.
        text_num_heads: Text encoder head count. Defaults to ``8``.
        text_num_layers: Text encoder depth. Defaults to ``12``.
        text_mlp_ratio: MLP expansion ratio in text blocks.
            Defaults to ``4.0``.
        hidden_act: MLP activation. Defaults to ``"gelu"``.
        eos_token_id: End-of-sequence token id used to locate the
            pooled position.
        input_tensor: Optional dict of pre-existing Keras tensors with
            keys ``"token_ids"`` and ``"padding_mask"``.
        name: Model name. Defaults to ``"MetaClip2TextModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MetaClip2Config
    HUB_REPO_SIBLINGS = METACLIP2_HUB_SIBLINGS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full MetaClip2Model;
        # build it from the repo's kf_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MetaClip2Model.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return MetaClip2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        max_seq_len=77,
        vocab_size=901629,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_tensor=None,
        name="MetaClip2TextModel",
        **kwargs,
    ):
        for k in (
            "embed_dim",
            "image_size",
            "vision_num_layers",
            "vision_hidden_dim",
            "vision_patch_size",
            "vision_num_heads",
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

        last_hidden_state, pooler_output = metaclip2_text_backbone(
            token_ids_input,
            attention_mask=padding_mask_input,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
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
        self.eos_token_id = eos_token_id
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
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2Model(BaseModel):
    """MetaCLIP 2 (multilingual / worldwide) contrastive vision-language model.

    Returns the projected vision + text embeddings (no head). Use
    :class:`MetaClip2ZeroShotClassify` for the contrastive similarity
    head or :class:`MetaClip2ImageClassify` for supervised image
    classification.

    MetaCLIP 2 is Meta's 2nd-generation CLIP, trained on multilingual data with
    the XLM-R tokenizer (vocab 901629). Architecturally it is identical to
    OpenAI CLIP except for:

    - Configurable MLP activation (``"gelu"`` or ``"quick_gelu"``).
    - EOS pooling uses explicit ``eos_token_id == 2`` match instead of
      argmax-over-token-ids (needed because mask_token_id > eos_token_id).
    - Wider / deeper text tower in larger variants.

    Reference:
      - https://arxiv.org/abs/2507.22062 ("MetaCLIP 2")
      - https://huggingface.co/docs/transformers/model_doc/metaclip_2
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MetaClip2Config
    HUB_REPO_SIBLINGS = METACLIP2_HUB_SIBLINGS
    HF_MODEL_TYPE = "metaclip_2"

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
            "vision_num_heads": vc.get("num_attention_heads"),
            "max_seq_len": tc.get("max_position_embeddings", 77),
            "vocab_size": tc["vocab_size"],
            "text_hidden_dim": tc["hidden_size"],
            "text_num_heads": tc["num_attention_heads"],
            "text_num_layers": tc["num_hidden_layers"],
            "vision_mlp_ratio": vc["intermediate_size"] / vc["hidden_size"],
            "text_mlp_ratio": tc["intermediate_size"] / tc["hidden_size"],
            "hidden_act": vc.get("hidden_act", "gelu"),
            "eos_token_id": tc.get("eos_token_id", METACLIP2_EOS_TOKEN_ID),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        vision_num_heads=None,
        max_seq_len=77,
        vocab_size=901629,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_tensor=None,
        name="MetaClip2Model",
        **kwargs,
    ):
        if vision_num_heads is None:
            vision_num_heads = vision_hidden_dim // 64
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

        vision_model = MetaClip2VisionModel(
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_num_heads=vision_num_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            input_tensor=images_input,
            name=f"{name}_vision_tower",
        )
        text_model = MetaClip2TextModel(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
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
        self.vision_num_heads = vision_num_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
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
                "vision_num_heads": self.vision_num_heads,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2ZeroShotClassify(BaseModel):
    """MetaCLIP 2 + contrastive similarity head for zero-shot classification / retrieval.

    Composes the same vision + text encoders as :class:`MetaClip2Model`
    and adds the standard CLIP-style head: L2-normalize both sides,
    then a learnable ``logit_scale`` temperature on the cosine-similarity
    matrix. Output is the ``(B, B)`` image-vs-text similarity logits.

    >>> MetaClip2ZeroShotClassify.from_weights("metaclip2_worldwide_b32_224")
    >>> MetaClip2ZeroShotClassify.from_weights("hf:facebook/metaclip-2-worldwide-b32")
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MetaClip2Config
    HUB_REPO_SIBLINGS = METACLIP2_HUB_SIBLINGS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return MetaClip2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=32,
        vision_num_heads=None,
        max_seq_len=77,
        vocab_size=901629,
        text_hidden_dim=512,
        text_num_heads=8,
        text_num_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_tensor=None,
        name="MetaClip2ZeroShotClassify",
        **kwargs,
    ):
        base = MetaClip2Model(
            embed_dim=embed_dim,
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_num_heads=vision_num_heads,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            vision_mlp_ratio=vision_mlp_ratio,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = metaclip2_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

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
        self.vision_num_heads = (
            vision_num_heads
            if vision_num_heads is not None
            else vision_hidden_dim // 64
        )
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.text_hidden_dim = text_hidden_dim
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
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
                "vision_num_heads": self.vision_num_heads,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2ImageClassify(BaseModel):
    """MetaCLIP 2 vision encoder + linear image-classification head.

    Uses **only the
    vision encoder** (no text encoder, no visual projection, no post-LN,
    no ``logit_scale``), drops the CLS token, mean-pools the patch
    tokens, and applies a single linear classifier producing
    ``num_classes`` logits.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MetaClip2Config
    HUB_REPO_SIBLINGS = METACLIP2_HUB_SIBLINGS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # This head shares the variant's weights repo with the full MetaClip2Model;
        # build it from the repo's kf_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MetaClip2Model.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from kerasformers.base.base_model import hf_num_classes

        config = MetaClip2Model.config_from_hf(hf_config)
        try:
            config["num_classes"] = hf_num_classes(hf_config)
        except KeyError:
            pass
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_image_classify_weights,
        )

        transfer_metaclip2_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_classes=1000,
        image_size=224,
        vision_num_layers=12,
        vision_hidden_dim=768,
        vision_patch_size=16,
        vision_num_heads=None,
        vision_mlp_ratio=4.0,
        hidden_act="gelu",
        input_tensor=None,
        name="MetaClip2ImageClassify",
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
            "eos_token_id",
        ):
            kwargs.pop(k, None)

        if vision_num_heads is None:
            vision_num_heads = vision_hidden_dim // 64
        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            images_input = layers.Input(shape=input_shape, name="images")
        else:
            images_input = input_tensor

        vision_model = MetaClip2VisionModel(
            image_size=image_size,
            vision_num_layers=vision_num_layers,
            vision_hidden_dim=vision_hidden_dim,
            vision_patch_size=vision_patch_size,
            vision_num_heads=vision_num_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
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
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
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
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
