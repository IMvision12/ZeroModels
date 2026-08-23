import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.vit.vit_layers import ViTAddPositionEmbs, ViTClassDistToken
from kerasformers.models.vit.vit_model import transformer_block
from kerasformers.utils import standardize_input_shape

from .tipsv2_config import Tipsv2Config
from .tipsv2_layers import (
    Tipsv2MaskedMeanPool,
    Tipsv2PaddingMask,
    Tipsv2RegisterTokens,
    Tipsv2TextEmbedding,
    tipsv2_text_encoder_layer,
)

TIPSV2_HUB_SIBLINGS = frozenset({"Tipsv2Model", "Tipsv2VisionModel", "Tipsv2TextModel"})


def tipsv2_vision_features(
    inputs,
    *,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    patch_size,
    num_register_tokens,
    layerscale_value,
    use_swiglu,
    layer_norm_eps,
    image_size,
    data_format,
    resize_mode,
):
    """DINOv2-style ViT: patch embed -> [CLS, registers, patches] -> blocks -> final LN."""
    grid = image_size // patch_size
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="conv1",
    )(inputs)
    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((-1, embed_dim))(x)
    x = ViTClassDistToken(use_distillation=False, name="cls_token")(x)
    x = ViTAddPositionEmbs(
        name="pos_embed",
        no_embed_class=False,
        use_distillation=False,
        grid_h=grid,
        grid_w=grid,
        resize_mode=resize_mode,
    )(x)
    x = Tipsv2RegisterTokens(num_tokens=num_register_tokens, name="register_tokens")(x)
    for i in range(depth):
        x = transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_norm=False,
            block_idx=i,
            layer_scale_init=layerscale_value,
            use_swiglu=use_swiglu,
        )
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name="final_layernorm"
    )(x)
    return x


def tipsv2_text_features(
    token_ids,
    attention_mask,
    *,
    vocab_size,
    hidden_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_seq_len,
    hidden_act,
    layer_norm_eps,
    scale_sqrt_depth,
):
    """Bidirectional text encoder: scaled token embed + sinusoidal pos -> blocks -> LN."""
    x = layers.Embedding(vocab_size, hidden_dim, name="text_model_token_embedding")(
        token_ids
    )
    x = Tipsv2TextEmbedding(
        max_seq_len=max_seq_len,
        embed_dim=hidden_dim,
        scale_sqrt_depth=scale_sqrt_depth,
        name="text_model_position_embedding",
    )(x)
    for i in range(num_layers):
        x = tipsv2_text_encoder_layer(
            x,
            attention_mask,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            name=f"text_model_encoder_layers_{i}",
        )
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name="text_model_final_layernorm"
    )(x)
    return x


def l2_normalize(x):
    return x / ops.sqrt(ops.sum(ops.power(x, 2), axis=-1, keepdims=True))


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2VisionModel(BaseModel):
    """TIPSv2 vision tower (DINOv2-style ViT with register tokens).

    Output dict: ``last_hidden_state`` ``(B, 1 + R + num_patches, hidden_dim)`` and
    ``pooler_output`` ``(B, hidden_dim)`` (the CLS token). Input pixels are expected in
    ``[0, 1]`` (the image processor rescales but does not normalize).
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Tipsv2Config
    HUB_REPO_SIBLINGS = TIPSV2_HUB_SIBLINGS
    HF_MODEL_TYPE = "tipsv2"

    @classmethod
    def _release_warm_start_cls(cls):
        return Tipsv2Model

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
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
        return Tipsv2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_tipsv2_hf_to_keras import transfer_tipsv2_weights

        transfer_tipsv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=448,
        patch_size=14,
        num_register_tokens=1,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_ratio=4.0,
        vision_use_swiglu_ffn=False,
        vision_layerscale_value=1.0,
        vision_layer_norm_eps=1e-6,
        resize_mode="bilinear",
        input_tensor=None,
        name="Tipsv2VisionModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)
        image_size = (
            input_shape[0] if data_format == "channels_last" else input_shape[1]
        )

        if input_tensor is None:
            images = layers.Input(shape=input_shape, name="images")
        elif not utils.is_keras_tensor(input_tensor):
            images = layers.Input(tensor=input_tensor, shape=input_shape, name="images")
        else:
            images = input_tensor

        x = tipsv2_vision_features(
            images,
            embed_dim=vision_hidden_dim,
            depth=vision_num_layers,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            layerscale_value=vision_layerscale_value,
            use_swiglu=vision_use_swiglu_ffn,
            layer_norm_eps=vision_layer_norm_eps,
            image_size=image_size,
            data_format=data_format,
            resize_mode=resize_mode,
        )
        outputs = {"last_hidden_state": x, "pooler_output": x[:, 0]}

        super().__init__(inputs=images, outputs=outputs, name=name, **kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_use_swiglu_ffn = vision_use_swiglu_ffn
        self.vision_layerscale_value = vision_layerscale_value
        self.vision_layer_norm_eps = vision_layer_norm_eps
        self.resize_mode = resize_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "vision_use_swiglu_ffn": self.vision_use_swiglu_ffn,
                "vision_layerscale_value": self.vision_layerscale_value,
                "vision_layer_norm_eps": self.vision_layer_norm_eps,
                "resize_mode": self.resize_mode,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2TextModel(BaseModel):
    """TIPSv2 text tower (bidirectional transformer with masked-mean pooling).

    Inputs dict ``{"token_ids", "padding_mask"}``. Output dict: ``last_hidden_state``
    ``(B, L, hidden_dim)`` and ``pooler_output`` ``(B, hidden_dim)`` (masked mean).
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Tipsv2Config
    HUB_REPO_SIBLINGS = TIPSV2_HUB_SIBLINGS
    HF_MODEL_TYPE = "tipsv2"

    @classmethod
    def _release_warm_start_cls(cls):
        return Tipsv2Model

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
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
        return Tipsv2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_tipsv2_hf_to_keras import transfer_tipsv2_weights

        transfer_tipsv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        vocab_size=32000,
        max_seq_len=64,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        text_hidden_act="relu",
        text_layer_norm_eps=1e-5,
        text_scale_sqrt_depth=True,
        text_pooling_epsilon=1e-8,
        input_tensor=None,
        name="Tipsv2TextModel",
        **kwargs,
    ):
        sources = input_tensor if isinstance(input_tensor, dict) else {}
        token_ids = sources.get("token_ids")
        if token_ids is None:
            token_ids = layers.Input(
                shape=(max_seq_len,), dtype="int32", name="token_ids"
            )
        padding_mask = sources.get("padding_mask")
        if padding_mask is None:
            padding_mask = layers.Input(
                shape=(max_seq_len,), dtype="int32", name="padding_mask"
            )

        attention_mask = Tipsv2PaddingMask(name="padding_to_mask")(padding_mask)
        x = tipsv2_text_features(
            token_ids,
            attention_mask,
            vocab_size=vocab_size,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            mlp_dim=text_mlp_dim,
            max_seq_len=max_seq_len,
            hidden_act=text_hidden_act,
            layer_norm_eps=text_layer_norm_eps,
            scale_sqrt_depth=text_scale_sqrt_depth,
        )
        pooled = Tipsv2MaskedMeanPool(
            epsilon=text_pooling_epsilon, name="masked_mean_pool"
        )(x, padding_mask)

        inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        outputs = {"last_hidden_state": x, "pooler_output": pooled}
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.text_hidden_act = text_hidden_act
        self.text_layer_norm_eps = text_layer_norm_eps
        self.text_scale_sqrt_depth = text_scale_sqrt_depth
        self.text_pooling_epsilon = text_pooling_epsilon
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "text_hidden_act": self.text_hidden_act,
                "text_layer_norm_eps": self.text_layer_norm_eps,
                "text_scale_sqrt_depth": self.text_scale_sqrt_depth,
                "text_pooling_epsilon": self.text_pooling_epsilon,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2Model(BaseModel):
    """TIPSv2 dual encoder with a temperature-scaled contrastive head.

    Composes :class:`Tipsv2VisionModel` and :class:`Tipsv2TextModel`, L2-normalizes
    each tower's pooled embedding, and produces cosine-similarity logits scaled by
    ``1 / temperature_init_value`` (TIPSv2 does not store the learned temperature).

    Inputs dict ``{"images", "token_ids", "padding_mask"}``. Output dict:
    ``image_embeddings``, ``text_embeddings`` (L2-normalized), ``logits_per_image``
    and ``logits_per_text`` (each ``(B, B)``).
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Tipsv2Config
    HUB_REPO_SIBLINGS = TIPSV2_HUB_SIBLINGS
    HF_MODEL_TYPE = "tipsv2"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "image_size": vc.get("image_size", 448),
            "patch_size": vc.get("patch_size", 14),
            "num_register_tokens": vc.get("num_register_tokens", 1),
            "vision_hidden_dim": vc.get("hidden_size", 768),
            "vision_num_layers": vc.get("num_hidden_layers", 12),
            "vision_num_heads": vc.get("num_attention_heads", 12),
            "vision_mlp_ratio": vc.get("mlp_ratio", 4.0),
            "vision_use_swiglu_ffn": vc.get("use_swiglu_ffn", False),
            "vision_layerscale_value": vc.get("layerscale_value", 1.0),
            "vision_layer_norm_eps": vc.get("layer_norm_eps", 1e-6),
            "vocab_size": tc.get("vocab_size", 32000),
            "max_seq_len": tc.get("max_position_embeddings", 64),
            "embed_dim": tc.get("hidden_size", 768),
            "text_hidden_dim": tc.get("hidden_size", 768),
            "text_num_layers": tc.get("num_hidden_layers", 12),
            "text_num_heads": tc.get("num_attention_heads", 12),
            "text_mlp_dim": tc.get("intermediate_size", 3072),
            "text_hidden_act": tc.get("hidden_act", "relu"),
            "text_layer_norm_eps": tc.get("layer_norm_eps", 1e-5),
            "text_scale_sqrt_depth": tc.get("scale_sqrt_depth", True),
            "text_pooling_epsilon": tc.get("pooling_epsilon", 1e-8),
            "temperature_init_value": hf_config.get(
                "temperature_init_value", 0.005065968260169029
            ),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_tipsv2_hf_to_keras import transfer_tipsv2_weights

        transfer_tipsv2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=448,
        patch_size=14,
        num_register_tokens=1,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_ratio=4.0,
        vision_use_swiglu_ffn=False,
        vision_layerscale_value=1.0,
        vision_layer_norm_eps=1e-6,
        resize_mode="bilinear",
        vocab_size=32000,
        max_seq_len=64,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        text_hidden_act="relu",
        text_layer_norm_eps=1e-5,
        text_scale_sqrt_depth=True,
        text_pooling_epsilon=1e-8,
        temperature_init_value=0.005065968260169029,
        input_tensor=None,
        name="Tipsv2Model",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)

        sources = input_tensor if isinstance(input_tensor, dict) else {}
        images = sources.get("images")
        if images is None:
            images = layers.Input(shape=input_shape, name="images")
        token_ids = sources.get("token_ids")
        if token_ids is None:
            token_ids = layers.Input(
                shape=(max_seq_len,), dtype="int32", name="token_ids"
            )
        padding_mask = sources.get("padding_mask")
        if padding_mask is None:
            padding_mask = layers.Input(
                shape=(max_seq_len,), dtype="int32", name="padding_mask"
            )

        vision_model = Tipsv2VisionModel(
            image_size=image_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            vision_use_swiglu_ffn=vision_use_swiglu_ffn,
            vision_layerscale_value=vision_layerscale_value,
            vision_layer_norm_eps=vision_layer_norm_eps,
            resize_mode=resize_mode,
            input_tensor=images,
            name=f"{name}_vision_tower",
        )
        text_model = Tipsv2TextModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_mlp_dim=text_mlp_dim,
            text_hidden_act=text_hidden_act,
            text_layer_norm_eps=text_layer_norm_eps,
            text_scale_sqrt_depth=text_scale_sqrt_depth,
            text_pooling_epsilon=text_pooling_epsilon,
            input_tensor={"token_ids": token_ids, "padding_mask": padding_mask},
            name=f"{name}_text_tower",
        )

        image_embeds = l2_normalize(vision_model.output["pooler_output"])
        text_embeds = l2_normalize(text_model.output["pooler_output"])

        logits_per_text = ops.matmul(text_embeds, ops.transpose(image_embeds)) / (
            temperature_init_value
        )
        logits_per_image = ops.transpose(logits_per_text)

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        outputs = {
            "image_embeddings": image_embeds,
            "text_embeddings": text_embeds,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vision_model = vision_model
        self.text_model = text_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_use_swiglu_ffn = vision_use_swiglu_ffn
        self.vision_layerscale_value = vision_layerscale_value
        self.vision_layer_norm_eps = vision_layer_norm_eps
        self.resize_mode = resize_mode
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.text_hidden_act = text_hidden_act
        self.text_layer_norm_eps = text_layer_norm_eps
        self.text_scale_sqrt_depth = text_scale_sqrt_depth
        self.text_pooling_epsilon = text_pooling_epsilon
        self.temperature_init_value = temperature_init_value
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "vision_use_swiglu_ffn": self.vision_use_swiglu_ffn,
                "vision_layerscale_value": self.vision_layerscale_value,
                "vision_layer_norm_eps": self.vision_layer_norm_eps,
                "resize_mode": self.resize_mode,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "text_hidden_act": self.text_hidden_act,
                "text_layer_norm_eps": self.text_layer_norm_eps,
                "text_scale_sqrt_depth": self.text_scale_sqrt_depth,
                "text_pooling_epsilon": self.text_pooling_epsilon,
                "temperature_init_value": self.temperature_init_value,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
