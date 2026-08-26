import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    MediaMerge,
    TextOnlyGeneration,
    TiedHead,
    merge_media,
)
from zeromodels.base.base_mixin import inference_scope

from .gemma3_config import Gemma3Config, Gemma3TextConfig
from .gemma3_layers import Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3VisionLayer

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3VisionModel(layers.Layer):
    """SigLIP vision tower: biased conv patch embed + learned position
    embeddings -> pre-LN encoder blocks -> final LayerNorm.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads.
        image_size: Square input size in pixels (896).
        patch_size: Patch size in pixels (14).
        norm_eps: LayerNorm epsilon.

    Call args:
        pixel_values: ``(num_images, H, W, 3)`` (or channels-first).

    Returns:
        ``(num_images, num_patches, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        image_size=896,
        patch_size=14,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.num_positions = (image_size // patch_size) ** 2
        self.data_format = keras.config.image_data_format()

        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=self.data_format,
            name="patch_embed",
        )
        self.position_embedding = layers.Embedding(
            self.num_positions, embed_dim, name="position_embedding"
        )
        self.blocks = [
            Gemma3VisionLayer(
                embed_dim, mlp_dim, num_heads, norm_eps, name=f"blocks_{i}"
            )
            for i in range(num_layers)
        ]
        self.post_layernorm = layers.LayerNormalization(
            epsilon=norm_eps, name="post_layernorm"
        )

    def call(self, pixel_values):
        x = self.patch_embed(pixel_values)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        b = ops.shape(x)[0]
        x = ops.reshape(x, (b, -1, self.embed_dim))
        positions = ops.arange(self.num_positions)
        x = x + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x)
        return self.post_layernorm(x)

    def compute_output_spec(self, pixel_values):
        return keras.KerasTensor(
            (pixel_values.shape[0], self.num_positions, self.embed_dim),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3MultiModalProjector(layers.Layer):
    """Gemma 3 vision projector: 4x4 average pool -> soft-token RMS norm ->
    matmul with the learned ``(vision_dim, text_dim)`` projection matrix.

    Args:
        vision_dim: Vision hidden width.
        text_dim: Text decoder hidden width.
        patches_per_image: Vision patch-grid side (64 for 896/14).
        tokens_per_side: Output token-grid side (16 for 256 tokens).
        norm_eps: Epsilon of the soft-token norm.
    """

    def __init__(
        self,
        vision_dim,
        text_dim,
        patches_per_image=64,
        tokens_per_side=16,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.patches_per_image = patches_per_image
        self.tokens_per_side = tokens_per_side
        self.norm_eps = norm_eps
        self.kernel_size = patches_per_image // tokens_per_side
        self.mm_soft_emb_norm = Gemma3RMSNorm(eps=norm_eps, name="mm_soft_emb_norm")

    def build(self, input_shape):
        self.mm_input_projection_weight = self.add_weight(
            name="mm_input_projection_weight",
            shape=(self.vision_dim, self.text_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, vision_outputs):
        b = ops.shape(vision_outputs)[0]
        p, k = self.patches_per_image, self.kernel_size
        x = ops.reshape(vision_outputs, (b, p, p, self.vision_dim))
        x = ops.reshape(x, (b, p // k, k, p // k, k, self.vision_dim))
        x = ops.mean(x, axis=(2, 4))  # 4x4 average pool
        x = ops.reshape(x, (b, (p // k) * (p // k), self.vision_dim))
        x = self.mm_soft_emb_norm(x)
        return ops.matmul(x, ops.cast(self.mm_input_projection_weight, x.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "patches_per_image": self.patches_per_image,
                "tokens_per_side": self.tokens_per_side,
                "norm_eps": self.norm_eps,
            }
        )
        return config


def gemma3_rope_tables(position_ids, head_dim, theta, scaling_factor, compute_dtype):
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    if scaling_factor is not None:
        inv_freq = inv_freq / scaling_factor
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def gemma3_image_groups(input_ids, image_token_id):
    # Consecutive runs of image placeholder tokens get a group id (>= 0);
    # everything else -1. Image groups attend bidirectionally.
    is_image = ops.cast(input_ids == image_token_id, "int32")
    prev = ops.concatenate([ops.zeros_like(is_image[:, :1]), is_image[:, :-1]], axis=1)
    new_start = is_image * (1 - prev)
    groups = ops.cumsum(new_start, axis=1) - 1
    return ops.where(is_image > 0, groups, -1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3Masks(layers.Layer):
    """Builds ONE additive mask (full or sliding) with the image-bidirectional overlay.

    Image-token groups attend bidirectionally (OR-ed in). Single-output on purpose:
    a tuple-returning layer with identical specs mis-wires in the functional graph
    (both consumers collapse to one output), so the model uses two instances
    (``sliding=False`` / ``sliding=True``). The dynamic arange / group logic is
    isolated behind ``compute_output_spec``; the KV-cache prefill uses ``build_masks``.
    """

    def __init__(self, sliding_window, image_token_id, sliding=False, **kwargs):
        super().__init__(**kwargs)
        self.sliding_window = sliding_window
        self.image_token_id = image_token_id
        self.sliding = sliding

    def call(self, input_ids, attention_mask):
        seq = ops.shape(input_ids)[1]
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = (ki <= qi)[None]
        groups = gemma3_image_groups(input_ids, self.image_token_id)
        same_image = ops.logical_and(
            groups[:, :, None] == groups[:, None, :], (groups >= 0)[:, :, None]
        )
        if self.sliding:
            in_window = (ki > qi - self.sliding_window)[None]
            keep = ops.logical_or(ops.logical_and(causal, in_window), same_image)
        else:
            keep = ops.logical_or(causal, same_image)
        mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[:, None]
        am = ops.cast(attention_mask, "float32")
        return mask + (1.0 - am)[:, None, None, :] * MASK_NEG

    def compute_output_spec(self, input_ids, attention_mask):
        seq = input_ids.shape[1]
        return keras.KerasTensor((input_ids.shape[0], 1, seq, seq), dtype="float32")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sliding_window": self.sliding_window,
                "image_token_id": self.image_token_id,
                "sliding": self.sliding,
            }
        )
        return config


def gemma3_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    vision_tower,
    projector,
    image_merge,
    full_mask_layer,
    sliding_mask_layer,
    is_sliding_flags,
    embed_dim,
    head_dim,
    rope_theta,
    rope_local_theta,
    rope_scaling_factor,
    compute_dtype,
):
    hidden = token_embedding(input_ids) * ops.cast(embed_dim**0.5, compute_dtype)
    if vision_tower is not None:
        features = vision_tower(pixel_values)
        image_embeds = projector(features)
        image_embeds = ops.reshape(image_embeds, (-1, embed_dim))
        hidden = image_merge(hidden, input_ids, image_embeds)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos_l, sin_l = gemma3_rope_tables(
        position_ids, head_dim, rope_local_theta, None, compute_dtype
    )
    cos_g, sin_g = gemma3_rope_tables(
        position_ids, head_dim, rope_theta, rope_scaling_factor, compute_dtype
    )
    full_mask = full_mask_layer(input_ids, attention_mask)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    for layer, sliding in zip(decoder_layers, is_sliding_flags):
        if sliding:
            hidden = layer(hidden, cos_l, sin_l, attention_mask=sliding_mask)
        else:
            hidden = layer(hidden, cos_g, sin_g, attention_mask=full_mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3Model(BaseModel):
    """Gemma 3 decoder-only backbone (optionally with a SigLIP vision tower).

    A functional model. Scaled embeddings, ``(1 + w)`` RMSNorms, a
    sliding-to-global layer pattern (``sliding_window_pattern``) with *dual rotary
    bases* (sliding layers use ``rope_local_theta``, global use ``rope_theta``),
    and, when ``vision_num_layers > 0``, an in-graph SigLIP tower whose merged
    image embeddings are scattered into the ``image_token_id`` slots; image-token
    groups attend bidirectionally. Returns ``last_hidden_state``; use
    :class:`Gemma3ConditionalGenerate` / :class:`Gemma3TextGenerate` for logits.
    """

    HF_MODEL_TYPE = "gemma3"
    default_load_dtype = "bfloat16"
    config_class = Gemma3Config
    output_logits = False

    def __init__(
        self,
        vocab_size=262144,
        embed_dim=1152,
        mlp_dim=6912,
        num_layers=26,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        query_pre_attn_scalar=256.0,
        sliding_window=512,
        sliding_window_pattern=6,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        rope_scaling_factor=None,
        tie_embeddings=True,
        vision_embed_dim=1152,
        vision_mlp_dim=4304,
        vision_num_layers=0,
        vision_num_heads=16,
        image_size=896,
        patch_size=14,
        vision_norm_eps=1e-6,
        mm_tokens_per_image=256,
        image_token_id=262144,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Gemma3DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                query_pre_attn_scalar,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Gemma3RMSNorm(eps=norm_eps, name="final_norm")
        vision_tower = None
        projector = None
        image_merge = None
        if vision_num_layers:
            vision_tower = Gemma3VisionModel(
                vision_embed_dim,
                vision_mlp_dim,
                vision_num_layers,
                vision_num_heads,
                image_size,
                patch_size,
                vision_norm_eps,
                name="vision_tower",
            )
            projector = Gemma3MultiModalProjector(
                vision_embed_dim,
                embed_dim,
                image_size // patch_size,
                int(mm_tokens_per_image**0.5),
                vision_norm_eps,
                name="multi_modal_projector",
            )
            image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        full_mask_layer = Gemma3Masks(
            sliding_window, image_token_id, sliding=False, name="full_mask"
        )
        sliding_mask_layer = Gemma3Masks(
            sliding_window, image_token_id, sliding=True, name="sliding_mask"
        )
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        is_sliding_flags = [
            bool((i + 1) % sliding_window_pattern) for i in range(num_layers)
        ]
        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        if vision_num_layers:
            img_shape = (
                (3, image_size, image_size)
                if keras.config.image_data_format() == "channels_first"
                else (image_size, image_size, 3)
            )
            inputs["pixel_values"] = layers.Input(
                shape=img_shape,
                dtype="float32",
                name="pixel_values",
            )
        hidden = gemma3_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("pixel_values"),
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            vision_tower=vision_tower,
            projector=projector,
            image_merge=image_merge,
            full_mask_layer=full_mask_layer,
            sliding_mask_layer=sliding_mask_layer,
            is_sliding_flags=is_sliding_flags,
            embed_dim=embed_dim,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_local_theta=rope_local_theta,
            rope_scaling_factor=rope_scaling_factor,
            compute_dtype=token_embedding.compute_dtype,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(token_embedding, name="lm_head")(hidden)
            )

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.vision_tower = vision_tower
        self.multi_modal_projector = projector
        self.image_merge = image_merge
        self.full_mask_layer = full_mask_layer
        self.sliding_mask_layer = sliding_mask_layer
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_norm_eps = vision_norm_eps
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id

        # Gemma's (1+w) RMSNorm + the vision tower can abort Keras' symbolic
        # auto-build on some backends; a concrete dummy forward materializes every
        # weight so from_weights (which loads before any forward) has a complete model.
        with inference_scope():
            self(self.dummy_inputs())

    def dummy_inputs(self):
        text = {
            "input_ids": ops.zeros((1, 4), dtype="int32"),
            "attention_mask": ops.ones((1, 4), dtype="int32"),
        }
        if self.vision_tower is None:
            return text
        n = self.mm_tokens_per_image
        text["input_ids"] = ops.concatenate(
            [
                ops.zeros((1, 1), dtype="int32"),
                ops.full((1, n), self.image_token_id, dtype="int32"),
                ops.ones((1, 1), dtype="int32"),
            ],
            axis=1,
        )
        text["attention_mask"] = ops.ones((1, n + 2), dtype="int32")
        text["pixel_values"] = ops.zeros(
            (1, 3, self.image_size, self.image_size)
            if keras.config.image_data_format() == "channels_first"
            else (1, self.image_size, self.image_size, 3),
            dtype="float32",
        )
        return text

    def is_sliding(self, layer_idx):
        return bool((layer_idx + 1) % self.sliding_window_pattern)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids, local):
        theta = self.rope_local_theta if local else self.rope_theta
        factor = None if local else self.rope_scaling_factor
        return gemma3_rope_tables(
            position_ids, self.head_dim, theta, factor, self.compute_dtype
        )

    def image_groups(self, input_ids):
        return gemma3_image_groups(input_ids, self.image_token_id)

    def build_masks(self, input_ids, attention_mask=None):
        seq = int(input_ids.shape[1])
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = (ki <= qi)[None]
        in_window = (ki > qi - self.sliding_window)[None]
        groups = self.image_groups(ops.cast(ops.convert_to_tensor(input_ids), "int32"))
        same_image = ops.logical_and(
            groups[:, :, None] == groups[:, None, :], (groups >= 0)[:, :, None]
        )
        full_keep = ops.logical_or(causal, same_image)
        sliding_keep = ops.logical_or(ops.logical_and(causal, in_window), same_image)
        full = ops.cast(ops.where(full_keep, 0.0, MASK_NEG), "float32")[:, None]
        sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[:, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    def get_image_features(self, pixel_values):
        return self.multi_modal_projector(self.vision_tower(pixel_values))

    def prepare_inputs(self, input_ids, pixel_values, attention_mask):
        if pixel_values is not None and self.vision_tower is None:
            raise ValueError(
                "This Gemma3 variant is text-only (no vision tower), so it cannot "
                "process images. Use a multimodal variant such as 'gemma-3-4b-it' "
                "(or 12b / 27b) for image inputs."
            )
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.embed_scaled(input_ids)
        if pixel_values is not None and self.vision_tower is not None:
            image_embeds = self.get_image_features(pixel_values)
            image_embeds = ops.reshape(image_embeds, (-1, self.embed_dim))
            inputs_embeds = merge_media(
                inputs_embeds,
                input_ids,
                image_embeds,
                self.image_token_id,
                self.embed_dim,
            )
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        return inputs_embeds, position_ids

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        vision = hf_config.get("vision_config")
        rope_scaling = text.get("rope_scaling") or {}
        out = {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim", 256),
            "query_pre_attn_scalar": text.get("query_pre_attn_scalar", 256.0),
            "sliding_window": text.get("sliding_window", 512),
            "sliding_window_pattern": text.get("sliding_window_pattern", 6),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 1000000.0),
            "rope_local_theta": text.get("rope_local_base_freq", 10000.0),
            "rope_scaling_factor": rope_scaling.get("factor"),
            "tie_embeddings": text.get("tie_word_embeddings", True),
            "image_token_id": hf_config.get(
                "image_token_id", hf_config.get("image_token_index", 262144)
            ),
            "mm_tokens_per_image": hf_config.get("mm_tokens_per_image", 256),
        }
        if vision is not None:
            out.update(
                {
                    "vision_embed_dim": vision["hidden_size"],
                    "vision_mlp_dim": vision["intermediate_size"],
                    "vision_num_layers": vision["num_hidden_layers"],
                    "vision_num_heads": vision["num_attention_heads"],
                    "image_size": vision.get("image_size", 896),
                    "patch_size": vision.get("patch_size", 14),
                    "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
                }
            )
        return out

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma3_hf_to_keras import transfer_gemma3_weights

        transfer_gemma3_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "rope_scaling_factor": self.rope_scaling_factor,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_norm_eps": self.vision_norm_eps,
                "mm_tokens_per_image": self.mm_tokens_per_image,
                "image_token_id": self.image_token_id,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3ConditionalGenerate(Gemma3Model, BaseGeneration):
    """Gemma 3 with a (tied) LM head + fast ``.generate()`` (text or image+text)."""

    eos_token_id = (1, 106)
    output_logits = True
    HUB_REPO_SIBLINGS = frozenset(
        {"Gemma3Model", "Gemma3ConditionalGenerate", "Gemma3TextGenerate"}
    )

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        # Multimodal prefill with the image-bidirectional masks; each layer's
        # K/V lands in a fixed (B, num_layers, 2, nkv, max_len, hd) cache.
        batch = int(token_ids.shape[0])
        hd, nkv = self.head_dim, self.num_kv_heads
        inputs_embeds, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(token_ids, padding_mask)
        hidden = inputs_embeds
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden, (k, v) = layer(
                    hidden, cos_l, sin_l, attention_mask=sliding_mask, use_cache=True
                )
            else:
                hidden, (k, v) = layer(
                    hidden, cos_g, sin_g, attention_mask=full_mask, use_cache=True
                )
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # Text-only decode step; sliding layers see only their window.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_l, sin_l = self.rope_tables(positions, local=True)
        cos_g, sin_g = self.rope_tables(positions, local=False)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        h = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                h, ck, cv = layer.decode_step(
                    h, cos_l, sin_l, cache[:, i, 0], cache[:, i, 1], pos, sliding_km
                )
            else:
                h, ck, cv = layer.decode_step(
                    h, cos_g, sin_g, cache[:, i, 0], cache[:, i, 1], pos, full_km
                )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3TextGenerate(TextOnlyGeneration, Gemma3ConditionalGenerate):
    """Gemma 3 text-only decoder + (tied) LM head with fast ``.generate()``.

    The text-only head for the 1B / 270M checkpoints (built with
    ``vision_num_layers=0`` so no SigLIP tower). :class:`TextOnlyGeneration` builds
    it text-only and drops the multimodal prefill inputs.
    """

    config_class = Gemma3TextConfig
