import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
    merge_media,
)
from zeromodels.base.base_mixin import inference_scope

from .janus_config import JanusConfig
from .janus_layers import (
    JanusTextDecoderLayer,
    JanusTextRMSNorm,
    JanusVisionLayer,
)

MASK_NEG = -1e9

# The backbone (JanusModel) and generative head (JanusConditionalGenerate) share the
# variant's weights repo, whose kf_config.json declares JanusModel.
JANUS_HUB_SIBLINGS = frozenset({"JanusModel", "JanusConditionalGenerate"})


@keras.saving.register_keras_serializable(package="zeromodels")
class JanusVisionModel(layers.Layer):
    """SigLIP vision tower: biased conv patch embed + learned position
    embeddings -> pre-LN encoder blocks (exact gelu) -> final LayerNorm.

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
        image_size=384,
        patch_size=16,
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
            JanusVisionLayer(
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
        x = x + self.position_embedding(ops.arange(self.num_positions))[None]
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


def janus_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def janus_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    *,
    token_embedding,
    vision_model,
    aligner_fc1,
    aligner_hidden,
    decoder_layers,
    final_norm,
    causal_mask,
    image_merge,
    head_dim,
    rope_theta,
    embed_dim,
    image_token_id,
    compute_dtype,
):
    # Zero the image placeholder before the lookup (its id can sit outside the
    # embedding range); the merge overwrites those slots, so this is parity-safe.
    hidden = token_embedding(
        ops.where(ops.equal(input_ids, image_token_id), 0, input_ids)
    )
    features = vision_model(pixel_values)
    image_embeds = aligner_hidden(ops.gelu(aligner_fc1(features), approximate=False))
    image_embeds = ops.reshape(image_embeds, (-1, embed_dim))
    hidden = image_merge(hidden, input_ids, image_embeds)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = janus_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class JanusModel(BaseModel):
    """Janus multimodal backbone: SigLIP tower + depth-2 GELU aligner + Llama decoder.

    Image patch tokens are aligned to the text width by
    ``hidden(gelu(fc1(x)))`` and scattered (in-graph, via
    :class:`~zeromodels.base.MediaMerge`) into the ``image_token_id``
    (``<image_placeholder>``) slots of the decoder input. A functional multimodal
    model over ``{input_ids, attention_mask, pixel_values}`` (the VQVAE generation
    path is not ported). Returns ``last_hidden_state``; use
    :class:`JanusConditionalGenerate` for logits / text.
    """

    HF_MODEL_TYPE = "janus"
    default_load_dtype = "bfloat16"  # official Janus-Pro checkpoints are bf16
    BASE_MODEL_CONFIG = None
    # Weights load by Hub repo id, e.g. from_weights("zeromodels/janus_pro_1b"),
    # via kf_config.json on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    config_class = JanusConfig
    HUB_REPO_SIBLINGS = JANUS_HUB_SIBLINGS
    output_logits = False

    def __init__(
        self,
        vocab_size=102400,
        embed_dim=2048,
        mlp_dim=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=10000.0,
        tie_embeddings=False,
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_layers=24,
        vision_num_heads=16,
        image_size=384,
        patch_size=16,
        vision_norm_eps=1e-6,
        image_token_id=100581,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads

        vision_model = JanusVisionModel(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            image_size,
            patch_size,
            vision_norm_eps,
            name="vision_model",
        )
        aligner_fc1 = layers.Dense(embed_dim, name="aligner_fc1")
        aligner_hidden = layers.Dense(embed_dim, name="aligner_hidden")
        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            JanusTextDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = JanusTextRMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        img_shape = (
            (3, image_size, image_size)
            if keras.config.image_data_format() == "channels_first"
            else (image_size, image_size, 3)
        )
        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=img_shape, dtype="float32", name="pixel_values"
            ),
        }
        hidden = janus_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            token_embedding=token_embedding,
            vision_model=vision_model,
            aligner_fc1=aligner_fc1,
            aligner_hidden=aligner_hidden,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            image_merge=image_merge,
            head_dim=head_dim,
            rope_theta=rope_theta,
            embed_dim=embed_dim,
            image_token_id=image_token_id,
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

        self.vision_model = vision_model
        self.aligner_fc1 = aligner_fc1
        self.aligner_hidden = aligner_hidden
        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.causal_mask_layer = causal_mask
        self.image_merge = image_merge
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_norm_eps = vision_norm_eps
        self.image_token_id = image_token_id

        num_patches = (image_size // patch_size) ** 2
        dummy_px = (
            (1, 3, image_size, image_size)
            if keras.config.image_data_format() == "channels_first"
            else (1, image_size, image_size, 3)
        )
        with inference_scope():
            self(
                {
                    "input_ids": ops.full(
                        (1, num_patches), image_token_id, dtype="int32"
                    ),
                    "attention_mask": ops.ones((1, num_patches), dtype="int32"),
                    "pixel_values": ops.zeros(dummy_px, dtype="float32"),
                }
            )

    def rope_tables(self, position_ids):
        return janus_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def get_image_features(self, pixel_values):
        features = self.vision_model(pixel_values)
        return self.aligner_hidden(
            ops.gelu(self.aligner_fc1(features), approximate=False)
        )

    def prepare_inputs(self, input_ids, pixel_values, attention_mask):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.token_embedding(
            ops.where(ops.equal(input_ids, self.image_token_id), 0, input_ids)
        )
        if pixel_values is not None:
            image_embeds = ops.reshape(
                self.get_image_features(pixel_values), (-1, self.embed_dim)
            )
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
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim"),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 10000.0),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision.get(
                "intermediate_size",
                int(vision["hidden_size"] * vision.get("mlp_ratio", 4.0)),
            ),
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": vision.get("image_size", 384),
            "patch_size": vision.get("patch_size", 16),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "image_token_id": hf_config.get("image_token_id", 100581),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_janus_hf_to_keras import transfer_janus_weights

        transfer_janus_weights(keras_model, hf_state_dict)

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
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_norm_eps": self.vision_norm_eps,
                "image_token_id": self.image_token_id,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class JanusConditionalGenerate(JanusModel, BaseGeneration):
    """Janus with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a bias-free ``lm_head`` on top of :class:`JanusModel`. The forward graph
    returns both ``logits`` and ``last_hidden_state``. Fast generation runs the
    vision tower + aligner + fused prefill ONCE in ``build_cache`` (consuming
    ``pixel_values``), then ``call_with_cache`` does text-only decode:

        gen.generate(input_ids, pixel_values=...)
    """

    # DeepSeek <｜end▁of▁sentence｜> stop id. Explicit generate() args override.
    eos_token_id = (100001,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
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
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
