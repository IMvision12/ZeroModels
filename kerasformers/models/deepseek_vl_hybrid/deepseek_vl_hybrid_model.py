import keras
from keras import layers, ops

from kerasformers.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
    merge_media,
)
from kerasformers.base.base_mixin import inference_scope
from kerasformers.models.deepseek_vl.deepseek_vl_layers import (
    DeepseekVLTextDecoderLayer,
    DeepseekVLTextRMSNorm,
)
from kerasformers.models.deepseek_vl.deepseek_vl_model import DeepseekVLVisionModel

from .deepseek_vl_hybrid_config import DeepseekVLHybridConfig
from .deepseek_vl_hybrid_layers import (
    DeepseekVLHybridAligner,
    DeepseekVLHybridSamEncoder,
    DeepseekVLSamVisionNeck,
    DeepseekVLSamVisionProj,
)

MASK_NEG = -1e9

# The backbone (DeepseekVLHybridModel) and generative head
# (DeepseekVLHybridConditionalGenerate) share the variant's weights repo, whose
# kf_config.json declares DeepseekVLHybridModel.
DEEPSEEK_VL_HYBRID_HUB_SIBLINGS = frozenset(
    {"DeepseekVLHybridModel", "DeepseekVLHybridConditionalGenerate"}
)


def deepseek_vl_hybrid_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


@keras.saving.register_keras_serializable(package="kerasformers")
class HybridImageFeatures(layers.Layer):
    """Weightless wrapper that runs both vision towers + fusion + aligner eagerly.

    Holds the vision sublayers and the ``high_res_vision_alpha`` weight by
    non-tracked references (they stay tracked on the model, so their saved-weight
    paths are unchanged). An explicit output spec keeps the SAM tuple output, the
    dynamic reshapes, and the live-``alpha`` fusion out of the symbolic build (a
    build-time ``glob * alpha`` would bake ``alpha``'s zero init); the combination
    runs at (eager) runtime with concrete shapes. Returns flattened image tokens
    ``(num_image_tokens, embed_dim)`` ready for :class:`MediaMerge`.
    """

    def __init__(
        self,
        vision_model,
        high_res_vision_model,
        high_res_vision_neck,
        high_res_vision_proj,
        aligner,
        alpha,
        embed_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        object.__setattr__(
            self,
            "refs",
            (
                vision_model,
                high_res_vision_model,
                high_res_vision_neck,
                high_res_vision_proj,
                aligner,
                alpha,
            ),
        )

    def call(self, pixel_values, high_res_pixel_values):
        vision_model, hr_model, hr_neck, hr_proj, aligner, alpha = self.refs
        low = vision_model(pixel_values)
        last, global_state = hr_model(high_res_pixel_values)
        last = hr_proj(last)
        glob = hr_proj(hr_neck(global_state))
        out = last + glob * ops.cast(alpha, last.dtype)
        out = ops.reshape(out, (ops.shape(out)[0], -1, ops.shape(out)[-1]))
        features = aligner(low, out)
        return ops.reshape(features, (-1, self.embed_dim))

    def compute_output_spec(self, pixel_values, high_res_pixel_values):
        return keras.KerasTensor((None, self.embed_dim), dtype=self.compute_dtype)


def deepseek_vl_hybrid_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    high_res_pixel_values,
    *,
    token_embedding,
    image_features,
    image_merge,
    decoder_layers,
    final_norm,
    causal_mask,
    image_token_id,
    head_dim,
    rope_theta,
    compute_dtype,
):
    media = ops.equal(input_ids, image_token_id)
    hidden = token_embedding(ops.where(media, 0, input_ids))
    embeds = image_features(pixel_values, high_res_pixel_values)
    hidden = image_merge(hidden, input_ids, embeds)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = deepseek_vl_hybrid_rope_tables(
        position_ids, head_dim, rope_theta, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridModel(BaseModel):
    """DeepSeek-VL Hybrid multimodal backbone (no LM head).

    A low-res SigLIP tower and a high-res SAM tower whose features are fused
    (``low + aligner(high * alpha)``) and scattered into the ``image_token_id``
    slots of a LLaMA-style decoder. A functional model: the forward is a static
    graph over ``input_ids`` / ``attention_mask`` / ``pixel_values`` /
    ``high_res_pixel_values`` (media always provided; a dummy 1-image clip makes
    the merge a no-op for text-only). Use
    :class:`DeepseekVLHybridConditionalGenerate` for logits / text.
    """

    HF_MODEL_TYPE = "deepseek_vl_hybrid"
    config_class = DeepseekVLHybridConfig
    HUB_REPO_SIBLINGS = DEEPSEEK_VL_HYBRID_HUB_SIBLINGS
    output_logits = False

    def __init__(
        self,
        vocab_size=102400,
        embed_dim=4096,
        mlp_dim=11008,
        num_layers=30,
        num_heads=32,
        num_kv_heads=32,
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
        high_res_embed_dim=768,
        high_res_mlp_dim=3072,
        high_res_num_layers=12,
        high_res_num_heads=12,
        high_res_image_size=1024,
        high_res_patch_size=16,
        high_res_output_channels=256,
        high_res_window_size=14,
        high_res_global_attn_indexes=(2, 5, 8, 11),
        high_res_norm_eps=1e-6,
        image_token_id=100015,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads

        vision_model = DeepseekVLVisionModel(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            image_size,
            patch_size,
            vision_norm_eps,
            name="vision_model",
        )
        high_res_vision_model = DeepseekVLHybridSamEncoder(
            high_res_embed_dim,
            high_res_num_layers,
            high_res_num_heads,
            high_res_mlp_dim,
            high_res_image_size,
            high_res_patch_size,
            high_res_output_channels,
            high_res_window_size,
            tuple(high_res_global_attn_indexes),
            high_res_norm_eps,
            name="high_res_vision_model",
        )
        high_res_vision_neck = DeepseekVLSamVisionNeck(
            high_res_output_channels, high_res_norm_eps, name="high_res_vision_neck"
        )
        high_res_vision_proj = DeepseekVLSamVisionProj(
            high_res_output_channels,
            image_size // patch_size,
            name="high_res_vision_proj",
        )
        high_res_vision_alpha = keras.Variable(
            initializer=keras.initializers.Zeros(),
            shape=(1,),
            trainable=True,
            name="high_res_vision_alpha",
        )
        aligner = DeepseekVLHybridAligner(embed_dim, name="aligner")
        image_features = HybridImageFeatures(
            vision_model,
            high_res_vision_model,
            high_res_vision_neck,
            high_res_vision_proj,
            aligner,
            high_res_vision_alpha,
            embed_dim,
            name="image_features",
        )
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            DeepseekVLTextDecoderLayer(
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
        final_norm = DeepseekVLTextRMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=(
                    (3, image_size, image_size)
                    if keras.config.image_data_format() == "channels_first"
                    else (image_size, image_size, 3)
                ),
                dtype="float32",
                name="pixel_values",
            ),
            "high_res_pixel_values": layers.Input(
                shape=(
                    (3, high_res_image_size, high_res_image_size)
                    if keras.config.image_data_format() == "channels_first"
                    else (high_res_image_size, high_res_image_size, 3)
                ),
                dtype="float32",
                name="high_res_pixel_values",
            ),
        }
        hidden = deepseek_vl_hybrid_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["high_res_pixel_values"],
            token_embedding=token_embedding,
            image_features=image_features,
            image_merge=image_merge,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            image_token_id=image_token_id,
            head_dim=head_dim,
            rope_theta=rope_theta,
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
        self.high_res_vision_model = high_res_vision_model
        self.high_res_vision_neck = high_res_vision_neck
        self.high_res_vision_proj = high_res_vision_proj
        self.high_res_vision_alpha = high_res_vision_alpha
        self.aligner = aligner
        self.image_features = image_features
        self.image_merge = image_merge
        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.causal_mask_layer = causal_mask
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
        self.high_res_embed_dim = high_res_embed_dim
        self.high_res_mlp_dim = high_res_mlp_dim
        self.high_res_num_layers = high_res_num_layers
        self.high_res_num_heads = high_res_num_heads
        self.high_res_image_size = high_res_image_size
        self.high_res_patch_size = high_res_patch_size
        self.high_res_output_channels = high_res_output_channels
        self.high_res_window_size = high_res_window_size
        self.high_res_global_attn_indexes = tuple(high_res_global_attn_indexes)
        self.high_res_norm_eps = high_res_norm_eps
        self.image_token_id = image_token_id

        # The dual vision towers + aligner do not all auto-build during Keras'
        # symbolic construction; a concrete dummy forward materializes every weight
        # so from_weights (which loads before any forward) has a complete model.
        num_patches = (image_size // patch_size) ** 2
        cf = keras.config.image_data_format() == "channels_first"
        px = (1, 3, image_size, image_size) if cf else (1, image_size, image_size, 3)
        hr_px = (
            (1, 3, high_res_image_size, high_res_image_size)
            if cf
            else (1, high_res_image_size, high_res_image_size, 3)
        )
        with inference_scope():
            self(
                {
                    "input_ids": ops.full(
                        (1, num_patches), image_token_id, dtype="int32"
                    ),
                    "attention_mask": ops.ones((1, num_patches), dtype="int32"),
                    "pixel_values": ops.zeros(px, dtype="float32"),
                    "high_res_pixel_values": ops.zeros(hr_px, dtype="float32"),
                }
            )

    def get_high_res_features(self, high_res_pixel_values):
        last, global_state = self.high_res_vision_model(high_res_pixel_values)
        last = self.high_res_vision_proj(last)
        glob = self.high_res_vision_neck(global_state)
        glob = self.high_res_vision_proj(glob)
        out = last + glob * ops.cast(self.high_res_vision_alpha, last.dtype)
        b = ops.shape(out)[0]
        return ops.reshape(out, (b, -1, ops.shape(out)[-1]))

    def get_image_features(self, pixel_values, high_res_pixel_values):
        low = self.vision_model(pixel_values)
        high = self.get_high_res_features(high_res_pixel_values)
        return self.aligner(low, high)

    def rope_tables(self, position_ids):
        return deepseek_vl_hybrid_rope_tables(
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

    def prepare_inputs(
        self, input_ids, pixel_values, high_res_pixel_values, attention_mask
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        media = ops.equal(input_ids, self.image_token_id)
        inputs_embeds = self.token_embedding(ops.where(media, 0, input_ids))
        if pixel_values is not None:
            image_embeds = ops.reshape(
                self.get_image_features(pixel_values, high_res_pixel_values),
                (-1, self.embed_dim),
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
        high = hf_config["high_res_vision_config"]
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
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": vision.get("image_size", 384),
            "patch_size": vision.get("patch_size", 16),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "high_res_embed_dim": high["hidden_size"],
            "high_res_mlp_dim": high.get("mlp_dim", high.get("intermediate_size")),
            "high_res_num_layers": high["num_hidden_layers"],
            "high_res_num_heads": high["num_attention_heads"],
            "high_res_image_size": high.get("image_size", 1024),
            "high_res_patch_size": high.get("patch_size", 16),
            "high_res_output_channels": high.get("output_channels", 256),
            "high_res_window_size": high.get("window_size", 14),
            "high_res_global_attn_indexes": tuple(
                high.get("global_attn_indexes", (2, 5, 8, 11))
            ),
            "high_res_norm_eps": high.get("layer_norm_eps", 1e-6),
            "image_token_id": hf_config.get("image_token_id", 100015),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_deepseek_vl_hybrid_hf_to_keras import (
            transfer_deepseek_vl_hybrid_weights,
        )

        transfer_deepseek_vl_hybrid_weights(keras_model, hf_state_dict)

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
                "high_res_embed_dim": self.high_res_embed_dim,
                "high_res_mlp_dim": self.high_res_mlp_dim,
                "high_res_num_layers": self.high_res_num_layers,
                "high_res_num_heads": self.high_res_num_heads,
                "high_res_image_size": self.high_res_image_size,
                "high_res_patch_size": self.high_res_patch_size,
                "high_res_output_channels": self.high_res_output_channels,
                "high_res_window_size": self.high_res_window_size,
                "high_res_global_attn_indexes": self.high_res_global_attn_indexes,
                "high_res_norm_eps": self.high_res_norm_eps,
                "image_token_id": self.image_token_id,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridConditionalGenerate(DeepseekVLHybridModel, BaseGeneration):
    """DeepSeek-VL Hybrid with an LM head + fast ``.generate()``.

    Adds a bias-free ``lm_head`` on top of :class:`DeepseekVLHybridModel`.
    ``build_cache`` runs both vision towers + aligner + fused prefill ONCE
    (consuming ``pixel_values`` and ``high_res_pixel_values``), then
    ``call_with_cache`` does text-only decode:

        gen.generate(input_ids, pixel_values=..., high_res_pixel_values=...)
    """

    eos_token_id = (100001,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        high_res_pixel_values=None,
    ):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden, position_ids = self.prepare_inputs(
            token_ids, pixel_values, high_res_pixel_values, padding_mask
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
