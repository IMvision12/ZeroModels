import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .glm_config import GlmConfig
from .glm_layers import GlmDecoderLayer, GlmRMSNorm

MASK_NEG = -1e9


def glm_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype):
    # Partial interleaved rope: one angle per channel pair, repeat-interleaved.
    # inv_freq folds to a build-time constant; only the position product runs per
    # call, so this wires directly into the functional graph.
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.repeat(freqs, 2, axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def glm_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    rotary_dim,
    rope_theta,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    # GLM uses plain arange positions (padding does not shift them).
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = glm_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class GlmModel(BaseModel):
    """GLM-4 decoder backbone (no LM head).

    Pre-norm decoder with grouped-query attention, partial *interleaved* rotary
    embeddings (``partial_rotary_factor`` of each head is rotated), biased q/k/v
    projections, and a fused-SwiGLU MLP. A functional model: the forward is a
    static graph over ``input_ids`` / ``attention_mask``. Returns raw features;
    use :class:`GlmTextGenerate` for logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim: Geometry.
        mlp_dim: SwiGLU hidden width (``intermediate_size``).
        partial_rotary_factor: Fraction of each head that receives rotary.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        attention_bias: Whether q/k/v carry bias.
        tie_embeddings: Whether the head ties to the token embedding.
    """

    HF_MODEL_TYPE = "glm"
    config_class = GlmConfig
    # GlmTextGenerate flips this on to also emit LM-head logits from the graph.
    output_logits = False

    def __init__(
        self,
        vocab_size=151552,
        embed_dim=4096,
        num_layers=40,
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
        mlp_dim=13696,
        partial_rotary_factor=0.5,
        norm_eps=0.00000015625,
        rope_theta=10000.0,
        attention_bias=True,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads
        rotary_dim = int(head_dim * partial_rotary_factor)

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            GlmDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = GlmRMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = glm_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            rotary_dim=rotary_dim,
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

        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.causal_mask_layer = causal_mask
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.tie_embeddings = tie_embeddings
        self.rotary_dim = rotary_dim

    def rope_tables(self, position_ids):
        # Imperative cos/sin for the KV-cache prefill / decode; the forward graph
        # wires glm_rope_tables directly.
        return glm_rope_tables(
            position_ids, self.rotary_dim, self.rope_theta, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    @classmethod
    def config_from_hf(cls, hf_config):
        rope = hf_config.get("rope_parameters") or {}
        prf = rope.get("partial_rotary_factor")
        if prf is None:
            prf = hf_config.get("partial_rotary_factor", 0.5)
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim"),
            "mlp_dim": hf_config["intermediate_size"],
            "partial_rotary_factor": prf,
            "norm_eps": hf_config.get("rms_norm_eps", 0.00000015625),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 10000.0)),
            "attention_bias": bool(hf_config.get("attention_bias", True)),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_glm_hf_to_keras import transfer_glm_weights

        transfer_glm_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "mlp_dim": self.mlp_dim,
                "partial_rotary_factor": self.partial_rotary_factor,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GlmTextGenerate(GlmModel, BaseGeneration):
    """GLM-4 with an LM head + fast ``.generate()``."""

    eos_token_id = (151329, 151336, 151338)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos_p, sin_p, attention_mask=causal, use_cache=True
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
        cos_t, sin_t = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos_t, sin_t, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
