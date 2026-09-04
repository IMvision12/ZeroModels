import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .qwen2_config import Qwen2Config
from .qwen2_layers import Qwen2DecoderLayer, Qwen2RMSNorm

MASK_NEG = -1e9


def qwen2_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    # cos/sin rotary tables for the given integer positions. Position-independent
    # inv_freq folds to a build-time constant, so this wires straight into the
    # functional graph (only the position product runs per call).
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def qwen2_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    head_dim,
    rope_theta,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = qwen2_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2Model(BaseModel):
    """Qwen2 dense decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x Qwen2DecoderLayer -> final RMSNorm``, with
    grouped-query attention and 1D rotary positions. A functional model: the
    forward is a static graph over ``input_ids`` / ``attention_mask``. Returns
    ``last_hidden_state``; use :class:`Qwen2TextGenerate` for logits / text.

        model = Qwen2Model.from_weights("qwen2-0.5b-instruct")
        out = model({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`Qwen2TextGenerate` ties the LM head to the
            token embedding instead of a separate projection.
    """

    HF_MODEL_TYPE = "qwen2"
    default_load_dtype = "bfloat16"
    config_class = Qwen2Config
    # Qwen2TextGenerate flips this on to also emit LM-head logits from the graph.
    output_logits = False

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=896,
        mlp_dim=4864,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=None,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_embeddings=True,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Qwen2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Qwen2RMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = qwen2_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
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

    def rope_tables(self, position_ids):
        # Imperative cos/sin for the KV-cache prefill / decode; the forward graph
        # wires qwen2_rope_tables directly.
        return qwen2_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config["num_key_value_heads"],
            "head_dim": hf_config.get("head_dim"),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen2_hf_to_keras import transfer_qwen2_weights

        transfer_qwen2_weights(keras_model, hf_state_dict)

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
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2TextGenerate(Qwen2Model, BaseGeneration):
    """Qwen2 backbone + a (tied) language-model head and fast ``.generate()``.

    Adds a vocabulary projection on top of :class:`Qwen2Model`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    (transposed) token embedding (weight tying). The forward graph returns both
    ``logits`` ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast
    generation comes from :class:`~zeromodels.base.BaseGeneration`, fulfilled
    here by ``build_cache`` (parallel prefill into a fixed KV cache) and
    ``call_with_cache`` (one compiled decode step). Constructor ``Args`` are
    inherited from :class:`Qwen2Model`.

        gen = Qwen2TextGenerate.from_weights("qwen2-0.5b-instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Default stop token: Qwen's <|im_end|> id (the generic BaseGeneration base
    # carries no model-specific eos). Explicit generate() args override this.
    eos_token_id = (151645,)
    output_logits = True
    # Ragged batches decode as one true batch (left-aligned + key_padding) rather
    # than being bucketed by length; see BaseGeneration.
    SUPPORTS_PADDED_DECODE = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill: run the prompt and write each layer's K/V into a
        # pre-allocated (batch, num_layers, 2, num_kv_heads, max_len, head_dim) cache.
        # Returns (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        # Positions are the slot indices (arange): any padding_mask here is
        # LEFT-aligned (generate() left-aligns before prefill), so the real tokens
        # sit flush against the right edge and arange gives them correct relative
        # geometry (RoPE is shift-invariant) while lining every row's last real
        # token up at prompt_len - 1 for a single shared decode position.
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal_ok = ki <= qi
        if padding_mask is not None:
            am = ops.cast(padding_mask, "bool")
            # A query may attend a causal key only if that key is real; keep the
            # diagonal open so a leading-pad query row is never fully masked (an
            # all-masked softmax is NaN, which would poison the cached pad K/V).
            key_ok = ops.logical_and(causal_ok[None], am[:, None, :])
            key_ok = ops.logical_or(key_ok, ops.cast(ops.eye(prompt_len), "bool")[None])
            causal = ops.cast(ops.where(key_ok, 0.0, MASK_NEG), "float32")[:, None]
        else:
            causal = ops.cast(ops.where(causal_ok, 0.0, MASK_NEG), "float32")[
                None, None
            ]
        hidden = self.token_embedding(token_ids)
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
            layer_caches.append(ops.stack([ck, cv], axis=1))  # (B, 2, nkv, max_len, hd)
        cache = ops.stack(layer_caches, axis=1)  # (B, num_layers, 2, nkv, max_len, hd)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index, key_padding=None):
        # One decode step: embed the single token, run every layer reading/writing
        # its cache slice at ``cache_update_index``, return (logits, updated cache).
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_t, sin_t = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        if key_padding is not None:
            # Hold the leading prompt pads (key_padding == 0) out of attention.
            key_mask = (
                key_mask
                + ops.cast(
                    ops.where(ops.cast(key_padding, "bool"), 0.0, MASK_NEG), "float32"
                )[:, None, None, :]
            )
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
