import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from kerasformers.base.base_mixin import inference_scope

from .gemma2_config import Gemma2Config
from .gemma2_layers import Gemma2DecoderLayer, Gemma2RMSNorm

MASK_NEG = -1e9


def gemma2_is_sliding(layer_idx):
    # HF: "sliding_attention" if bool((i + 1) % 2) -> even layers slide.
    return bool((layer_idx + 1) % 2)


def gemma2_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def gemma2_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    full_mask_layer,
    sliding_mask_layer,
    embed_dim,
    head_dim,
    rope_theta,
    compute_dtype,
):
    hidden = token_embedding(input_ids) * ops.cast(embed_dim**0.5, compute_dtype)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = gemma2_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    full_mask = full_mask_layer(input_ids, attention_mask)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    for i, layer in enumerate(decoder_layers):
        mask = sliding_mask if gemma2_is_sliding(i) else full_mask
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma2Model(BaseModel):
    """Gemma 2 decoder-only transformer backbone (no LM head).

    Gemma's scaled embeddings, ``(1 + w)`` RMSNorms, and GeGLU, plus the
    Gemma 2 additions: a four-norm sandwich around every residual branch,
    attention-logit tanh softcapping (50.0) applied before the mask,
    ``query_pre_attn_scalar`` attention scaling, and alternating
    sliding-window (even layers) / full (odd layers) causal attention.
    A functional model; returns ``last_hidden_state``: use
    :class:`Gemma2TextGenerate` for logits / text (which also applies the
    final-logit softcap, 30.0).

        model = Gemma2Model.from_weights("kerasformers/gemma-2-2b")
        out = model({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim.
        query_pre_attn_scalar: Attention scaling denominator.
        attn_logit_softcapping: Attention tanh softcap (``None`` disables).
        final_logit_softcapping: LM-head tanh softcap (``None`` disables).
        sliding_window: Window of the sliding (even) layers.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`Gemma2TextGenerate` ties the LM head
            (Gemma 2 checkpoints do).
    """

    HF_MODEL_TYPE = "gemma2"
    default_load_dtype = "bfloat16"
    config_class = Gemma2Config
    output_logits = False

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=2304,
        mlp_dim=9216,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        query_pre_attn_scalar=256.0,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=4096,
        norm_eps=1e-6,
        rope_theta=10000.0,
        tie_embeddings=True,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Gemma2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                query_pre_attn_scalar,
                attn_logit_softcapping,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Gemma2RMSNorm(eps=norm_eps, name="final_norm")
        full_mask_layer = CausalMask(name="causal_mask")
        sliding_mask_layer = CausalMask(
            sliding_window=sliding_window, name="sliding_mask"
        )
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = gemma2_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            full_mask_layer=full_mask_layer,
            sliding_mask_layer=sliding_mask_layer,
            embed_dim=embed_dim,
            head_dim=head_dim,
            rope_theta=rope_theta,
            compute_dtype=token_embedding.compute_dtype,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            raw = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(token_embedding, name="lm_head")(hidden)
            )
            if final_logit_softcapping is not None:
                raw = ops.tanh(raw / final_logit_softcapping) * final_logit_softcapping
            outputs["logits"] = raw

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
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
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.sliding_window = sliding_window
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings

        # Gemma's ``(1 + w)`` RMSNorm aborts Keras' symbolic auto-build trace on
        # some backends, leaving attention/MLP sublayers unbuilt; a concrete dummy
        # forward materializes every weight so ``from_weights`` (which loads before
        # any forward) has a complete model to populate.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def is_sliding(self, layer_idx):
        return gemma2_is_sliding(layer_idx)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids):
        return gemma2_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
        )

    def build_masks(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
        sliding_keep = ops.logical_and(causal, ki > qi - self.sliding_window)
        sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
            None, None
        ]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim", 256),
            "query_pre_attn_scalar": hf_config.get("query_pre_attn_scalar", 256.0),
            "attn_logit_softcapping": hf_config.get("attn_logit_softcapping"),
            "final_logit_softcapping": hf_config.get("final_logit_softcapping"),
            "sliding_window": hf_config.get("sliding_window", 4096),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma2_hf_to_keras import transfer_gemma2_weights

        transfer_gemma2_weights(keras_model, hf_state_dict)

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
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "final_logit_softcapping": self.final_logit_softcapping,
                "sliding_window": self.sliding_window,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma2TextGenerate(Gemma2Model, BaseGeneration):
    """Gemma 2 backbone + a (tied) LM head with final-logit softcapping and
    fast ``.generate()``.

    The vocabulary projection (tied token embedding) is followed by
    ``tanh(logits / 30) * 30`` when ``final_logit_softcapping`` is set:
    matching the Gemma 2 checkpoints. The forward graph returns both ``logits``
    and ``last_hidden_state``. Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration` via ``build_cache`` /
    ``call_with_cache``, respecting the per-layer full / sliding masks.
    Constructor ``Args`` are inherited from :class:`Gemma2Model`.

        gen = Gemma2TextGenerate.from_weights("kerasformers/gemma-2-2b-it")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Gemma <eos> / <end_of_turn> stop ids. Explicit generate() args override.
    eos_token_id = (1, 107)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = ops.tanh(logits / cap) * cap
        return logits

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill into a fixed (B, num_layers, 2, num_kv_heads,
        # max_len, head_dim) cache with per-layer full / sliding masks.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        full_mask, sliding_mask = self.build_masks(prompt_len, padding_mask)
        hidden = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=mask, use_cache=True
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
        # One decode step; sliding layers see only (pos - window, pos].
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
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
            km = sliding_km if self.is_sliding(i) else full_km
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, km
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
