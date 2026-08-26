import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from zeromodels.base.base_mixin import inference_scope

from .gemma_config import GemmaConfig
from .gemma_layers import GemmaDecoderLayer, GemmaRMSNorm

MASK_NEG = -1e9


def gemma_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def gemma_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    embed_dim,
    head_dim,
    rope_theta,
    compute_dtype,
):
    # Gemma multiplies token embeddings by sqrt(embed_dim).
    hidden = token_embedding(input_ids) * ops.cast(embed_dim**0.5, compute_dtype)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = gemma_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class GemmaModel(BaseModel):
    """Gemma decoder-only transformer backbone (no LM head).

    ``token_embedding * sqrt(embed_dim) -> num_layers x GemmaDecoderLayer ->
    final RMSNorm`` with the Gemma signature pieces: ``(1 + w)`` RMSNorm,
    GeGLU (tanh-approximate gelu) MLPs, scaled token embeddings, ``head_dim``
    256 decoupled from ``embed_dim // num_heads`` (the 2B is MQA with one
    K/V head), and a tied LM head. A functional model; returns
    ``last_hidden_state``: use :class:`GemmaTextGenerate` for logits / text.

        model = GemmaModel.from_weights("zeromodels/gemma-2b")
        out = model({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer.
        head_dim: Per-head dim.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`GemmaTextGenerate` ties the LM head to the
            token embedding (Gemma checkpoints do).
    """

    HF_MODEL_TYPE = "gemma"
    default_load_dtype = "bfloat16"
    config_class = GemmaConfig
    output_logits = False

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=2048,
        mlp_dim=16384,
        num_layers=18,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
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
            GemmaDecoderLayer(
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
        final_norm = GemmaRMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = gemma_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            embed_dim=embed_dim,
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

        # Gemma's ``(1 + w)`` RMSNorm aborts Keras' symbolic auto-build trace on
        # some backends, leaving the attention/MLP sublayers unbuilt; a concrete
        # dummy forward materializes every weight so ``from_weights`` (which loads
        # before any forward) has a complete model to populate.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def embed_scaled(self, input_ids):
        # Gemma multiplies token embeddings by sqrt(embed_dim).
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids):
        return gemma_rope_tables(
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
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim", 256),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma_hf_to_keras import transfer_gemma_weights

        transfer_gemma_weights(keras_model, hf_state_dict)

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
class GemmaTextGenerate(GemmaModel, BaseGeneration):
    """Gemma backbone + a (tied) language-model head and fast ``.generate()``.

    Adds the vocabulary projection on top of :class:`GemmaModel`, the
    transposed scaled token embedding when ``tie_embeddings`` (all Gemma
    checkpoints), else a separate bias-free ``lm_head``. The forward graph returns
    both ``logits`` ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast
    generation comes from :class:`~zeromodels.base.BaseGeneration` via
    ``build_cache`` / ``call_with_cache``. Constructor ``Args`` are inherited
    from :class:`GemmaModel`.

        gen = GemmaTextGenerate.from_weights("zeromodels/gemma-2b-it")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Gemma <eos> / <end_of_turn> stop ids. Explicit generate() args override.
    eos_token_id = (1, 107)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill into a fixed (B, num_layers, 2, num_kv_heads,
        # max_len, head_dim) cache. Returns (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if padding_mask is not None:
            causal = (
                causal + (1.0 - ops.cast(am, "float32"))[:, None, None, :] * MASK_NEG
            )
        hidden = self.embed_scaled(token_ids)
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
        # One decode step at position ``cache_update_index``.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
