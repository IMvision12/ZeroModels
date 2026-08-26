import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .mixtral_config import MIXTRAL_CONFIG, MIXTRAL_WEIGHTS_URLS
from .mixtral_layers import MixtralDecoderLayer, MixtralRMSNorm

MASK_NEG = -1e9


def mixtral_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    # cos/sin rotary tables; ``inv_freq`` folds to a build-time constant, only the
    # position product runs per call, so this wires directly into the graph.
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def mixtral_backbone_features(
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
    cos, sin = mixtral_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class MixtralModel(BaseModel):
    """Mixtral sparse-MoE decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x MixtralDecoderLayer -> final RMSNorm``:
    a Mistral-style attention stack whose every feed-forward is an
    8-expert top-2 softmax-routed mixture of experts. (This port evaluates
    all experts densely and combines by the routing weights: mathematically
    identical to sparse routing, compute O(num_experts).) A functional model;
    returns ``last_hidden_state``: use :class:`MixtralTextGenerate` for
    logits / text.

        model = MixtralModel.from_weights("mixtral-8x7b-instruct")
        out = model({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: Per-expert SwiGLU hidden width.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        num_experts: Routed expert count (8).
        num_experts_per_tok: Top-k experts per token (2).
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        sliding_window: Causal attention window; ``None`` (the released
            checkpoints) means full causal attention.
        tie_embeddings: Whether :class:`MixtralTextGenerate` ties the LM head
            (Mixtral checkpoints do not).
    """

    HF_MODEL_TYPE = "mixtral"
    BASE_MODEL_CONFIG = MIXTRAL_CONFIG
    BASE_WEIGHT_CONFIG = MIXTRAL_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=32000,
        embed_dim=4096,
        mlp_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=None,
        num_experts=8,
        num_experts_per_tok=2,
        norm_eps=1e-5,
        rope_theta=1000000.0,
        sliding_window=None,
        tie_embeddings=False,
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
            MixtralDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_experts,
                num_experts_per_tok,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = MixtralRMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(sliding_window=sliding_window, name="causal_mask")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = mixtral_backbone_features(
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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.tie_embeddings = tie_embeddings

    def rope_tables(self, position_ids):
        return mixtral_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        keep = ki <= qi
        if self.sliding_window is not None:
            keep = ops.logical_and(keep, ki > qi - self.sliding_window)
        mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

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
            "head_dim": hf_config.get("head_dim"),
            "num_experts": hf_config.get("num_local_experts", 8),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 2),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "sliding_window": hf_config.get("sliding_window"),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mixtral_hf_to_keras import transfer_mixtral_weights

        transfer_mixtral_weights(keras_model, hf_state_dict)

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
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "sliding_window": self.sliding_window,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MixtralTextGenerate(MixtralModel, BaseGeneration):
    """Mixtral backbone + a language-model head and fast ``.generate()``.

    Adds a bias-free ``lm_head`` on top of :class:`MixtralModel` (Mixtral does
    not tie embeddings). The forward graph returns ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast generation
    comes from :class:`~zeromodels.base.BaseGeneration`, fulfilled here by
    ``build_cache`` (parallel prefill into a fixed KV cache) and
    ``call_with_cache`` (one compiled decode step). Constructor ``Args`` are
    inherited from :class:`MixtralModel`.

        gen = MixtralTextGenerate.from_weights("mixtral-8x7b-instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    eos_token_id = (2,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden = self.token_embedding(token_ids)
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
        ar = ops.arange(max_len)
        keep = ar <= pos
        if self.sliding_window is not None:
            keep = ops.logical_and(keep, ar > pos - self.sliding_window)
        key_mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
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
