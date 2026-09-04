import math

import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .llama_config import LLAMA_CONFIG, LLAMA_WEIGHTS_URLS
from .llama_layers import LlamaDecoderLayer, LlamaRMSNorm

MASK_NEG = -1e9


def llama3_scaled_inv_freq(
    head_dim, base, factor, low_freq_factor, high_freq_factor, original_max_pos
):
    """Llama-3.1 frequency-banded rope scaling (port of HF's ``llama3`` rope type).

    Long-wavelength (low-frequency) bands are divided by ``factor``,
    short-wavelength bands are kept, and the band between
    ``original_max_pos / low_freq_factor`` and
    ``original_max_pos / high_freq_factor`` is smoothly interpolated. Returns
    the scaled ``(head_dim // 2,)`` inverse-frequency tensor.
    """
    inv_freq = 1.0 / ops.power(
        float(base), ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq
    scaled = ops.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Guard the degenerate low==high case (e.g. Llama 4 Scout): the medium band
    # is then empty, so the smoothed values are never selected.
    denom = high_freq_factor - low_freq_factor
    denom = denom if denom != 0 else 1.0
    smooth = (original_max_pos / wavelen - low_freq_factor) / denom
    smoothed = (1.0 - smooth) * scaled / factor + smooth * scaled
    is_medium = ops.logical_and(
        wavelen >= high_freq_wavelen, wavelen <= low_freq_wavelen
    )
    return ops.where(is_medium, smoothed, scaled)


def llama_rope_tables(
    position_ids,
    head_dim,
    rope_theta,
    rope_factor,
    rope_low_freq_factor,
    rope_high_freq_factor,
    rope_original_max_pos,
    compute_dtype,
):
    # cos/sin rotary tables for the given integer positions, in the compute dtype,
    # using the llama3-scaled inverse frequencies when rope_factor is set (Llama
    # 3.1 / 3.2 / 3.3) and plain ones otherwise (Llama 3). Position-independent
    # ``inv_freq`` folds to a build-time constant; only the position product runs
    # per call, so this is safe to wire directly into the functional graph.
    if rope_factor is not None:
        inv_freq = llama3_scaled_inv_freq(
            head_dim,
            rope_theta,
            rope_factor,
            rope_low_freq_factor,
            rope_high_freq_factor,
            rope_original_max_pos,
        )
    else:
        inv_freq = 1.0 / ops.power(
            rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
        )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def llama_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    rope_kwargs,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = llama_rope_tables(
        position_ids, compute_dtype=compute_dtype, **rope_kwargs
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class LlamaModel(BaseModel):
    """Llama 3 family decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x LlamaDecoderLayer -> final RMSNorm``,
    with grouped-query attention (8 KV heads on every variant), bias-free qkv
    projections, SwiGLU MLPs, and half-rotation rotary positions. Covers the
    whole ``model_type: "llama"`` 3.x line: Llama 3 (plain rope), Llama
    3.1 / 3.3 (frequency-banded ``llama3`` rope scaling, ``rope_factor=8``),
    and Llama 3.2 1B/3B (``rope_factor=32`` and a tied LM head). A functional
    model: the forward is a static graph over ``input_ids`` / ``attention_mask``.
    Returns ``last_hidden_state``; use :class:`LlamaTextGenerate` for logits / text.

        model = LlamaModel.from_weights("llama3.2-1b")
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
        rope_factor: ``llama3`` rope-scaling factor; ``None`` disables scaling
            (Llama 3). 8.0 for 3.1/3.3, 32.0 for 3.2.
        rope_low_freq_factor / rope_high_freq_factor: band edges of the
            scaled / interpolated wavelength regions.
        rope_original_max_pos: Pretraining context the bands are relative to.
        tie_embeddings: Whether :class:`LlamaTextGenerate` ties the LM head to the
            token embedding (Llama 3.2 1B/3B: ``True``).
    """

    HF_MODEL_TYPE = "llama"
    BASE_MODEL_CONFIG = LLAMA_CONFIG
    BASE_WEIGHT_CONFIG = LLAMA_WEIGHTS_URLS
    # LlamaTextGenerate flips this on to also emit LM-head logits from the graph.
    output_logits = False

    def __init__(
        self,
        vocab_size=128256,
        embed_dim=2048,
        mlp_dim=8192,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=None,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_factor=32.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_pos=8192,
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
            LlamaDecoderLayer(
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
        final_norm = LlamaRMSNorm(eps=norm_eps, name="final_norm")
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
        rope_kwargs = {
            "head_dim": head_dim,
            "rope_theta": rope_theta,
            "rope_factor": rope_factor,
            "rope_low_freq_factor": rope_low_freq_factor,
            "rope_high_freq_factor": rope_high_freq_factor,
            "rope_original_max_pos": rope_original_max_pos,
        }
        hidden = llama_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            rope_kwargs=rope_kwargs,
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
        self.rope_factor = rope_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_original_max_pos = rope_original_max_pos
        self.tie_embeddings = tie_embeddings

    def rope_tables(self, position_ids):
        # Imperative cos/sin for the KV-cache prefill / decode (concrete positions);
        # the forward graph wires llama_rope_tables directly.
        return llama_rope_tables(
            position_ids,
            self.head_dim,
            self.rope_theta,
            self.rope_factor,
            self.rope_low_freq_factor,
            self.rope_high_freq_factor,
            self.rope_original_max_pos,
            self.compute_dtype,
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        rope_scaling = hf_config.get("rope_scaling") or {}
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
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "rope_theta": hf_config.get("rope_theta", 500000.0),
            "rope_factor": rope_scaling.get("factor"),
            "rope_low_freq_factor": rope_scaling.get("low_freq_factor", 1.0),
            "rope_high_freq_factor": rope_scaling.get("high_freq_factor", 4.0),
            "rope_original_max_pos": rope_scaling.get(
                "original_max_position_embeddings", 8192
            ),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_llama_hf_to_keras import transfer_llama_weights

        transfer_llama_weights(keras_model, hf_state_dict)

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
                "rope_factor": self.rope_factor,
                "rope_low_freq_factor": self.rope_low_freq_factor,
                "rope_high_freq_factor": self.rope_high_freq_factor,
                "rope_original_max_pos": self.rope_original_max_pos,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LlamaTextGenerate(LlamaModel, BaseGeneration):
    """Llama 3 family backbone + a language-model head and fast ``.generate()``.

    Adds a vocabulary projection on top of :class:`LlamaModel`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False`` (3-8B and up),
    otherwise the transposed token embedding (3.2 1B/3B weight tying).
    The forward graph returns both ``logits`` ``(batch, seq, vocab_size)`` and
    ``last_hidden_state``. Fast generation comes from
    :class:`~zeromodels.base.BaseGeneration`, fulfilled here by
    ``build_cache`` (parallel prefill into a fixed KV cache) and
    ``call_with_cache`` (one compiled decode step). Constructor ``Args`` are
    inherited from :class:`LlamaModel`.

        gen = LlamaTextGenerate.from_weights("llama3.2-1b-instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Llama 3 stop ids: <|end_of_text|>, <|eom_id|>, <|eot_id|>. Explicit
    # generate() args override this.
    eos_token_id = (128001, 128008, 128009)
    output_logits = True
    SUPPORTS_PADDED_DECODE = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill: run the prompt and write each layer's K/V into a
        # pre-allocated (B, num_layers, 2, num_kv_heads, max_len, head_dim)
        # cache. Returns (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal_ok = ki <= qi
        if padding_mask is not None:
            am = ops.cast(padding_mask, "bool")
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

    def call_with_cache(self, token_ids, cache, cache_update_index, key_padding=None):
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        if key_padding is not None:
            pad_block = ops.cast(
                ops.where(ops.cast(key_padding, "bool"), 0.0, MASK_NEG), "float32"
            )[:, None, None, :]
            key_mask = key_mask + pad_block
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
