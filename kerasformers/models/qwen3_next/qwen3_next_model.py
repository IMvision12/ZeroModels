import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .qwen3_next_config import (
    QWEN3_NEXT_CONFIG,
    QWEN3_NEXT_WEIGHTS_URLS,
    Qwen3NextConfig,
)
from .qwen3_next_layers import Qwen3NextDecoderLayer, Qwen3NextRMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextModel(SubclassedBaseModel):
    """Qwen3.5 (Qwen3-Next) hybrid decoder-only backbone (no LM head).

    ``token_embedding -> num_layers x Qwen3NextDecoderLayer -> final RMSNorm``. The
    decoder layers alternate two token mixers: most are Gated-DeltaNet *linear
    attention* (conv1d + delta-rule recurrence), and every ``full_attention_interval``
    -th layer is *gated full attention* (GQA, QK-norm, partial rotary, sigmoid
    output gate). RMSNorm is zero-centered. This is a subclassed (imperative)
    :class:`BaseModel`: the forward pass runs eagerly with ``keras.ops``. Returns
    raw features; use :class:`Qwen3NextTextGenerate` for logits / text.

        model = Qwen3NextModel.from_weights("hf:Qwen/Qwen3.5-...")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads in the full-attention layers.
        num_kv_heads: Key/value heads in the full-attention layers (GQA).
        head_dim: Per-head dim of the full-attention layers.
        norm_eps: RMSNorm epsilon (shared everywhere, incl. QK-norm).
        rope_theta: Rotary base frequency.
        partial_rotary_factor: Fraction of ``head_dim`` that gets rotary
            (``rotary_dim = int(head_dim * partial_rotary_factor)``).
        tie_embeddings: Whether :class:`Qwen3NextTextGenerate` ties the LM head to the
            token embedding instead of a separate projection.
        full_attention_interval: Place a full-attention layer every Nth block;
            all others are Gated-DeltaNet linear-attention layers.
        linear_conv_kernel_dim: Causal conv1d kernel width in the linear layers.
        linear_key_head_dim, linear_value_head_dim: Per-head dims of the linear
            attention key/value.
        linear_num_key_heads, linear_num_value_heads: Head counts of the linear
            attention key/value.
    """

    HF_MODEL_TYPE = "qwen3_next"
    default_load_dtype = "bfloat16"
    config_class = Qwen3NextConfig
    BASE_MODEL_CONFIG = QWEN3_NEXT_CONFIG
    BASE_WEIGHT_CONFIG = QWEN3_NEXT_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=5120,
        num_layers=48,
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_embeddings=False,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_experts=512,
        num_experts_per_tok=10,
        moe_mlp_dim=512,
        shared_mlp_dim=512,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.tie_embeddings = tie_embeddings
        self.full_attention_interval = full_attention_interval
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = tuple(mlp_only_layers)
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        layer_cfg = {
            "embed_dim": embed_dim,
            "mlp_dim": mlp_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": self.rotary_dim,
            "norm_eps": norm_eps,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_mlp_dim": moe_mlp_dim,
            "shared_mlp_dim": shared_mlp_dim,
            "norm_topk_prob": norm_topk_prob,
        }
        self.layer_types = [
            "full_attention"
            if (i + 1) % full_attention_interval == 0
            else "linear_attention"
            for i in range(num_layers)
        ]
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3NextDecoderLayer(
                layer_cfg,
                self.layer_types[i],
                use_moe=(
                    i not in self.mlp_only_layers
                    and num_experts > 0
                    and (i + 1) % decoder_sparse_step == 0
                ),
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3NextRMSNorm(eps=norm_eps, name="final_norm")

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        hidden = self.token_embedding(input_ids)
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        inv_freq = 1.0 / ops.power(
            self.rope_theta,
            ops.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos, sin = ops.cos(emb), ops.sin(emb)
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        attn_mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        pad_mask = None
        if attention_mask is not None:
            am_f = ops.cast(am, "float32")
            attn_mask = attn_mask + (1.0 - am_f)[:, None, None, :] * MASK_NEG
            pad_mask = am_f[:, :, None]
        for layer in self.decoder_layers:
            hidden = layer(
                hidden, cos, sin, attention_mask=attn_mask, pad_mask=pad_mask
            )
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        c = hf_config.get("text_config", hf_config)
        rope = c.get("rope_parameters", c)
        return {
            "vocab_size": c["vocab_size"],
            "embed_dim": c["hidden_size"],
            "mlp_dim": c["intermediate_size"],
            "num_layers": c["num_hidden_layers"],
            "num_heads": c["num_attention_heads"],
            "num_kv_heads": c["num_key_value_heads"],
            "head_dim": c["head_dim"],
            "norm_eps": c.get("rms_norm_eps", 1e-6),
            "rope_theta": rope.get("rope_theta", c.get("rope_theta", 10000000.0)),
            "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
            "tie_embeddings": c.get("tie_word_embeddings", True),
            "full_attention_interval": c["full_attention_interval"],
            "linear_conv_kernel_dim": c["linear_conv_kernel_dim"],
            "linear_key_head_dim": c["linear_key_head_dim"],
            "linear_value_head_dim": c["linear_value_head_dim"],
            "linear_num_key_heads": c["linear_num_key_heads"],
            "linear_num_value_heads": c["linear_num_value_heads"],
            "num_experts": c.get("num_experts", 512),
            "num_experts_per_tok": c.get("num_experts_per_tok", 10),
            "moe_mlp_dim": c.get("moe_intermediate_size", 512),
            "shared_mlp_dim": c.get("shared_expert_intermediate_size", 512),
            "norm_topk_prob": bool(c.get("norm_topk_prob", True)),
            "decoder_sparse_step": c.get("decoder_sparse_step", 1),
            "mlp_only_layers": tuple(c.get("mlp_only_layers") or ()),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen3_next_hf_to_keras import transfer_qwen3_next_weights

        transfer_qwen3_next_weights(keras_model, hf_state_dict)

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
                "partial_rotary_factor": self.partial_rotary_factor,
                "tie_embeddings": self.tie_embeddings,
                "full_attention_interval": self.full_attention_interval,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextTextGenerate(Qwen3NextModel, BaseGeneration):
    """Qwen3.5 backbone + a language-model head and fast ``.generate()``.

    Adds a vocabulary projection on top of :class:`Qwen3NextModel` (a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, else the tied token
    embedding). ``call`` returns both ``logits`` and ``last_hidden_state``. Fast
    generation comes from :class:`~kerasformers.base.BaseGeneration`, fulfilled here
    by ``build_cache`` / ``call_with_cache`` over a **hybrid per-layer cache**: a
    fixed-slot ``(key, value)`` for the full-attention layers and a
    ``(conv_state, recurrent_state)`` for the Gated-DeltaNet layers (whose recurrence
    is identical to prefill, so its decode step is exact). Constructor ``Args`` are
    inherited from :class:`Qwen3NextModel`.

        gen = Qwen3NextTextGenerate.from_weights("hf:Qwen/Qwen3.5-...")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Qwen3.5 default stop id. Explicit generate() args override this.
    eos_token_id = (151645, 151643)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        logits = (
            self.lm_head(hidden)
            if self.lm_head is not None
            else ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
        )
        return {"logits": logits, "last_hidden_state": hidden}

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def rope_tables(self, position_ids):
        # Partial-rotary cos/sin over the first ``rotary_dim`` channels (float32, to
        # match the eager call()).
        inv_freq = 1.0 / ops.power(
            self.rope_theta,
            ops.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return ops.cos(emb), ops.sin(emb)

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill into a HYBRID per-layer cache: fixed-slot (k, v) for the
        # full-attention layers, (conv_state, recurrent_state) for the DeltaNet ones.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv, hd = self.num_kv_heads, self.head_dim
        pad_mask = None
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
            am_f = ops.cast(am, "float32")
            pad_mask = am_f[:, :, None]
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if padding_mask is not None:
            causal = causal + (1.0 - am_f)[:, None, None, :] * MASK_NEG
        hidden = self.token_embedding(token_ids)
        cache = []
        for i, layer in enumerate(self.decoder_layers):
            hidden, state = layer(
                hidden,
                cos,
                sin,
                attention_mask=causal,
                use_cache=True,
                pad_mask=pad_mask,
            )
            if self.layer_types[i] == "full_attention":
                k, v = state
                ck = ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
                )
                cv = ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
                )
                cache.append((ck, cv))
            else:
                cache.append(state)
        cache = tuple(cache)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # One decode step: DeltaNet layers advance their recurrence; full-attention
        # layers write into their fixed KV slot at ``pos`` and read [0, pos].
        batch = int(token_ids.shape[0])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        full_idx = next(
            (i for i, t in enumerate(self.layer_types) if t == "full_attention"), None
        )
        key_mask = None
        if full_idx is not None:
            max_len = int(cache[full_idx][0].shape[2])
            key_mask = ops.cast(
                ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
            )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, state = layer.decode_step(h, cos, sin, cache[i], pos, key_mask)
            new_cache.append(state)
        cache = tuple(new_cache)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
