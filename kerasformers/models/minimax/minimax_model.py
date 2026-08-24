import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from kerasformers.base.base_mixin import inference_scope

from .minimax_config import MINIMAX_CONFIG, MINIMAX_WEIGHTS_URLS
from .minimax_layers import MiniMaxDecoderLayer, MiniMaxRMSNorm

MASK_NEG = -1e9


def minimax_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def minimax_backbone_features(
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
    # HF MiniMax derives positions with a plain arange (not mask cumsum); cumsum of
    # ones reproduces 0..seq-1 without a dynamic arange in the functional graph.
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = minimax_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype)
    full_mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(
            hidden, cos, sin, attention_mask=full_mask, padding_mask=attention_mask
        )
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxModel(BaseModel):
    """MiniMax-01 hybrid-attention MoE decoder (MiniMax-Text-01).

    Alternates lightning (linear) attention with full softmax attention per
    ``layer_types`` (the 456B checkpoint runs 7 lightning layers per softmax
    layer), every layer ending in a Mixtral-style softmax top-2 MoE, all wired
    with the MiniMax norm-first weighted residuals
    (``norm(x) * alpha + sublayer(norm(x)) * beta``). Rotary embeddings only
    feed the full-attention layers. A functional model; returns
    ``last_hidden_state``: use :class:`MiniMaxTextGenerate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Residual-stream width.
        mlp_dim: Per-expert SwiGLU hidden width.
        num_layers: Decoder blocks.
        num_heads / num_kv_heads / head_dim: Attention geometry (lightning
            layers use ``num_heads`` for q, k, and v alike).
        num_experts / num_experts_per_tok: MoE shape (32 / 2).
        layer_types: Per-layer ``"full_attention"`` / ``"linear_attention"``.
        block_size: Lightning-attention chunk length (256).
        full_attn_alpha / full_attn_beta / linear_attn_alpha /
        linear_attn_beta / mlp_alpha / mlp_beta: Residual weights.
        partial_rotary_factor: Fraction of head channels rotated by RoPE
            (the released checkpoints use 1.0, full rotation, matching the
            HF reference implementation).
        rope_theta: Rotary base frequency.
        norm_eps: RMSNorm epsilon.
        tie_embeddings: Whether :class:`MiniMaxTextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "minimax"
    BASE_MODEL_CONFIG = MINIMAX_CONFIG
    BASE_WEIGHT_CONFIG = MINIMAX_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=200064,
        embed_dim=6144,
        mlp_dim=9216,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_experts=32,
        num_experts_per_tok=2,
        layer_types=None,
        block_size=256,
        full_attn_alpha=1.0,
        full_attn_beta=1.0,
        linear_attn_alpha=1.0,
        linear_attn_beta=1.0,
        mlp_alpha=1.0,
        mlp_beta=1.0,
        partial_rotary_factor=1.0,
        rope_theta=10000000.0,
        norm_eps=1e-5,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        if layer_types is None:
            layer_types = tuple(
                "full_attention" if (i + 1) % 2 else "linear_attention"
                for i in range(num_layers)
            )
        layer_types = tuple(layer_types)
        head_dim = head_dim or embed_dim // num_heads
        rotary_dim = int(head_dim * partial_rotary_factor)

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = []
        for i, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                attn_alpha, attn_beta = linear_attn_alpha, linear_attn_beta
            else:
                attn_alpha, attn_beta = full_attn_alpha, full_attn_beta
            decoder_layers.append(
                MiniMaxDecoderLayer(
                    embed_dim,
                    mlp_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    num_experts,
                    num_experts_per_tok,
                    layer_type,
                    attn_alpha,
                    attn_beta,
                    mlp_alpha,
                    mlp_beta,
                    num_layers,
                    i,
                    block_size,
                    norm_eps,
                    name=f"decoder_layer_{i}",
                )
            )
        final_norm = MiniMaxRMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = minimax_backbone_features(
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
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.layer_types = layer_types
        self.block_size = block_size
        self.full_attn_alpha = full_attn_alpha
        self.full_attn_beta = full_attn_beta
        self.linear_attn_alpha = linear_attn_alpha
        self.linear_attn_beta = linear_attn_beta
        self.mlp_alpha = mlp_alpha
        self.mlp_beta = mlp_beta
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.rotary_dim = rotary_dim

        # The lightning (linear) attention layer's forward aborts Keras' symbolic
        # auto-build trace on some backends, leaving sublayers unbuilt after
        # __init__; a concrete dummy forward materializes every weight so
        # from_weights (which loads before any forward) has a complete model.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def rope_tables(self, position_ids):
        return minimax_rope_tables(
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
            "layer_types": tuple(hf_config.get("layer_types") or ()) or None,
            "block_size": hf_config.get("block_size", 256),
            "full_attn_alpha": hf_config.get("full_attn_alpha_factor", 1.0),
            "full_attn_beta": hf_config.get("full_attn_beta_factor", 1.0),
            "linear_attn_alpha": hf_config.get("linear_attn_alpha_factor", 1.0),
            "linear_attn_beta": hf_config.get("linear_attn_beta_factor", 1.0),
            "mlp_alpha": hf_config.get("mlp_alpha_factor", 1.0),
            "mlp_beta": hf_config.get("mlp_beta_factor", 1.0),
            "partial_rotary_factor": rope.get(
                "partial_rotary_factor", hf_config.get("partial_rotary_factor", 1.0)
            ),
            "rope_theta": rope.get(
                "rope_theta", hf_config.get("rope_theta", 10000000.0)
            ),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_minimax_hf_to_keras import transfer_minimax_weights

        transfer_minimax_weights(keras_model, hf_state_dict)

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
                "layer_types": self.layer_types,
                "block_size": self.block_size,
                "full_attn_alpha": self.full_attn_alpha,
                "full_attn_beta": self.full_attn_beta,
                "linear_attn_alpha": self.linear_attn_alpha,
                "linear_attn_beta": self.linear_attn_beta,
                "mlp_alpha": self.mlp_alpha,
                "mlp_beta": self.mlp_beta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_theta": self.rope_theta,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxTextGenerate(MiniMaxModel, BaseGeneration):
    """MiniMax with an LM head + fast ``.generate()``.

    The hybrid cache is a per-layer tuple: full-attention layers carry a
    fixed ``(batch, 2, kv_heads, max_len, head_dim)`` KV buffer, lightning
    layers carry the constant-size ``(batch, heads, head_dim, head_dim)``
    running KV state, so the decode loop stays compilable and lightning
    layers never grow with sequence length.
    """

    # MiniMax-Text-01 <end_of_sentence>. Explicit generate() args override.
    eos_token_id = (200020,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        full_mask = self.causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            hidden, piece = layer(
                hidden,
                cos,
                sin,
                attention_mask=full_mask,
                padding_mask=padding_mask,
                use_cache=True,
            )
            if layer.layer_type == "linear_attention":
                caches.append(piece)
            else:
                k, v = piece
                ck = ops.slice_update(
                    ops.zeros(
                        (batch, self.num_kv_heads, max_len, self.head_dim),
                        dtype=k.dtype,
                    ),
                    (0, 0, 0, 0),
                    k,
                )
                cv = ops.slice_update(
                    ops.zeros(
                        (batch, self.num_kv_heads, max_len, self.head_dim),
                        dtype=v.dtype,
                    ),
                    (0, 0, 0, 0),
                    v,
                )
                caches.append(ops.stack([ck, cv], axis=1))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        max_len = None
        for piece, layer in zip(cache, self.decoder_layers):
            if layer.layer_type != "linear_attention":
                max_len = int(piece.shape[3])
                break
        key_mask = None
        if max_len is not None:
            key_mask = ops.cast(
                ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
            )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, piece = layer.decode_step(h, cos, sin, cache[i], pos, key_mask)
            new_cache.append(piece)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
