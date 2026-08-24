import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .minimax_m2_config import MINIMAX_M2_CONFIG, MINIMAX_M2_WEIGHTS_URLS
from .minimax_m2_layers import MiniMaxM2DecoderLayer, MiniMaxM2RMSNorm

MASK_NEG = -1e9


def minimax_m2_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def minimax_m2_backbone_features(
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
    # Plain arange positions (matches the imperative build_cache path), via cumsum
    # of ones so the functional graph needs no dynamic-length arange.
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = minimax_m2_rope_tables(
        position_ids, rotary_dim, rope_theta, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxM2Model(BaseModel):
    """MiniMax-M2 MoE decoder (230B-A10B).

    Standard pre-norm GQA transformer with full-width QK RMSNorm and a
    sigmoid-scored top-8-of-256 MoE on every layer (DeepSeek-style aux-free
    selection bias; the gathered weights stay the unbiased sigmoid scores).
    A functional model; returns ``last_hidden_state``: use
    :class:`MiniMaxM2TextGenerate` for logits / text.

    Args:
        vocab_size: Token vocabulary size (200064).
        embed_dim: Residual-stream width (3072).
        mlp_dim: Per-expert SwiGLU hidden width (1536).
        num_layers: Decoder blocks (62).
        num_heads / num_kv_heads / head_dim: 48 / 8 / 128.
        num_experts / num_experts_per_tok: 256 / 8.
        partial_rotary_factor: Fraction of head channels rotated by RoPE
            (the released checkpoint runs 1.0, full rotation, matching the
            HF reference implementation).
        rope_theta: Rotary base frequency (5e6).
        norm_eps: RMSNorm epsilon (1e-6).
        tie_embeddings: Whether :class:`MiniMaxM2TextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "minimax_m2"
    BASE_MODEL_CONFIG = MINIMAX_M2_CONFIG
    BASE_WEIGHT_CONFIG = MINIMAX_M2_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=200064,
        embed_dim=3072,
        mlp_dim=1536,
        num_layers=62,
        num_heads=48,
        num_kv_heads=8,
        head_dim=128,
        num_experts=256,
        num_experts_per_tok=8,
        partial_rotary_factor=1.0,
        rope_theta=5000000.0,
        norm_eps=1e-6,
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
            MiniMaxM2DecoderLayer(
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
        final_norm = MiniMaxM2RMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = minimax_m2_backbone_features(
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
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.rotary_dim = rotary_dim

    def rope_tables(self, position_ids):
        return minimax_m2_rope_tables(
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
            "num_experts": hf_config.get("num_local_experts", 256),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 8),
            "partial_rotary_factor": rope.get(
                "partial_rotary_factor", hf_config.get("partial_rotary_factor", 1.0)
            ),
            "rope_theta": rope.get(
                "rope_theta", hf_config.get("rope_theta", 5000000.0)
            ),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_minimax_m2_hf_to_keras import transfer_minimax_m2_weights

        transfer_minimax_m2_weights(keras_model, hf_state_dict)

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
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_theta": self.rope_theta,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxM2TextGenerate(MiniMaxM2Model, BaseGeneration):
    """MiniMax-M2 with an LM head + fast ``.generate()``."""

    # MiniMax-M2 eos `[e~[`. Explicit generate() args override.
    eos_token_id = (200020,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(seq, padding_mask)
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
