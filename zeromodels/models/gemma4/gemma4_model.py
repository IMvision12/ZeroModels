import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    TextOnlyGeneration,
    TiedHead,
)
from zeromodels.base.base_mixin import inference_scope

from .gemma4_config import Gemma4Config, Gemma4TextConfig
from .gemma4_layers import (
    Gemma4AudioLayer,
    Gemma4AudioRelPositionalEncoding,
    Gemma4AudioSubSampleConvProjection,
    Gemma4DecoderLayer,
    Gemma4MultimodalEmbedder,
    Gemma4RMSNorm,
    Gemma4VisionEncoderLayer,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionRotaryEmbedding,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4Reshape4D(layers.Layer):
    """Weightless ``(b, s, num_layers*per_dim) -> (b, s, num_layers, per_dim)`` reshape.

    An explicit output spec keeps the dynamic ``(b, s)`` out of the symbolic build
    (a bare graph reshape would bake ``None`` into the Reshape node); it runs at
    (eager) runtime with concrete shapes.
    """

    def __init__(self, num_layers, per_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.per_dim = per_dim

    def call(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        return ops.reshape(x, (b, s, self.num_layers, self.per_dim))

    def compute_output_spec(self, x):
        return keras.KerasTensor(
            (x.shape[0], x.shape[1], self.num_layers, self.per_dim), dtype=x.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "per_dim": self.per_dim})
        return config


def gemma4_rope_tables(position_ids, head_dim, rot_dim, theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, rot_dim, 2, dtype="float32") / head_dim
    )
    if rot_dim < head_dim:
        inv_freq = ops.concatenate(
            [inv_freq, ops.zeros(((head_dim - rot_dim) // 2,), dtype="float32")], axis=0
        )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return (
        ops.cast(ops.cos(emb), compute_dtype),
        ops.cast(ops.sin(emb), compute_dtype),
    )


def gemma4_per_layer_inputs(
    input_ids,
    inputs_embeds,
    *,
    embed_tokens_per_layer,
    per_layer_model_projection,
    per_layer_projection_norm,
    reshape_4d,
    hidden_size_per_layer_input,
    embed_dim,
    compute_dtype,
):
    # PLE: token-identity embedding (scaled) + a context projection of the scaled
    # main embedding, combined as (proj + identity) / sqrt(2).
    ple = embed_tokens_per_layer(input_ids) * ops.cast(
        hidden_size_per_layer_input**0.5, compute_dtype
    )
    ple = reshape_4d(ple)
    proj = per_layer_model_projection(inputs_embeds) * ops.cast(
        embed_dim**-0.5, compute_dtype
    )
    proj = reshape_4d(proj)
    proj = per_layer_projection_norm(proj)
    return (proj + ple) * ops.cast(2.0**-0.5, compute_dtype)


def gemma4_run_layers(
    hidden,
    cos_l,
    sin_l,
    cos_g,
    sin_g,
    full_mask,
    sliding_mask,
    *,
    decoder_layers,
    layer_types,
    num_kv_shared_layers,
    first_kv_shared,
    final_norm,
    per_layer_inputs=None,
):
    # ``shared`` holds the K/V of the last non-shared layer per attention type,
    # which the tail shared layers reuse (transformers KV-sharing).
    shared = {}
    for i, layer in enumerate(decoder_layers):
        sliding = layer_types[i] != "full_attention"
        layer_type = "sliding" if sliding else "global"
        cos, sin, mask = (
            (cos_l, sin_l, sliding_mask) if sliding else (cos_g, sin_g, full_mask)
        )
        pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        is_shared = num_kv_shared_layers > 0 and i >= first_kv_shared
        hidden, kv = layer(
            hidden,
            cos,
            sin,
            attention_mask=mask,
            shared_kv=shared.get(layer_type) if is_shared else None,
            per_layer_input=pli,
        )
        if num_kv_shared_layers > 0 and not is_shared:
            shared[layer_type] = kv
    return final_norm(hidden)


def gemma4_text_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    embed_tokens_per_layer,
    per_layer_model_projection,
    per_layer_projection_norm,
    reshape_4d,
    decoder_layers,
    final_norm,
    full_mask_layer,
    sliding_mask_layer,
    num_layers,
    layer_types,
    hidden_size_per_layer_input,
    embed_dim,
    head_dim,
    global_head_dim,
    global_rot_dim,
    rope_theta,
    rope_local_theta,
    num_kv_shared_layers,
    first_kv_shared,
    compute_dtype,
):
    hidden = token_embedding(input_ids) * ops.cast(embed_dim**0.5, compute_dtype)
    per_layer_inputs = None
    if hidden_size_per_layer_input:
        per_layer_inputs = gemma4_per_layer_inputs(
            input_ids,
            hidden,
            embed_tokens_per_layer=embed_tokens_per_layer,
            per_layer_model_projection=per_layer_model_projection,
            per_layer_projection_norm=per_layer_projection_norm,
            reshape_4d=reshape_4d,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            embed_dim=embed_dim,
            compute_dtype=compute_dtype,
        )
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos_l, sin_l = gemma4_rope_tables(
        position_ids, head_dim, head_dim, rope_local_theta, compute_dtype
    )
    cos_g, sin_g = gemma4_rope_tables(
        position_ids, global_head_dim, global_rot_dim, rope_theta, compute_dtype
    )
    full_mask = full_mask_layer(input_ids, attention_mask)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    return gemma4_run_layers(
        hidden,
        cos_l,
        sin_l,
        cos_g,
        sin_g,
        full_mask,
        sliding_mask,
        decoder_layers=decoder_layers,
        layer_types=layer_types,
        num_kv_shared_layers=num_kv_shared_layers,
        first_kv_shared=first_kv_shared,
        final_norm=final_norm,
        per_layer_inputs=per_layer_inputs,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4Model(BaseModel):
    """Gemma 4 text decoder backbone (no LM head).

    Gemma's scaled embeddings and ``(1 + w)`` norms with Gemma 4's per-layer
    attention geometry: sliding layers (5:1 pattern) use ``head_dim`` 256
    with full default rope (theta 1e4); global layers use
    ``global_head_dim`` 512 with few K/V heads, ``K = V`` attention (no value
    projection: the value is the weightlessly-normed key projection), and
    "proportional" *partial* rotary (the first quarter of the head, theta
    1e6). Attention scores are unscaled; per-head q/k norms carry the scale.
    Feed-forwards are GeGLU; on the 26B-A4B a parallel 128-expert top-8
    branch (per-expert-scaled router) is added. Each layer's output is
    multiplied by a learned ``layer_scalar``. This is the text tower only; the
    vision and audio towers of the multimodal checkpoints live in
    :class:`Gemma4MultimodalModel` (also in this module; loading a multimodal
    checkpoint here transfers just its ``model.*`` text
    weights). The E2B/E4B "Elastic" variants add Per-Layer Embeddings
    (``hidden_size_per_layer_input``), tail layers that share an earlier layer's
    K/V (``num_kv_shared_layers``), and an optional double-wide MLP on those
    shared layers. Returns raw features; use :class:`Gemma4ConditionalGenerate`.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: K/V heads on sliding layers.
        num_global_kv_heads: K/V heads on global layers.
        head_dim: Sliding-layer per-head dim (256).
        global_head_dim: Global-layer per-head dim (512).
        k_eq_v: Global layers reuse the key projection as the value.
        enable_moe: Whether layers carry the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim: MoE parameters.
        sliding_window: Window of the sliding layers.
        sliding_window_pattern: Every ``pattern``-th layer is global (6).
        partial_rotary_factor: Fraction of the global head that is rotated.
        final_logit_softcapping: LM-head tanh softcap (30.0).
        norm_eps: RMSNorm epsilon.
        rope_theta: Global-layer rotary base (1e6).
        rope_local_theta: Sliding-layer rotary base (1e4).
        tie_embeddings: Whether :class:`Gemma4ConditionalGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = ("gemma4", "gemma4_text")
    config_class = Gemma4TextConfig
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    def __init__(
        self,
        vocab_size=262144,
        embed_dim=3840,
        mlp_dim=15360,
        num_layers=48,
        num_heads=16,
        num_kv_heads=8,
        num_global_kv_heads=1,
        head_dim=256,
        global_head_dim=512,
        k_eq_v=True,
        enable_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        sliding_window=1024,
        sliding_window_pattern=6,
        layer_types=None,
        partial_rotary_factor=0.25,
        final_logit_softcapping=30.0,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        tie_embeddings=True,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262144,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        global_rot_dim = 2 * int(partial_rotary_factor * global_head_dim // 2)
        # Layers at index >= first_kv_shared reuse an earlier layer's K/V (E-variants).
        first_kv_shared = num_layers - num_kv_shared_layers
        # Per-layer sliding/global schedule. Honor the checkpoint's explicit
        # ``layer_types`` when present (E2B/E4B place global layers on a 5:1
        # schedule); otherwise derive from ``sliding_window_pattern``.
        resolved_layer_types = (
            list(layer_types)
            if layer_types
            else [
                "full_attention"
                if (i + 1) % sliding_window_pattern == 0
                else "sliding_attention"
                for i in range(num_layers)
            ]
        )

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        embed_tokens_per_layer = None
        per_layer_model_projection = None
        per_layer_projection_norm = None
        reshape_4d = None
        if hidden_size_per_layer_input:
            embed_tokens_per_layer = layers.Embedding(
                vocab_size_per_layer_input,
                num_layers * hidden_size_per_layer_input,
                name="embed_tokens_per_layer",
            )
            per_layer_model_projection = layers.Dense(
                num_layers * hidden_size_per_layer_input,
                use_bias=False,
                name="per_layer_model_projection",
            )
            per_layer_projection_norm = Gemma4RMSNorm(
                eps=norm_eps, name="per_layer_projection_norm"
            )
            reshape_4d = Gemma4Reshape4D(
                num_layers, hidden_size_per_layer_input, name="ple_reshape"
            )
        decoder_layers = []
        for i in range(num_layers):
            sliding = resolved_layer_types[i] != "full_attention"
            is_kv_shared = num_kv_shared_layers > 0 and i >= first_kv_shared
            layer_mlp_dim = (
                mlp_dim * 2 if (use_double_wide_mlp and is_kv_shared) else mlp_dim
            )
            decoder_layers.append(
                Gemma4DecoderLayer(
                    embed_dim,
                    layer_mlp_dim,
                    num_heads,
                    num_kv_heads if sliding else num_global_kv_heads,
                    head_dim if sliding else global_head_dim,
                    k_eq_v=(not sliding) and k_eq_v,
                    is_kv_shared=is_kv_shared,
                    hidden_size_per_layer_input=hidden_size_per_layer_input,
                    is_moe=enable_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_mlp_dim=moe_mlp_dim,
                    norm_eps=norm_eps,
                    name=f"decoder_layer_{i}",
                )
            )
        final_norm = Gemma4RMSNorm(eps=norm_eps, name="final_norm")
        full_mask_layer = CausalMask(name="full_mask")
        sliding_mask_layer = CausalMask(
            sliding_window=sliding_window, name="sliding_mask"
        )

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = gemma4_text_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            embed_tokens_per_layer=embed_tokens_per_layer,
            per_layer_model_projection=per_layer_model_projection,
            per_layer_projection_norm=per_layer_projection_norm,
            reshape_4d=reshape_4d,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            full_mask_layer=full_mask_layer,
            sliding_mask_layer=sliding_mask_layer,
            num_layers=num_layers,
            layer_types=resolved_layer_types,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            embed_dim=embed_dim,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            global_rot_dim=global_rot_dim,
            rope_theta=rope_theta,
            rope_local_theta=rope_local_theta,
            num_kv_shared_layers=num_kv_shared_layers,
            first_kv_shared=first_kv_shared,
            compute_dtype=token_embedding.compute_dtype,
        )
        super().__init__(
            inputs=inputs,
            outputs={"last_hidden_state": hidden},
            name=name or type(self).__name__,
            **kwargs,
        )

        self.token_embedding = token_embedding
        self.embed_tokens_per_layer = embed_tokens_per_layer
        self.per_layer_model_projection = per_layer_model_projection
        self.per_layer_projection_norm = per_layer_projection_norm
        self.reshape_4d = reshape_4d
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.full_mask_layer = full_mask_layer
        self.sliding_mask_layer = sliding_mask_layer
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_global_kv_heads = num_global_kv_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.k_eq_v = k_eq_v
        self.enable_moe = enable_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.partial_rotary_factor = partial_rotary_factor
        self.final_logit_softcapping = final_logit_softcapping
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.tie_embeddings = tie_embeddings
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        self.global_rot_dim = global_rot_dim
        self.first_kv_shared = first_kv_shared
        self.layer_types = resolved_layer_types

        # Gemma's ``(1 + w)`` RMSNorm aborts Keras' symbolic auto-build on some
        # backends; a concrete dummy forward materializes every weight so
        # ``from_weights`` (which loads before any forward) has a complete model.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def is_sliding(self, layer_idx):
        return self.layer_types[layer_idx] != "full_attention"

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids, local):
        if local:
            hd, rot = self.head_dim, self.head_dim
            theta = self.rope_local_theta
        else:
            hd, rot = self.global_head_dim, self.global_rot_dim
            theta = self.rope_theta
        inv_freq = 1.0 / ops.power(theta, ops.arange(0, rot, 2, dtype="float32") / hd)
        if rot < hd:
            inv_freq = ops.concatenate(
                [inv_freq, ops.zeros(((hd - rot) // 2,), dtype="float32")], axis=0
            )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def compute_position_ids(self, attention_mask, batch, seq):
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            return ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        return ops.broadcast_to(ops.arange(seq), (batch, seq))

    def build_masks(self, seq, attention_mask=None, block_ids=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        within = ki > qi - self.sliding_window
        if block_ids is None:
            full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
            sliding_keep = ops.logical_and(causal, within)
            sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
                None, None
            ]
        else:
            q_grp = block_ids[:, :, None]
            kv_grp = block_ids[:, None, :]
            block = ops.logical_and(q_grp == kv_grp, q_grp >= 0)
            full_keep = ops.broadcast_to(causal, ops.shape(block))
            sliding_keep = ops.logical_and(ops.logical_or(causal, block), within)
            full = ops.cast(ops.where(full_keep, 0.0, MASK_NEG), "float32")[:, None]
            sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
                :, None
            ]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    def compute_per_layer_inputs(self, input_ids, inputs_embeds):
        # PLE: token-identity embedding (scaled) + a context projection of the
        # scaled main embedding, combined as (proj + identity) / sqrt(2). Shape
        # (batch, seq, num_layers, hidden_size_per_layer_input). The (b, s, L, d)
        # reshape goes through reshape_4d, whose compute_output_spec keeps the
        # dynamic (b, s) out of the graph; a bare ops.reshape here bakes None as
        # the batch dim and crashes the multimodal functional graph on replay.
        ple = self.embed_tokens_per_layer(input_ids) * ops.cast(
            self.hidden_size_per_layer_input**0.5, self.compute_dtype
        )
        ple = self.reshape_4d(ple)
        proj = self.per_layer_model_projection(inputs_embeds) * ops.cast(
            self.embed_dim**-0.5, self.compute_dtype
        )
        proj = self.reshape_4d(proj)
        proj = self.per_layer_projection_norm(proj)
        return (proj + ple) * ops.cast(2.0**-0.5, self.compute_dtype)

    def run_layers(
        self,
        hidden,
        cos_l,
        sin_l,
        cos_g,
        sin_g,
        full_mask,
        sliding_mask,
        per_layer_inputs=None,
    ):
        # ``shared`` holds the K/V of the last non-shared layer per attention type,
        # which the shared layers at the tail reuse (transformers KV-sharing).
        shared = {}
        for i, layer in enumerate(self.decoder_layers):
            sliding = self.is_sliding(i)
            layer_type = "sliding" if sliding else "global"
            cos, sin, mask = (
                (cos_l, sin_l, sliding_mask) if sliding else (cos_g, sin_g, full_mask)
            )
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            is_shared = self.num_kv_shared_layers > 0 and i >= self.first_kv_shared
            hidden, kv = layer(
                hidden,
                cos,
                sin,
                attention_mask=mask,
                shared_kv=shared.get(layer_type) if is_shared else None,
                per_layer_input=pli,
            )
            if self.num_kv_shared_layers > 0 and not is_shared:
                shared[layer_type] = kv
        return self.final_norm(hidden)

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        rope = text.get("rope_parameters") or {}
        full_rope = rope.get("full_attention") or {}
        sliding_rope = rope.get("sliding_attention") or {}
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "num_global_kv_heads": text.get("num_global_key_value_heads")
            or text.get("num_key_value_heads", 1),
            "head_dim": text.get("head_dim", 256),
            "global_head_dim": text.get("global_head_dim", 512),
            "k_eq_v": bool(text.get("attention_k_eq_v", False)),
            "enable_moe": bool(text.get("enable_moe_block", False)),
            "num_experts": text.get("num_experts") or 0,
            "num_experts_per_tok": text.get("top_k_experts") or 0,
            "moe_mlp_dim": text.get("moe_intermediate_size") or 0,
            "sliding_window": text.get("sliding_window", 1024),
            "sliding_window_pattern": text.get("sliding_window_pattern", 6),
            "layer_types": text.get("layer_types"),
            "partial_rotary_factor": full_rope.get("partial_rotary_factor", 0.25),
            "final_logit_softcapping": text.get("final_logit_softcapping"),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": full_rope.get(
                "rope_theta", text.get("rope_theta", 1000000.0)
            ),
            "rope_local_theta": sliding_rope.get(
                "rope_theta", text.get("rope_local_base_freq", 10000.0)
            ),
            "tie_embeddings": text.get("tie_word_embeddings", True),
            "hidden_size_per_layer_input": text.get("hidden_size_per_layer_input", 0)
            or 0,
            "vocab_size_per_layer_input": text.get(
                "vocab_size_per_layer_input", 262144
            ),
            "num_kv_shared_layers": text.get("num_kv_shared_layers", 0) or 0,
            "use_double_wide_mlp": bool(text.get("use_double_wide_mlp", False)),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_hf_to_keras import transfer_gemma4_weights

        transfer_gemma4_weights(keras_model, hf_state_dict)

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
                "num_global_kv_heads": self.num_global_kv_heads,
                "head_dim": self.head_dim,
                "global_head_dim": self.global_head_dim,
                "k_eq_v": self.k_eq_v,
                "enable_moe": self.enable_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "layer_types": self.layer_types,
                "partial_rotary_factor": self.partial_rotary_factor,
                "final_logit_softcapping": self.final_logit_softcapping,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "tie_embeddings": self.tie_embeddings,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "num_kv_shared_layers": self.num_kv_shared_layers,
                "use_double_wide_mlp": self.use_double_wide_mlp,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4VisionModel(layers.Layer):
    """Gemma 4 vision encoder: patch embed, rotary transformer, spatial pool."""

    def __init__(
        self,
        hidden_size=768,
        num_layers=16,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        intermediate_size=3072,
        patch_size=16,
        position_embedding_size=10240,
        pooling_kernel_size=3,
        rope_theta=100.0,
        eps=1e-6,
        standardize=False,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.pooling_kernel_size = pooling_kernel_size
        self.standardize = standardize
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            hidden_size, patch_size, position_embedding_size, name="patch_embedder"
        )
        self.rotary_emb = Gemma4VisionRotaryEmbedding(
            head_dim, rope_theta, name="rotary_emb"
        )
        self.vlayers = [
            Gemma4VisionEncoderLayer(
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                eps,
                use_clipped_linears,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.pooler = Gemma4VisionPooler(hidden_size, name="pooler")

    def build(self, input_shape):
        if self.standardize:
            self.std_bias = self.add_weight(
                shape=(self.hidden_size,),
                initializer="zeros",
                trainable=False,
                name="std_bias",
            )
            self.std_scale = self.add_weight(
                shape=(self.hidden_size,),
                initializer="ones",
                trainable=False,
                name="std_scale",
            )

    def call(self, pixel_values, pixel_position_ids, attention_mask=None):
        padding = ops.all(pixel_position_ids == -1, axis=-1)
        h = self.patch_embedder(pixel_values, pixel_position_ids, padding)
        cos, sin = self.rotary_emb(pixel_position_ids)
        for layer in self.vlayers:
            h = layer(h, cos, sin, attention_mask)
        num_patches = int(ops.shape(pixel_values)[1])
        output_length = num_patches // (self.pooling_kernel_size**2)
        hidden = self.pooler(
            h, pixel_position_ids, padding, output_length=output_length
        )
        if self.standardize:
            hidden = (hidden - ops.cast(self.std_bias, "float32")) * ops.cast(
                self.std_scale, "float32"
            )
        return ops.cast(hidden, h.dtype)

    def compute_output_spec(
        self, pixel_values, pixel_position_ids, attention_mask=None
    ):
        return keras.KerasTensor(
            (pixel_values.shape[0], None, self.hidden_size), dtype=self.compute_dtype
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4AudioModel(layers.Layer):
    """Gemma 4 audio encoder (USM conformer): subsample, conformer stack, projection."""

    def __init__(
        self,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        conv_channels=(128, 32),
        conv_kernel_size=5,
        chunk_size=12,
        context_left=13,
        context_right=0,
        logit_cap=50.0,
        invalid_logits=-1e9,
        residual_weight=0.5,
        norm_eps=1e-6,
        output_proj_dims=1536,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.max_past_horizon = context_left - 1
        self.max_future_horizon = context_right
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            conv_channels, hidden_size, norm_eps, name="subsample_conv_projection"
        )
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(
            hidden_size, self.context_size, name="rel_pos_enc"
        )
        self.alayers = [
            Gemma4AudioLayer(
                hidden_size,
                num_heads,
                chunk_size,
                context_left,
                context_right,
                conv_kernel_size,
                norm_eps,
                residual_weight,
                logit_cap,
                invalid_logits,
                use_clipped_linears,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.output_proj = layers.Dense(
            output_proj_dims, use_bias=True, name="output_proj"
        )

    def blocked_mask(self, valid_mask, seq_len):
        # valid_mask: [B, seq_len] bool (True = valid frame). Build the 4D
        # sliding-window bidirectional mask then fold it to blocked 5D
        # [B, 1, num_blocks, chunk, context].
        q = ops.arange(seq_len)[:, None]
        kv = ops.arange(seq_len)[None, :]
        dist = q - kv
        # HF sliding_window_mask_function((context_left - 1, context_right)):
        # left keeps 0 <= dist < max_past_horizon, right keeps -dist < context_right.
        window = ops.logical_and(dist >= 0, dist < self.max_past_horizon)
        window = ops.logical_or(
            window, ops.logical_and(dist < 0, -dist < self.max_future_horizon)
        )
        mask = ops.logical_and(window[None], valid_mask[:, None, :])
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        mask = ops.pad(mask, [[0, 0], [0, pad], [0, pad]])
        b = ops.shape(mask)[0]
        padded = num_blocks * self.chunk_size
        mask = ops.reshape(mask, (b, num_blocks, self.chunk_size, padded))
        mask = ops.pad(
            mask,
            [[0, 0], [0, 0], [0, 0], [self.max_past_horizon, self.max_future_horizon]],
        )
        starts = ops.arange(num_blocks) * self.chunk_size
        offsets = ops.arange(
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        kv_idx = starts[:, None] + offsets[None, :]
        gathered = ops.take_along_axis(mask, kv_idx[None, :, None, :], axis=3)
        return gathered[:, None]

    def call(self, input_features, input_features_mask=None):
        hidden, out_mask = self.subsample_conv_projection(
            input_features, input_features_mask
        )
        pos = self.rel_pos_enc.compute(hidden.dtype)
        seq_len = int(ops.shape(hidden)[1])
        mask = None
        if out_mask is not None:
            mask = self.blocked_mask(ops.cast(out_mask, "bool"), seq_len)
        for layer in self.alayers:
            hidden = layer(hidden, pos, mask)
        return self.output_proj(hidden), out_mask

    def compute_output_spec(self, input_features, input_features_mask=None):
        b = input_features.shape[0]
        return (
            keras.KerasTensor((b, None, self.hidden_size), dtype=self.compute_dtype),
            keras.KerasTensor((b, None), dtype="bool"),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "chunk_size": self.chunk_size})
        return config


def gemma4_scatter_soft_tokens(text_embeds, slot_mask, features):
    # Replace each True slot (row-major) with successive rows of features
    # (masked_scatter); a no-op when slot_mask is empty.
    shape = ops.shape(text_embeds)
    flat_mask = ops.reshape(slot_mask, (-1,))
    rank = ops.cumsum(ops.cast(flat_mask, "int32")) - 1
    rank = ops.clip(rank, 0, ops.shape(features)[0] - 1)
    gathered = ops.reshape(ops.take(features, rank, axis=0), shape)
    return ops.where(ops.expand_dims(slot_mask, -1), gathered, text_embeds)


def gemma4_compact_valid(features, valid_mask):
    # Gather valid (non-padding) frames to the front in row-major order.
    shape = ops.shape(features)
    flat = ops.reshape(features, (-1, shape[-1]))
    vmask = ops.reshape(valid_mask, (-1,))
    n = ops.shape(flat)[0]
    rank = ops.cumsum(ops.cast(vmask, "int32")) - 1
    target = ops.where(vmask, rank, n)
    buffer = ops.zeros((n + 1, shape[-1]), dtype=flat.dtype)
    buffer = ops.scatter_update(buffer, target[:, None], flat)
    return buffer[:n]


def gemma4_block_sequence_ids(is_vision):
    zeros = ops.zeros_like(is_vision[:, :1])
    prev = ops.concatenate([zeros, is_vision[:, :-1]], axis=1)
    new_starts = ops.logical_and(is_vision, ops.logical_not(prev))
    group = ops.cumsum(ops.cast(new_starts, "int32"), axis=1) - 1
    return ops.where(is_vision, group, ops.full_like(group, -1))


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4VisionBlockMask(layers.Layer):
    """Additive full/sliding mask with optional bidirectional attention inside
    contiguous vision-token blocks (Gemma 4). The dynamic ``arange`` is isolated
    behind ``compute_output_spec``; the block ids are derived from ``input_ids``.
    """

    def __init__(
        self,
        sliding,
        sliding_window,
        image_token_id,
        video_token_id,
        use_bidirectional_vision=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sliding = sliding
        self.sliding_window = sliding_window
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.use_bidirectional_vision = use_bidirectional_vision

    def call(self, input_ids, attention_mask=None):
        seq = ops.shape(input_ids)[1]
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        within = ki > qi - self.sliding_window
        if self.use_bidirectional_vision:
            is_vision = ops.logical_or(
                ops.equal(input_ids, self.image_token_id),
                ops.equal(input_ids, self.video_token_id),
            )
            block_ids = gemma4_block_sequence_ids(is_vision)
            same = ops.logical_and(
                block_ids[:, :, None] == block_ids[:, None, :],
                block_ids[:, :, None] >= 0,
            )
            if self.sliding:
                keep = ops.logical_and(ops.logical_or(causal, same), within)
            else:
                keep = ops.logical_or(ops.broadcast_to(causal, ops.shape(same)), same)
            mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[:, None]
        else:
            keep = ops.logical_and(causal, within) if self.sliding else causal
            mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(attention_mask, "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def compute_output_spec(self, input_ids, attention_mask=None):
        seq = input_ids.shape[1]
        return keras.KerasTensor((input_ids.shape[0], 1, seq, seq), dtype="float32")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sliding": self.sliding,
                "sliding_window": self.sliding_window,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "use_bidirectional_vision": self.use_bidirectional_vision,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4SoftTokenMerge(layers.Layer):
    """Weightless soft-token merge: optionally strip padding rows from
    ``features`` (``compact``, keeping the valid rows in row-major order), then
    scatter the surviving rows onto the True positions of ``slot_mask`` in
    ``text_embeds`` (HF ``masked_scatter``). The whole body (including deriving
    the validity mask from ``valid_source``) runs only in ``call``; the
    data-dependent gather / scatter and axis-reducing derivation are isolated
    behind ``compute_output_spec`` so they never run during functional graph
    tracing, where the soft-token count is a dynamic ``None`` dimension.

    Args:
        compact: Strip padding rows before scatter (encoder-free vision, audio).
        positions_valid: Treat ``valid_source`` as ``(b, n, 2)`` position ids
            (padding marked ``(-1, -1)``) rather than a ready boolean mask.
    """

    def __init__(self, compact, positions_valid=False, **kwargs):
        super().__init__(**kwargs)
        self.compact = compact
        self.positions_valid = positions_valid

    def call(self, text_embeds, features, slot_mask, valid_source=None):
        if self.compact:
            if self.positions_valid:
                valid = ops.logical_not(ops.all(ops.equal(valid_source, -1), axis=-1))
            else:
                valid = ops.cast(valid_source, "bool")
            feats = gemma4_compact_valid(features, valid)
        else:
            feats = ops.reshape(features, (-1, ops.shape(features)[-1]))
        return gemma4_scatter_soft_tokens(
            text_embeds, slot_mask, ops.cast(feats, text_embeds.dtype)
        )

    def compute_output_spec(self, text_embeds, features, slot_mask, valid_source=None):
        return keras.KerasTensor(text_embeds.shape, dtype=text_embeds.dtype)

    def get_config(self):
        config = super().get_config()
        config["compact"] = self.compact
        config["positions_valid"] = self.positions_valid
        return config


def gemma4_multimodal_features(
    input_ids,
    attention_mask,
    pixel_values,
    pixel_position_ids,
    input_features,
    input_features_mask,
    *,
    language_model,
    vision_model,
    embed_vision,
    vision_merge,
    audio_tower,
    embed_audio,
    audio_merge,
    full_mask_layer,
    sliding_mask_layer,
    image_token_id,
    video_token_id,
    audio_token_id,
    pad_token_id,
):
    lm = language_model
    is_image = ops.equal(input_ids, image_token_id)
    is_audio = ops.equal(input_ids, audio_token_id)
    multimodal = ops.logical_or(
        ops.logical_or(is_image, ops.equal(input_ids, video_token_id)), is_audio
    )
    hidden = lm.embed_scaled(ops.where(multimodal, pad_token_id, input_ids))
    if vision_model is not None:
        soft = embed_vision(vision_model(pixel_values, pixel_position_ids))
        hidden = vision_merge(hidden, soft, is_image)
    if audio_tower is not None:
        audio_out, out_mask = audio_tower(input_features, input_features_mask)
        audio_soft = embed_audio(audio_out)
        hidden = audio_merge(hidden, audio_soft, is_audio, out_mask)
    per_layer_inputs = (
        lm.compute_per_layer_inputs(input_ids, hidden)
        if lm.hidden_size_per_layer_input
        else None
    )
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos_l, sin_l = lm.rope_tables(position_ids, local=True)
    cos_g, sin_g = lm.rope_tables(position_ids, local=False)
    full_mask = full_mask_layer(input_ids, attention_mask)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    return lm.run_layers(
        hidden,
        cos_l,
        sin_l,
        cos_g,
        sin_g,
        full_mask,
        sliding_mask,
        per_layer_inputs=per_layer_inputs,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4MultimodalModel(BaseModel):
    """Gemma 4 vision + text backbone (no LM head).

    Composes the NaViT vision tower (:class:`Gemma4VisionModel`), the soft-token
    projector (:class:`Gemma4MultimodalEmbedder`) and the text decoder
    (:class:`Gemma4Model`). Image patches become pooled soft tokens, are
    projected into the text embedding space and scattered onto the
    ``image_token_id`` slots of the prompt. On the sliding-window layers those
    soft tokens attend bidirectionally within their image block (the ``vision``
    setting of Gemma 4's ``use_bidirectional_attention``); the global layers stay
    strictly causal. Returns raw text features; the LM head lives in
    :class:`Gemma4ConditionalGenerate`.

    Args:
        text_config: Keyword arguments forwarded to :class:`Gemma4Model`.
        vision_config: Keyword arguments forwarded to :class:`Gemma4VisionModel`.
        image_token_id: Prompt token id whose slots receive image soft tokens.
        video_token_id: Prompt token id whose slots receive video soft tokens.
        audio_token_id: Prompt token id marking audio soft-token slots.
        pad_token_id: Token id used to embed multimodal slots before scatter.
        use_bidirectional_vision: Enable blockwise bidirectional vision masking.
    """

    HF_MODEL_TYPE = ("gemma4",)
    config_class = Gemma4Config
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    output_logits = False

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=258880,
        video_token_id=258884,
        audio_token_id=258881,
        pad_token_id=0,
        use_bidirectional_vision=True,
        name=None,
        **kwargs,
    ):
        nm = kwargs.pop("name", None) or name
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        text_config = dict(text_config or {})
        vision_config = dict(vision_config) if vision_config else None
        audio_config = dict(audio_config) if audio_config else None

        language_model = Gemma4Model(**text_config, name="language_model")
        vision_model = embed_vision = audio_tower = embed_audio = None
        vision_merge = audio_merge = None
        if vision_config is not None:
            vision_model = Gemma4VisionModel(**vision_config, name="vision_tower")
            embed_vision = Gemma4MultimodalEmbedder(
                language_model.embed_dim,
                eps=language_model.norm_eps,
                name="embed_vision",
            )
            vision_merge = Gemma4SoftTokenMerge(compact=False, name="vision_merge")
        if audio_config is not None:
            audio_tower = Gemma4AudioModel(**audio_config, name="audio_tower")
            embed_audio = Gemma4MultimodalEmbedder(
                language_model.embed_dim,
                eps=language_model.norm_eps,
                name="embed_audio",
            )
            audio_merge = Gemma4SoftTokenMerge(compact=True, name="audio_merge")
        lm_head = None
        if self.output_logits and not language_model.tie_embeddings:
            lm_head = layers.Dense(
                language_model.vocab_size, use_bias=False, name="lm_head"
            )

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        inputs = {"input_ids": input_ids_in, "attention_mask": attn_in}
        has_towers = vision_model is not None or audio_tower is not None
        if has_towers:
            full_mask_layer = Gemma4VisionBlockMask(
                False,
                language_model.sliding_window,
                image_token_id,
                video_token_id,
                use_bidirectional_vision,
                name="full_mask",
            )
            sliding_mask_layer = Gemma4VisionBlockMask(
                True,
                language_model.sliding_window,
                image_token_id,
                video_token_id,
                use_bidirectional_vision,
                name="sliding_mask",
            )
            pv = pvp = feat = feat_mask = None
            if vision_model is not None:
                patch_dim = (
                    3 * vision_config["patch_size"] * vision_config["patch_size"]
                )
                pv = layers.Input(shape=(None, patch_dim), name="pixel_values")
                pvp = layers.Input(
                    shape=(None, 2), dtype="int32", name="pixel_position_ids"
                )
                inputs["pixel_values"] = pv
                inputs["pixel_position_ids"] = pvp
            if audio_tower is not None:
                audio_in_dim = audio_config.get(
                    "input_dim", audio_config.get("conv_channels", (128,))[0]
                )
                feat = layers.Input(shape=(None, audio_in_dim), name="input_features")
                feat_mask = layers.Input(
                    shape=(None,), dtype="bool", name="input_features_mask"
                )
                inputs["input_features"] = feat
                inputs["input_features_mask"] = feat_mask
            hidden = gemma4_multimodal_features(
                input_ids_in,
                attn_in,
                pv,
                pvp,
                feat,
                feat_mask,
                language_model=language_model,
                vision_model=vision_model,
                embed_vision=embed_vision,
                vision_merge=vision_merge,
                audio_tower=audio_tower,
                embed_audio=embed_audio,
                audio_merge=audio_merge,
                full_mask_layer=full_mask_layer,
                sliding_mask_layer=sliding_mask_layer,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                audio_token_id=audio_token_id,
                pad_token_id=pad_token_id,
            )
        else:
            # Text-only checkpoint: reuse the composed functional text backbone.
            hidden = language_model(
                {"input_ids": input_ids_in, "attention_mask": attn_in}
            )["last_hidden_state"]

        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            raw = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(language_model.token_embedding, name="lm_head")(hidden)
            )
            cap = language_model.final_logit_softcapping
            if cap is not None:
                raw = ops.tanh(raw / cap) * cap
            outputs["logits"] = raw

        super().__init__(inputs=inputs, outputs=outputs, name=nm or type(self).__name__)

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.pad_token_id = pad_token_id
        self.use_bidirectional_vision = use_bidirectional_vision
        self.language_model = language_model
        self.vision_model = vision_model
        self.embed_vision = embed_vision
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio
        self.lm_head = lm_head

        # The vision/audio towers' sublayers don't auto-build during the functional
        # graph construction (compute_output_spec skips their call); a concrete
        # dummy forward materializes them. Text-only reuses the already-built lm.
        if has_towers:
            with inference_scope():
                self.build_for_transfer()

    def build_for_transfer(self):
        # Materialize every weight with one minimal forward. The functional graph
        # requires all configured inputs at once (image + audio soft tokens placed
        # in their placeholder slots), so build a single dummy with matching counts.
        ids = [0]
        pv = pvp = feat = feat_mask = None
        if self.vision_model is not None:
            patch = self.vision_config.get("patch_size", 16)
            pool = self.vision_config.get("pooling_kernel_size", 3)
            num_patches = pool * pool
            n_img = num_patches // (pool * pool)  # merged image tokens
            pv = ops.zeros((1, num_patches, 3 * patch * patch), dtype="float32")
            coords = ops.stack(
                ops.meshgrid(ops.arange(pool), ops.arange(pool), indexing="xy"), axis=-1
            )
            pvp = ops.cast(ops.reshape(coords, (1, num_patches, 2)), "int32")
            ids += [self.image_token_id] * max(n_img, 1)
        if self.audio_tower is not None:
            chunk = self.audio_config.get("chunk_size", 12)
            in_dim = self.audio_config.get(
                "input_dim", self.audio_config.get("conv_channels", (128,))[0]
            )
            frames = 4 * chunk
            feat = ops.zeros((1, frames, in_dim), dtype="float32")
            feat_mask = ops.ones((1, frames), dtype="bool")
            ids += [self.audio_token_id] * chunk
        input_ids = ops.convert_to_tensor([ids], dtype="int32")
        dummy = {
            "input_ids": input_ids,
            "attention_mask": ops.ones_like(input_ids),
        }
        if pv is not None:
            dummy["pixel_values"] = pv
            dummy["pixel_position_ids"] = pvp
        if feat is not None:
            dummy["input_features"] = feat
            dummy["input_features_mask"] = feat_mask
        self(dummy)

    def scatter_soft_tokens(self, text_embeds, slot_mask, features):
        # Replace every True position of slot_mask (row-major over batch then
        # sequence) with successive rows of features, mirroring HF's
        # masked_scatter. features is [num_soft_tokens, hidden].
        shape = ops.shape(text_embeds)
        flat_mask = ops.reshape(slot_mask, (-1,))
        rank = ops.cumsum(ops.cast(flat_mask, "int32")) - 1
        rank = ops.clip(rank, 0, ops.shape(features)[0] - 1)
        gathered = ops.take(features, rank, axis=0)
        gathered = ops.reshape(gathered, shape)
        return ops.where(ops.expand_dims(slot_mask, -1), gathered, text_embeds)

    def compact_valid(self, features, valid_mask):
        # Gather the valid (non-padding) audio frames to the front in row-major
        # (batch, frame) order, mirroring HF's boolean-index padding strip.
        shape = ops.shape(features)
        flat = ops.reshape(features, (-1, shape[2]))
        vmask = ops.reshape(valid_mask, (-1,))
        n = ops.shape(flat)[0]
        rank = ops.cumsum(ops.cast(vmask, "int32")) - 1
        target = ops.where(vmask, rank, n)
        buffer = ops.zeros((n + 1, shape[2]), dtype=flat.dtype)
        buffer = ops.scatter_update(buffer, target[:, None], flat)
        return buffer[:n]

    def block_sequence_ids(self, is_vision):
        # Assign a per-image block id to contiguous vision runs; -1 for text.
        zeros = ops.zeros_like(is_vision[:, :1])
        prev = ops.concatenate([zeros, is_vision[:, :-1]], axis=1)
        new_starts = ops.logical_and(is_vision, ops.logical_not(prev))
        group = ops.cumsum(ops.cast(new_starts, "int32"), axis=1) - 1
        return ops.where(is_vision, group, ops.full_like(group, -1))

    def fuse_embeds(
        self,
        input_ids,
        pixel_values=None,
        pixel_position_ids=None,
        input_features=None,
        input_features_mask=None,
    ):
        # Embed text (multimodal slots replaced by pad), then scatter projected
        # vision and audio soft tokens into their placeholder positions. Returns
        # the fused embeddings and the vision-token mask (for blockwise masking).
        lm = self.language_model
        is_image = input_ids == self.image_token_id
        is_video = input_ids == self.video_token_id
        is_audio = input_ids == self.audio_token_id
        is_vision = ops.logical_or(is_image, is_video)
        multimodal = ops.logical_or(is_vision, is_audio)

        hidden = lm.embed_scaled(ops.where(multimodal, self.pad_token_id, input_ids))

        if pixel_values is not None and self.vision_model is not None:
            soft = self.vision_model(
                ops.convert_to_tensor(pixel_values),
                ops.cast(ops.convert_to_tensor(pixel_position_ids), "int32"),
            )
            soft = self.embed_vision(soft)
            features = ops.cast(ops.reshape(soft, (-1, lm.embed_dim)), hidden.dtype)
            hidden = self.scatter_soft_tokens(hidden, is_image, features)

        if input_features is not None and self.audio_tower is not None:
            audio_out, out_mask = self.audio_tower(
                ops.convert_to_tensor(input_features),
                None
                if input_features_mask is None
                else ops.cast(ops.convert_to_tensor(input_features_mask), "bool"),
            )
            audio_soft = self.embed_audio(audio_out)
            if out_mask is not None:
                features = self.compact_valid(audio_soft, ops.cast(out_mask, "bool"))
            else:
                features = ops.reshape(audio_soft, (-1, lm.embed_dim))
            hidden = self.scatter_soft_tokens(
                hidden, is_audio, ops.cast(features, hidden.dtype)
            )
        return hidden, is_vision

    def prefill_rope_masks(self, is_vision, attention_mask, batch, seq):
        lm = self.language_model
        position_ids = lm.compute_position_ids(attention_mask, batch, seq)
        cos_l, sin_l = lm.rope_tables(position_ids, local=True)
        cos_g, sin_g = lm.rope_tables(position_ids, local=False)
        block_ids = (
            self.block_sequence_ids(is_vision)
            if self.use_bidirectional_vision
            else None
        )
        full_mask, sliding_mask = lm.build_masks(seq, attention_mask, block_ids)
        return (cos_l, sin_l, cos_g, sin_g), (full_mask, sliding_mask)

    @staticmethod
    def vision_config_from_hf(vision):
        rope = vision.get("rope_parameters") or {}
        return {
            "hidden_size": vision["hidden_size"],
            "num_layers": vision["num_hidden_layers"],
            "num_heads": vision["num_attention_heads"],
            "num_kv_heads": vision.get(
                "num_key_value_heads", vision["num_attention_heads"]
            ),
            "head_dim": vision.get("head_dim", 64),
            "intermediate_size": vision["intermediate_size"],
            "patch_size": vision.get("patch_size", 16),
            "position_embedding_size": vision.get("position_embedding_size", 10240),
            "pooling_kernel_size": vision.get("pooling_kernel_size", 3),
            "rope_theta": rope.get("rope_theta", 100.0),
            "eps": vision.get("rms_norm_eps", 1e-6),
            "standardize": bool(vision.get("standardize", False)),
            "use_clipped_linears": bool(vision.get("use_clipped_linears", False)),
        }

    @staticmethod
    def audio_config_from_hf(audio):
        return {
            "hidden_size": audio["hidden_size"],
            "num_layers": audio["num_hidden_layers"],
            "num_heads": audio["num_attention_heads"],
            "conv_channels": tuple(audio.get("subsampling_conv_channels", (128, 32))),
            "conv_kernel_size": audio.get("conv_kernel_size", 5),
            "chunk_size": audio.get("attention_chunk_size", 12),
            "context_left": audio.get("attention_context_left", 13),
            "context_right": audio.get("attention_context_right", 0),
            "logit_cap": audio.get("attention_logit_cap", 50.0),
            "invalid_logits": audio.get("attention_invalid_logits_value", -1e9),
            "residual_weight": audio.get("residual_weight", 0.5),
            "norm_eps": audio.get("rms_norm_eps", 1e-6),
            "output_proj_dims": audio.get("output_proj_dims", 1536),
            "use_clipped_linears": bool(audio.get("use_clipped_linears", True)),
        }

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config.get("vision_config")
        audio = hf_config.get("audio_config")
        # This family owns only the NaViT / USM "gemma4" towers; the encoder-free
        # "gemma4_unified" towers live in models/gemma4_unified, so guard on the
        # tower model_type (a stray unified sub-config would load text-only here).
        vision_ok = bool(vision) and vision.get("model_type") == "gemma4_vision"
        audio_ok = bool(audio) and audio.get("model_type") == "gemma4_audio"
        return {
            "text_config": Gemma4Model.config_from_hf(hf_config),
            "vision_config": cls.vision_config_from_hf(vision) if vision_ok else None,
            "audio_config": cls.audio_config_from_hf(audio) if audio_ok else None,
            "image_token_id": hf_config.get("image_token_id", 258880),
            "video_token_id": hf_config.get("video_token_id", 258884),
            "audio_token_id": hf_config.get("audio_token_id", 258881),
            "pad_token_id": text.get("pad_token_id", 0),
            "use_bidirectional_vision": vision_ok
            and text.get("use_bidirectional_attention") == "vision",
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_hf_to_keras import transfer_gemma4_weights

        transfer_gemma4_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_config": self.text_config,
                "vision_config": self.vision_config,
                "audio_config": self.audio_config,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "audio_token_id": self.audio_token_id,
                "pad_token_id": self.pad_token_id,
                "use_bidirectional_vision": self.use_bidirectional_vision,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4ConditionalGenerate(Gemma4MultimodalModel, BaseGeneration):
    """Gemma 4 backbone + a (tied) LM head with fast ``.generate()``.

    The single multimodal generation entry point: it drives text-only checkpoints
    and the vision / audio multimodal ones through the same API. When a vision or audio
    tower is present the prefill fuses the soft tokens and applies the blockwise
    vision mask; text-only prompts (or checkpoints built without towers) skip
    straight to the text decoder. Decoding is always text-only and reuses the
    per-layer sliding / global K/V cache geometry. Pass ``pixel_values`` /
    ``pixel_position_ids`` / ``input_features`` / ``input_features_mask`` as
    keyword prefill inputs to ``generate`` when the checkpoint has the towers.
    """

    HF_MODEL_TYPE = ("gemma4", "gemma4_text")
    config_class = Gemma4Config
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    eos_token_id = (1, 106)
    output_logits = True
    # text-only checkpoints load with either head off the same weights
    HUB_REPO_SIBLINGS = frozenset({"Gemma4ConditionalGenerate", "Gemma4TextGenerate"})

    def project(self, hidden):
        lm = self.language_model
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(lm.token_embedding.embeddings))
        if lm.final_logit_softcapping is not None:
            cap = lm.final_logit_softcapping
            logits = ops.tanh(logits / cap) * cap
        return logits

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        pixel_position_ids=None,
        input_features=None,
        input_features_mask=None,
    ):
        lm = self.language_model
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        hidden, is_vision = self.fuse_embeds(
            token_ids,
            pixel_values,
            pixel_position_ids,
            input_features,
            input_features_mask,
        )
        rope, masks = self.prefill_rope_masks(
            is_vision, padding_mask, batch, prompt_len
        )
        cos_l, sin_l, cos_g, sin_g = rope
        full_mask, sliding_mask = masks
        per_layer_inputs = (
            lm.compute_per_layer_inputs(token_ids, hidden)
            if lm.hidden_size_per_layer_input
            else None
        )
        layer_caches = []
        shared_kv = {}  # layer_type -> storing layer's prompt-length (k, v)
        shared_stacked = {}  # layer_type -> storing layer's padded [ck, cv]
        for i, layer in enumerate(lm.decoder_layers):
            sliding = lm.is_sliding(i)
            layer_type = "sliding" if sliding else "global"
            cos, sin, mask = (
                (cos_l, sin_l, sliding_mask) if sliding else (cos_g, sin_g, full_mask)
            )
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            is_shared = lm.num_kv_shared_layers > 0 and i >= lm.first_kv_shared
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                attention_mask=mask,
                shared_kv=shared_kv.get(layer_type) if is_shared else None,
                per_layer_input=pli,
            )
            if is_shared:
                layer_caches.append(shared_stacked[layer_type])
                continue
            nkv = int(k.shape[1])
            hd = int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            stacked = ops.stack([ck, cv], axis=1)
            layer_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared_kv[layer_type] = (k, v)
                shared_stacked[layer_type] = stacked
        logits = self.project(lm.final_norm(hidden)[:, -1, :])
        return tuple(layer_caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        lm = self.language_model
        batch = int(token_ids.shape[0])
        max_len = int(cache[0].shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_l, sin_l = lm.rope_tables(positions, local=True)
        cos_g, sin_g = lm.rope_tables(positions, local=False)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - lm.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        token_ids = ops.cast(token_ids, "int32")
        h = lm.embed_scaled(token_ids)
        per_layer_inputs = (
            lm.compute_per_layer_inputs(token_ids, h)
            if lm.hidden_size_per_layer_input
            else None
        )
        new_caches = []
        shared_stacked = {}  # layer_type -> storing layer's updated [ck, cv]
        for i, layer in enumerate(lm.decoder_layers):
            sliding = lm.is_sliding(i)
            layer_type = "sliding" if sliding else "global"
            cos, sin, km = (
                (cos_l, sin_l, sliding_km) if sliding else (cos_g, sin_g, full_km)
            )
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            is_shared = lm.num_kv_shared_layers > 0 and i >= lm.first_kv_shared
            if is_shared:
                stacked = shared_stacked[layer_type]
                h, _, _ = layer.decode_step(
                    h,
                    cos,
                    sin,
                    stacked[:, 0],
                    stacked[:, 1],
                    pos,
                    km,
                    per_layer_input=pli,
                )
                new_caches.append(stacked)
                continue
            h, ck, cv = layer.decode_step(
                h,
                cos,
                sin,
                cache[i][:, 0],
                cache[i][:, 1],
                pos,
                km,
                per_layer_input=pli,
            )
            stacked = ops.stack([ck, cv], axis=1)
            new_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared_stacked[layer_type] = stacked
        logits = self.project(lm.final_norm(h))[:, 0, :]
        return logits, tuple(new_caches)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4TextGenerate(TextOnlyGeneration, Gemma4ConditionalGenerate):
    """Gemma 4 text-only decoder + (tied) LM head with fast ``.generate()``.

    The text-only counterpart to :class:`Gemma4ConditionalGenerate` (built with no vision
    or audio tower). All generation logic is inherited; :class:`TextOnlyGeneration` builds
    it text-only and drops the multimodal prefill inputs.

        gen = Gemma4TextGenerate.from_weights("zeromodels/gemma-4-...")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    config_class = Gemma4TextConfig
