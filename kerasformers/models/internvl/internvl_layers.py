import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention
from kerasformers.models.qwen3_moe.qwen3_moe_layers import Qwen3MoeSparseBlock


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLRMSNorm(layers.Layer):
    """Root-mean-square layer norm (used by both the text decoder and, on the
    InternViT-6B tower, the vision blocks and QK norms).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction, no bias. Shape-preserving: ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-6``.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


def make_vision_norm(norm_type, eps, name):
    # InternViT-300M towers use LayerNorm, InternViT-6B towers use RMSNorm.
    if norm_type == "rms_norm":
        return InternVLRMSNorm(eps=eps, name=name)
    return layers.LayerNormalization(epsilon=eps, name=name)


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLVisionEmbeddings(layers.Layer):
    """InternViT input embeddings: conv patch projection + CLS token + absolute
    position embeddings.

    Pixels ``(batch, H, W, 3)`` (channels-first inputs are transposed) are
    patch-projected by a strided Conv2D, flattened to a token sequence, a
    learnable CLS token is prepended, and learnable absolute position
    embeddings are added: bicubically interpolated when the input grid
    differs from the pretrained one.

    Args:
        embed_dim: Vision hidden width.
        image_size: Pretrained square input size in pixels.
        patch_size: Patch size in pixels.
    """

    def __init__(self, embed_dim, image_size=448, patch_size=14, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_positions = (image_size // patch_size) ** 2 + 1
        self.data_format = keras.config.image_data_format()
        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=self.data_format,
            name="patch_embed",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_positions, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def interpolated_pos_embed(self, grid_h, grid_w):
        if grid_h * grid_w + 1 == self.num_positions and grid_h == grid_w:
            return self.position_embeddings
        cls_pos = self.position_embeddings[:, :1]
        patch_pos = self.position_embeddings[:, 1:]
        stored = int(round((self.num_positions - 1) ** 0.5))
        patch_pos = ops.reshape(patch_pos, (stored, stored, self.embed_dim))
        # Bicubic resample of the learned grid: Keys cubic (a = -0.75),
        # half-pixel centers, replicated borders -- exactly
        # F.interpolate(mode="bicubic", align_corners=False). Spelled out
        # because ops.image.resize is backend divergent (jax/tf use a = -0.5
        # and disagree with torch/the reference by ~0.3).
        mats = []
        for out_size, in_size in ((grid_h, stored), (grid_w, stored)):
            scale = in_size / out_size
            center = (ops.arange(out_size, dtype="float32") + 0.5) * scale - 0.5
            start = ops.floor(center)
            frac = center - start
            a = -0.75
            d0, d1, d2, d3 = frac + 1.0, frac, 1.0 - frac, 2.0 - frac
            taps = (
                ((a * d0 - 5.0 * a) * d0 + 8.0 * a) * d0 - 4.0 * a,
                ((a + 2.0) * d1 - (a + 3.0)) * d1 * d1 + 1.0,
                ((a + 2.0) * d2 - (a + 3.0)) * d2 * d2 + 1.0,
                ((a * d3 - 5.0 * a) * d3 + 8.0 * a) * d3 - 4.0 * a,
            )
            matrix = ops.zeros((out_size, in_size), dtype="float32")
            for offset, tap in zip((-1, 0, 1, 2), taps):
                index = ops.clip(ops.cast(start, "int32") + offset, 0, in_size - 1)
                # one_hot + add so clamped duplicate border taps accumulate
                onehot = ops.one_hot(index, in_size, dtype="float32")
                matrix = matrix + onehot * tap[:, None]
            mats.append(matrix)
        patch_pos = ops.einsum("hi,ijc->hjc", mats[0], ops.cast(patch_pos, "float32"))
        patch_pos = ops.einsum("wj,hjc->hwc", mats[1], patch_pos)
        patch_pos = ops.reshape(patch_pos, (1, grid_h * grid_w, self.embed_dim))
        return ops.concatenate([cls_pos, ops.cast(patch_pos, cls_pos.dtype)], axis=1)

    def call(self, pixel_values):
        b = ops.shape(pixel_values)[0]
        if self.data_format == "channels_first":
            grid_h = int(pixel_values.shape[2]) // self.patch_size
            grid_w = int(pixel_values.shape[3]) // self.patch_size
        else:
            grid_h = int(pixel_values.shape[1]) // self.patch_size
            grid_w = int(pixel_values.shape[2]) // self.patch_size
        x = self.patch_embed(pixel_values)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))  # -> (B, gh, gw, D)
        x = ops.reshape(x, (b, grid_h * grid_w, self.embed_dim))
        cls = ops.broadcast_to(
            ops.cast(self.cls_token, x.dtype), (b, 1, self.embed_dim)
        )
        x = ops.concatenate([cls, x], axis=1)
        return x + ops.cast(self.interpolated_pos_embed(grid_h, grid_w), x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLVisionAttention(layers.Layer):
    """InternViT full (bidirectional) self-attention.

    Separate ``query`` / ``key`` / ``value`` projections (with bias on the
    300M tower, bias-free on the 6B tower) and a biased ``output_proj``. On
    the 6B tower the *full-width* query / key vectors are RMS-normalized
    before the head split (``qk_norm``). No rotary: InternViT uses absolute
    position embeddings.

    Args:
        embed_dim: Vision hidden width.
        num_heads: Attention heads.
        attention_bias: Whether q/k/v carry a bias.
        qk_norm: Whether to RMS-normalize the full-width query / key.
        norm_eps: Epsilon of the QK norms.

    Call args:
        hidden_states: ``(batch, seq, embed_dim)``.

    Returns:
        ``(batch, seq, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attention_bias=True,
        qk_norm=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(embed_dim, use_bias=attention_bias, name="query")
        self.key = layers.Dense(embed_dim, use_bias=attention_bias, name="key")
        self.value = layers.Dense(embed_dim, use_bias=attention_bias, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=True, name="output_proj")
        self.query_norm = (
            InternVLRMSNorm(eps=norm_eps, name="query_norm") if qk_norm else None
        )
        self.key_norm = (
            InternVLRMSNorm(eps=norm_eps, name="key_norm") if qk_norm else None
        )

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        if self.query_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)
        q = ops.transpose(
            ops.reshape(q, (b, seq, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        k = ops.transpose(
            ops.reshape(k, (b, seq, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        v = ops.transpose(
            ops.reshape(v, (b, seq, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        out = fused_attention(q, k, v, self.scaling)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, seq, self.embed_dim))
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLVisionMLP(layers.Layer):
    """InternViT feed-forward: ``fc2(gelu(fc1(x)))`` with biased projections."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.fc1 = layers.Dense(mlp_dim, name="fc1")
        self.fc2 = layers.Dense(embed_dim, name="fc2")

    def call(self, x):
        return self.fc2(ops.gelu(self.fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLVisionLayer(layers.Layer):
    """One InternViT block: pre-norm attention and MLP, each residual branch
    scaled by a learned per-channel layer-scale vector.

    Computes ``h = x + lambda_1 * attention(layernorm_before(x))`` followed by
    ``h = h + lambda_2 * mlp(layernorm_after(h))``. The norms are LayerNorm on
    the 300M tower and RMSNorm on the 6B tower (``norm_type``).

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: MLP hidden width.
        num_heads: Attention heads.
        attention_bias: Whether q/k/v carry a bias.
        qk_norm: Whether the attention RMS-normalizes full-width q/k.
        norm_type: ``"layer_norm"`` or ``"rms_norm"``.
        norm_eps: Norm epsilon.
        layer_scale_init: Initial value of ``lambda_1`` / ``lambda_2``.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        attention_bias=True,
        qk_norm=False,
        norm_type="layer_norm",
        norm_eps=1e-6,
        layer_scale_init=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.layer_scale_init = layer_scale_init
        self.layernorm_before = make_vision_norm(
            norm_type, norm_eps, "layernorm_before"
        )
        self.layernorm_after = make_vision_norm(norm_type, norm_eps, "layernorm_after")
        self.attention = InternVLVisionAttention(
            embed_dim, num_heads, attention_bias, qk_norm, norm_eps, name="attention"
        )
        self.mlp = InternVLVisionMLP(embed_dim, mlp_dim, name="mlp")

    def build(self, input_shape):
        self.lambda_1 = self.add_weight(
            name="lambda_1",
            shape=(self.embed_dim,),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        self.lambda_2 = self.add_weight(
            name="lambda_2",
            shape=(self.embed_dim,),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        attn_out = self.attention(self.layernorm_before(hidden_states))
        hidden_states = hidden_states + self.lambda_1 * attn_out
        mlp_out = self.mlp(self.layernorm_after(hidden_states))
        return hidden_states + self.lambda_2 * mlp_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "norm_type": self.norm_type,
                "norm_eps": self.norm_eps,
                "layer_scale_init": self.layer_scale_init,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLMultiModalProjector(layers.Layer):
    """Vision-to-text projection: ``linear_2(gelu(linear_1(layer_norm(x))))``.

    Applied to the pixel-shuffled vision features (width
    ``vision_hidden * (1 / downsample_ratio)^2``), projecting them to the text
    decoder width.

    Args:
        input_dim: Pixel-shuffled feature width.
        text_dim: Text decoder hidden width.
    """

    def __init__(self, input_dim, text_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.linear_1 = layers.Dense(text_dim, name="linear_1")
        self.linear_2 = layers.Dense(text_dim, name="linear_2")

    def call(self, x):
        return self.linear_2(
            ops.gelu(self.linear_1(self.layer_norm(x)), approximate=False)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "text_dim": self.text_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLTextAttention(layers.Layer):
    """Grouped-query causal self-attention for the InternVL text decoder.

    Backbone-parametric so one layer covers every InternVL text tower: Qwen2
    (biased ``query`` / ``key`` / ``value``, no QK norm) and Qwen3 / Qwen3-MoE
    (bias-free QKV, per-head RMSNorm on the query and key before rotary).
    ``output_proj`` is always bias-free; half-rotation rotary positions are
    applied to the per-head query and key, and K/V heads are repeated for GQA. A
    KV cache can be threaded through ``past_key_value``.

    Args:
        embed_dim: Text width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        attention_bias: Whether ``query`` / ``key`` / ``value`` carry a bias
            (Qwen2: True, Qwen3 / Qwen3-MoE: False).
        qk_norm: Whether to RMS-normalize the per-head query and key before
            rotary (Qwen3 / Qwen3-MoE: True).
        norm_eps: Epsilon for the QK norms.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: rotary tables ``(batch, q_len, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_value: optional ``(past_k, past_v)``, each
            ``(batch, num_kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))``
        when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        attention_bias=True,
        qk_norm=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim, use_bias=attention_bias, name="query"
        )
        self.key = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="key"
        )
        self.value = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = (
            InternVLRMSNorm(eps=norm_eps, name="query_norm") if qk_norm else None
        )
        self.key_norm = (
            InternVLRMSNorm(eps=norm_eps, name="key_norm") if qk_norm else None
        )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        if self.query_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, 1, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        if self.query_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(q, kk, vv, self.scaling, key_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLTextMLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``, bias-free."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLDecoderLayer(layers.Layer):
    """One InternVL text block: pre-norm attention, then a pre-norm feed-forward.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + mlp(mlp_norm(h))``: RMSNorm pre-normalization with residual
    adds. The rotary tables, mask, and KV cache pass straight through to the
    attention. Backbone-parametric: the attention takes the Qwen2 vs Qwen3
    (``qk_norm`` / ``attention_bias``) flags, and the feed-forward is either the
    dense SwiGLU (:class:`InternVLTextMLP`) or, for Qwen3-MoE towers, the reused
    :class:`~kerasformers.models.qwen3_moe.qwen3_moe_layers.Qwen3MoeSparseBlock`.

    Args:
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense SwiGLU hidden width (ignored when ``use_moe``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        norm_eps: Epsilon shared by both RMSNorms (and the QK norms).
        attention_bias: Whether the attention QKV carry a bias (Qwen2: True).
        qk_norm: Whether the attention RMS-normalizes per-head q/k (Qwen3: True).
        use_moe: Swap the dense SwiGLU for a sparse Qwen3-MoE block.
        num_experts / num_experts_per_tok / moe_mlp_dim / norm_topk_prob: MoE
            routing shape, used only when ``use_moe``.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as
            in :class:`InternVLTextAttention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache`` is
        set.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps=1e-6,
        attention_bias=True,
        qk_norm=False,
        use_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        norm_topk_prob=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.attention_norm = InternVLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = InternVLTextAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_bias=attention_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            name="attention",
        )
        self.mlp_norm = InternVLRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = (
            Qwen3MoeSparseBlock(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                norm_topk_prob=norm_topk_prob,
                name="mlp",
            )
            if use_moe
            else InternVLTextMLP(embed_dim, mlp_dim, name="mlp")
        )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_out = self.attention(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "use_moe": self.use_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
            }
        )
        return config
