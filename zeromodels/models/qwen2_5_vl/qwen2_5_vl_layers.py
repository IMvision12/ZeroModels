import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLRMSNorm(layers.Layer):
    """Root-mean-square layer norm (Llama / Qwen style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction and no bias. Shape-preserving ``(..., dim) -> (..., dim)``.
    Used by both the Qwen2.5 text decoder and the Qwen2.5-VL vision tower.

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


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLMLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Two parallel projections to ``mlp_dim``: a SiLU-gated ``gate`` and a linear
    ``up``: are multiplied elementwise, then projected back to ``embed_dim`` by
    ``down``. Shape-preserving on the last axis. The Qwen2.5 text decoder uses it
    bias-free; the Qwen2.5-VL vision blocks reuse it with ``use_bias=True``.

    Args:
        embed_dim: Model / residual-stream width (input and output dim).
        mlp_dim: Hidden expansion width of the ``gate`` / ``up`` projections.
        use_bias: Whether the three projections carry a bias (vision uses ``True``).
    """

    def __init__(self, embed_dim, mlp_dim, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.use_bias = use_bias
        self.gate = layers.Dense(mlp_dim, use_bias=use_bias, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=use_bias, name="up")
        self.down = layers.Dense(embed_dim, use_bias=use_bias, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLAttention(layers.Layer):
    """Grouped-query causal self-attention with multimodal rotary positions.

    The Qwen2.5 text decoder's attention: ``query`` / ``key`` / ``value`` carry a
    bias, ``output_proj`` does not. When ``num_kv_heads < num_heads`` (GQA) the K/V
    heads are repeated to match the query heads. The merged M-RoPE ``cos`` / ``sin``
    (computed by the model) are applied to Q and K, and a KV cache can be threaded
    through ``past_key_value`` for O(1)-per-token incremental decoding.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (``<= num_heads`` for GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: merged rotary tables ``(batch, q_len, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)`` (``0`` keep / large-negative block), or
            ``None``.
        past_key_value: optional ``(past_k, past_v)``, each
            ``(batch, num_kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))`` when
        ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(
            num_heads * self.head_dim, use_bias=True, name="query"
        )
        self.key = layers.Dense(num_kv_heads * self.head_dim, use_bias=True, name="key")
        self.value = layers.Dense(
            num_kv_heads * self.head_dim, use_bias=True, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def _split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        return ops.transpose(
            ops.reshape(x, (b, s, num_heads, self.head_dim)), (0, 2, 1, 3)
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
        query = self._split_heads(self.query(hidden_states), self.num_heads)
        key = self._split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self._split_heads(self.value(hidden_states), self.num_kv_heads)

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        out = fused_attention(query, key, value, self.scaling, attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_heads * self.head_dim))
        out = self.output_proj(out)
        return (out, new_key_value) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against a fixed-size KV cache written at ``write_pos``;
        # ``cos``/``sin`` are the merged M-RoPE tables. ``key_mask`` blocks empty slots.
        b = ops.shape(hidden_states)[0]
        query = self._split_heads(self.query(hidden_states), self.num_heads)
        key = self._split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self._split_heads(self.value(hidden_states), self.num_kv_heads)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )
        # rope runs in float32; match the cache dtype before writing
        key = ops.cast(key, cache_k.dtype)
        value = ops.cast(value, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), key)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), value)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(query, kk, vv, self.scaling, key_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, 1, self.num_heads * self.head_dim))
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLDecoderLayer(layers.Layer):
    """One Qwen2.5 decoder block: pre-norm GQA attention, then pre-norm SwiGLU.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + mlp(mlp_norm(h))``: RMSNorm pre-normalization with residual adds.
    The rotary tables, mask, and KV cache pass straight through to the attention.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: Epsilon shared by both RMSNorms.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as in
            :class:`Qwen2_5VLAttention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim=None,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.norm_eps = norm_eps
        self.attention_norm = Qwen2_5VLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen2_5VLAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim=self.head_dim,
            name="attention",
        )
        self.mlp_norm = Qwen2_5VLRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Qwen2_5VLMLP(embed_dim, mlp_dim, name="mlp")

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
        new_key_value = None
        if use_cache:
            attn_out, new_key_value = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5_VisionPatchEmbed(layers.Layer):
    """Patch embedding for the Qwen2.5-VL vision tower.

    HF uses a ``Conv3d`` whose kernel equals its stride, tiling each
    ``(temporal_patch_size, patch_size, patch_size)`` patch exactly once, i.e. a
    per-patch linear projection. The processor already flattens every patch to a
    ``in_channels * temporal_patch_size * patch_size**2`` vector, so this is just a
    bias-free ``Dense`` (no spatial axes, hence layout-agnostic).

    Call args:
        x: ``(num_patches, in_channels * temporal_patch_size * patch_size**2)``.

    Returns:
        ``(num_patches, embed_dim)``.
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim, use_bias=False, name="proj")

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLVisionAttention(layers.Layer):
    """Full (non-causal) vision attention with 2D rotary positions.

    Operates on the flattened patch sequence of (possibly) several images. A fused
    ``qkv`` projection and the output ``proj`` both carry a bias; 2D vision rotary
    ``cos`` / ``sin`` are applied to Q and K. The additive ``attention_mask`` makes
    attention block-diagonal: full per image, or per spatial **window** in the
    Qwen2.5-VL windowed layers, and is built from cumulative seqlens by the vision
    model.

    Call args:
        hidden_states: ``(seq, embed_dim)``.
        cos, sin: vision rotary tables ``(seq, head_dim)``.
        attention_mask: additive mask ``(1, 1, seq, seq)`` or ``None``.

    Returns:
        ``(seq, embed_dim)``.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = layers.Dense(embed_dim * 3, use_bias=True, name="qkv")
        self.proj = layers.Dense(embed_dim, use_bias=True, name="proj")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        seq = ops.shape(hidden_states)[0]
        qkv = ops.reshape(
            self.qkv(hidden_states), (seq, 3, self.num_heads, self.head_dim)
        )
        qkv = ops.transpose(qkv, (1, 0, 2, 3))
        query, key, value = qkv[0], qkv[1], qkv[2]

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )

        query = ops.expand_dims(ops.transpose(query, (1, 0, 2)), axis=0)
        key = ops.expand_dims(ops.transpose(key, (1, 0, 2)), axis=0)
        value = ops.expand_dims(ops.transpose(value, (1, 0, 2)), axis=0)

        out = fused_attention(query, key, value, self.scaling, attention_mask)
        out = ops.transpose(out[0], (1, 0, 2))
        out = ops.reshape(out, (seq, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLVisionBlock(layers.Layer):
    """Pre-norm Qwen2.5-VL vision block: ``h += attn(norm1(h)); h += mlp(norm2(h))``.

    Unlike Qwen2-VL (LayerNorm + quick-gelu MLP), Qwen2.5-VL normalizes with
    RMSNorm and uses a biased SwiGLU MLP. The attention mask selects full or
    windowed attention depending on the layer.

    Args:
        embed_dim: Vision hidden width.
        num_heads: Number of attention heads.
        intermediate_size: SwiGLU hidden width of the vision MLP.

    Call args:
        hidden_states: ``(seq, embed_dim)``.
        cos, sin: vision rotary tables ``(seq, head_dim)``.
        attention_mask: additive block-diagonal mask, or ``None``.
    """

    def __init__(self, embed_dim, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm1 = Qwen2_5VLRMSNorm(eps=1e-6, name="norm1")
        self.norm2 = Qwen2_5VLRMSNorm(eps=1e-6, name="norm2")
        self.attn = Qwen2_5VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen2_5VLMLP(embed_dim, intermediate_size, use_bias=True, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2_5VLPatchMerger(layers.Layer):
    """Merge each 2x2 (``spatial_merge_size``) patch group and project to the LLM
    hidden size.

    RMSNorm-normalizes (``ln_q``, over the vision ``context_dim``), groups every
    ``spatial_merge_size**2`` patches into one ``context_dim * merge**2`` vector,
    then a two-layer MLP with exact GELU projects to ``dim`` (the LLM hidden size).
    This is the bridge from vision tokens to language tokens.

    Args:
        dim: Output (LLM hidden) width.
        context_dim: Vision hidden width of each incoming patch.
        spatial_merge_size: Patch-merge factor per spatial axis.
    """

    def __init__(self, dim, context_dim, spatial_merge_size=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5VLRMSNorm(eps=1e-6, name="ln_q")
        self.mlp_fc1 = layers.Dense(self.hidden_size, use_bias=True, name="mlp_fc1")
        self.mlp_fc2 = layers.Dense(dim, use_bias=True, name="mlp_fc2")

    def call(self, x):
        x = ops.reshape(self.ln_q(x), (-1, self.hidden_size))
        return self.mlp_fc2(ops.gelu(self.mlp_fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config
