import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import active_attn_implementation, fused_attention
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name
from zeromodels.models.t5.t5_layers import T5LayerNorm, T5SelfAttentionLayer

NORM_EPS = 1e-6


def sincos_pos_embed_2d(embed_dim, grid_size, base_size):
    """The MMDiT 2D sinusoidal position table ``(grid_size**2, embed_dim)``
    (diffusers ``get_2d_sincos_pos_embed``): positions ``arange(grid_size) *
    base_size / grid_size``, the first half of the channels embedding the column
    and the second half the row, each as ``[sin, cos]`` over ``embed_dim // 4``
    geometric frequencies. Computed in float64 like the reference, stored float32.
    """
    positions = np.arange(grid_size, dtype=np.float64) / (grid_size / base_size)
    cols, rows = np.meshgrid(positions, positions, indexing="xy")  # (H, W) each
    half = embed_dim // 2
    omega = np.arange(half // 2, dtype=np.float64) / (half / 2.0)
    omega = 1.0 / 10000**omega

    def embed(pos):
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    table = np.concatenate([embed(cols), embed(rows)], axis=1)
    return table.astype(np.float32)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3PatchEmbed(layers.Layer):
    """MMDiT patch embedding (diffusers ``StableDiffusion3PatchEmbed`` with SD3 cropping): a
    ``patch_size`` strided convolution flattens the latent to tokens, plus the
    sinusoidal position table of a ``pos_embed_max_size`` grid cropped (centered)
    to the actual latent grid. The table is a non-trainable weight: the checkpoints
    carry their own (the released SD 3 tables are not the textbook grid, and the
    reference loads them rather than recomputing), initialized to the sinusoid.

    Args:
        embed_dim: Token width.
        patch_size: Patch (and stride) size.
        pos_embed_max_size: Side of the precomputed position grid.
        base_size: The latent side (in patches) the table is scaled to.
        module_path: Diffusers module path (``pos_embed``).
        data_format / channels_axis: Layout of the latent.
    """

    def __init__(
        self,
        embed_dim,
        patch_size,
        pos_embed_max_size,
        base_size,
        module_path="pos_embed",
        data_format=None,
        channels_axis=None,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.base_size = base_size
        self.module_path = module_path
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )
        self.proj = layers.Conv2D(
            embed_dim,
            patch_size,
            strides=patch_size,
            padding="valid",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.proj"),
        )

    def grid(self, input_shape):
        if self.data_format == "channels_first":
            return input_shape[2] // self.patch_size, input_shape[3] // self.patch_size
        return input_shape[1] // self.patch_size, input_shape[2] // self.patch_size

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.height, self.width = self.grid(input_shape)
        if max(self.height, self.width) > self.pos_embed_max_size:
            raise ValueError(
                f"A {self.height}x{self.width} latent grid exceeds the "
                f"{self.pos_embed_max_size}-patch position table."
            )
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.pos_embed_max_size**2, self.embed_dim),
            initializer=keras.initializers.Constant(
                sincos_pos_embed_2d(
                    self.embed_dim, self.pos_embed_max_size, self.base_size
                )
            ),
            trainable=False,
        )
        self.built = True

    def call(self, x):
        h = self.proj(x)
        if self.data_format == "channels_first":
            h = ops.transpose(h, (0, 2, 3, 1))
        h = ops.reshape(h, (-1, self.height * self.width, self.embed_dim))
        # the centered crop of the table (the graph is built for one grid)
        top = (self.pos_embed_max_size - self.height) // 2
        left = (self.pos_embed_max_size - self.width) // 2
        table = ops.reshape(
            self.pos_embed,
            (self.pos_embed_max_size, self.pos_embed_max_size, self.embed_dim),
        )
        table = table[top : top + self.height, left : left + self.width]
        table = ops.reshape(table, (1, self.height * self.width, self.embed_dim))
        return h + ops.cast(table, h.dtype)

    def compute_output_shape(self, input_shape):
        height, width = self.grid(input_shape)
        return (input_shape[0], height * width, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "patch_size": self.patch_size,
                "pos_embed_max_size": self.pos_embed_max_size,
                "base_size": self.base_size,
                "module_path": self.module_path,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3AdaLayerNorm(layers.Layer):
    """Adaptive LayerNorm of the MMDiT blocks: a LayerNorm without affine
    parameters, modulated by ``num_chunks`` vectors projected from the SiLU'd
    conditioning embedding (diffusers ``AdaLayerNormZero`` (6 chunks),
    ``SD35AdaLayerNormZeroX`` (9, the dual-attention blocks) and
    ``AdaLayerNormContinuous`` (2, the final norm and the last block's context)).

    ``call(x, temb)`` returns ``(x_modulated, gate_msa, shift_mlp, scale_mlp,
    gate_mlp)`` for 6 chunks (plus ``x_modulated_2, gate_msa2`` for 9), or just the
    modulated ``x`` for 2 chunks (``scale, shift`` order there).

    Args:
        dim: Token width.
        num_chunks: 2, 6 or 9.
        module_path: Diffusers module path (``transformer_blocks.0.norm1``).
    """

    def __init__(self, dim, num_chunks, module_path, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.num_chunks = num_chunks
        self.module_path = module_path
        self.linear = layers.Dense(
            num_chunks * dim, name=safe_name(f"{module_path}.linear")
        )
        self.norm = layers.LayerNormalization(
            epsilon=NORM_EPS,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.norm"),
        )

    def build(self, input_shape):
        x_shape, temb_shape = input_shape
        self.linear.build(temb_shape)
        self.norm.build(x_shape)
        self.built = True

    def call(self, inputs):
        x, temb = inputs
        emb = self.linear(ops.silu(temb))
        chunks = ops.split(emb, self.num_chunks, axis=-1)
        normed = self.norm(x)
        if self.num_chunks == 2:
            scale, shift = chunks
            return normed * (1.0 + scale[:, None, :]) + shift[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks[:6]
        modulated = normed * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        outputs = [modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        if self.num_chunks == 9:
            shift_msa2, scale_msa2, gate_msa2 = chunks[6:]
            outputs.append(
                normed * (1.0 + scale_msa2[:, None, :]) + shift_msa2[:, None, :]
            )
            outputs.append(gate_msa2)
        return outputs

    def compute_output_shape(self, input_shape):
        x_shape, temb_shape = input_shape
        if self.num_chunks == 2:
            return tuple(x_shape)
        vector = (x_shape[0], self.dim)
        outputs = [tuple(x_shape), vector, vector, vector, vector]
        if self.num_chunks == 9:
            outputs += [tuple(x_shape), vector]
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_chunks": self.num_chunks,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3JointAttention(layers.Layer):
    """MMDiT joint attention (diffusers ``Attention`` + ``JointAttnProcessor``):
    the latent tokens and the text tokens get their own q / k / v projections
    (``to_q`` / ``add_q_proj``, ...), attend jointly over the concatenated
    sequence, and are projected back separately (``to_out.0`` / ``to_add_out``).

    Args:
        dim: Token width (inner attention width).
        heads: Attention heads.
        module_path: Diffusers module path (``transformer_blocks.0.attn``).
        context: Whether the text stream takes part (the dual-attention ``attn2``
            of SD 3.5 attends the latent tokens alone).
        context_out: Whether the text stream is projected back (not in the last
            block, whose text output is unused).
        qk_norm: ``"rms_norm"`` normalizes q and k per head (SD 3.5), ``None`` not.
        attn_implementation: The :func:`fused_attention` implementation used
            when the model was built and is run without an explicit one
            (``Model.from_weights(attn_implementation=...)``). ``"fused"``: the
            joint sequence is 4429 tokens at 1024px, where the portable
            ``"sdpa"`` math materializes ``(B, heads, 4429, 4429)`` float32
            logits (3.8 GB) per block.
    """

    def __init__(
        self,
        dim,
        heads,
        module_path,
        context=True,
        context_out=True,
        qk_norm=None,
        attn_implementation="fused",
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.module_path = module_path
        self.context = context
        self.context_out = context_out
        self.qk_norm = qk_norm
        self.attn_implementation = attn_implementation
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q, self.to_k, self.to_v = (
            self.dense("to_q"),
            self.dense("to_k"),
            self.dense("to_v"),
        )
        self.to_out = self.dense("to_out.0")
        if qk_norm == "rms_norm":
            self.norm_q = self.rms_norm("norm_q")
            self.norm_k = self.rms_norm("norm_k")
        if context:
            self.add_q_proj = self.dense("add_q_proj")
            self.add_k_proj = self.dense("add_k_proj")
            self.add_v_proj = self.dense("add_v_proj")
            if qk_norm == "rms_norm":
                self.norm_added_q = self.rms_norm("norm_added_q")
                self.norm_added_k = self.rms_norm("norm_added_k")
            if context_out:
                self.to_add_out = self.dense("to_add_out")

    def dense(self, leaf):
        return layers.Dense(self.dim, name=safe_name(f"{self.module_path}.{leaf}"))

    def rms_norm(self, leaf):
        return layers.RMSNormalization(
            epsilon=NORM_EPS, name=safe_name(f"{self.module_path}.{leaf}")
        )

    def build(self, input_shape, context_shape=None):
        head_shape = (input_shape[0], self.heads, input_shape[1], self.head_dim)
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out):
            layer.build(input_shape)
        if self.qk_norm == "rms_norm":
            self.norm_q.build(head_shape)
            self.norm_k.build(head_shape)
        if self.context:
            ctx_heads = (context_shape[0], self.heads, context_shape[1], self.head_dim)
            for layer in (self.add_q_proj, self.add_k_proj, self.add_v_proj):
                layer.build(context_shape)
            if self.qk_norm == "rms_norm":
                self.norm_added_q.build(ctx_heads)
                self.norm_added_k.build(ctx_heads)
            if self.context_out:
                self.to_add_out.build(context_shape)
        self.built = True

    def split_heads(self, t):
        t = ops.reshape(t, (-1, ops.shape(t)[1], self.heads, self.head_dim))
        return ops.transpose(t, (0, 2, 1, 3))

    def call(self, x, context=None):
        q = self.split_heads(self.to_q(x))
        k = self.split_heads(self.to_k(x))
        v = self.split_heads(self.to_v(x))
        if self.qk_norm == "rms_norm":
            q, k = self.norm_q(q), self.norm_k(k)
        n_x = ops.shape(x)[1]
        if self.context:
            cq = self.split_heads(self.add_q_proj(context))
            ck = self.split_heads(self.add_k_proj(context))
            cv = self.split_heads(self.add_v_proj(context))
            if self.qk_norm == "rms_norm":
                cq, ck = self.norm_added_q(cq), self.norm_added_k(ck)
            # the latent tokens first, then the text tokens, as the reference
            q = ops.concatenate([q, cq], axis=2)
            k = ops.concatenate([k, ck], axis=2)
            v = ops.concatenate([v, cv], axis=2)
        out = fused_attention(
            q,
            k,
            v,
            self.scale,
            attn_implementation=active_attn_implementation()
            or self.attn_implementation,
        )
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, ops.shape(out)[1], self.dim))
        if not self.context:
            return self.to_out(out)
        x_out = self.to_out(out[:, :n_x])
        if not self.context_out:
            return x_out
        return x_out, self.to_add_out(out[:, n_x:])

    def compute_output_shape(self, input_shape, context_shape=None):
        x_out = tuple(input_shape[:-1]) + (self.dim,)
        if self.context and self.context_out:
            return x_out, tuple(context_shape[:-1]) + (self.dim,)
        return x_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "heads": self.heads,
                "module_path": self.module_path,
                "context": self.context,
                "context_out": self.context_out,
                "qk_norm": self.qk_norm,
                "attn_implementation": self.attn_implementation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3GELUFeedForward(layers.Layer):
    """Diffusers ``FeedForward(activation_fn="gelu-approximate")``: ``net.0.proj``
    to ``mult * dim``, tanh GELU, ``net.2`` back to ``dim``.

    Args:
        dim: Token width.
        module_path: Diffusers module path (``transformer_blocks.0.ff``).
        mult: Inner width multiplier (4).
    """

    def __init__(self, dim, module_path, mult=4, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.module_path = module_path
        self.mult = mult
        self.proj = layers.Dense(
            dim * mult, name=safe_name(f"{module_path}.net.0.proj")
        )
        self.out = layers.Dense(dim, name=safe_name(f"{module_path}.net.2"))

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.out.build(tuple(input_shape[:-1]) + (self.dim * self.mult,))
        self.built = True

    def call(self, x):
        return self.out(ops.gelu(self.proj(x), approximate=True))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"dim": self.dim, "module_path": self.module_path, "mult": self.mult}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3JointTransformerBlock(layers.Layer):
    """One MMDiT block (diffusers ``StableDiffusion3JointTransformerBlock``): the latent and text
    streams are each AdaLN-modulated by the conditioning embedding, attend jointly,
    and pass through their own gated feed-forwards. ``call([x, context, temb])``
    returns ``[x, context]``, or ``x`` alone for the last block
    (``context_pre_only``: its text stream is not updated). SD 3.5's dual-attention
    blocks add a second, latent-only attention (``attn2``) with its own modulation.

    Args:
        dim: Token width.
        heads: Attention heads.
        module_path: Diffusers module path (``transformer_blocks.0``).
        context_pre_only: Last-block variant (2-chunk context norm, no text output).
        qk_norm: ``"rms_norm"`` or ``None``.
        use_dual_attention: The SD 3.5 dual-attention variant.
    """

    def __init__(
        self,
        dim,
        heads,
        module_path,
        context_pre_only=False,
        qk_norm=None,
        use_dual_attention=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.module_path = module_path
        self.context_pre_only = context_pre_only
        self.qk_norm = qk_norm
        self.use_dual_attention = use_dual_attention
        self.norm1 = StableDiffusion3AdaLayerNorm(
            dim, 9 if use_dual_attention else 6, module_path=f"{module_path}.norm1"
        )
        self.norm1_context = StableDiffusion3AdaLayerNorm(
            dim,
            2 if context_pre_only else 6,
            module_path=f"{module_path}.norm1_context",
        )
        self.attn = StableDiffusion3JointAttention(
            dim,
            heads,
            module_path=f"{module_path}.attn",
            context=True,
            context_out=not context_pre_only,
            qk_norm=qk_norm,
        )
        if use_dual_attention:
            self.attn2 = StableDiffusion3JointAttention(
                dim,
                heads,
                module_path=f"{module_path}.attn2",
                context=False,
                qk_norm=qk_norm,
            )
        self.norm2 = layers.LayerNormalization(
            epsilon=NORM_EPS,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.norm2"),
        )
        self.ff = StableDiffusion3GELUFeedForward(dim, module_path=f"{module_path}.ff")
        if not context_pre_only:
            self.norm2_context = layers.LayerNormalization(
                epsilon=NORM_EPS,
                center=False,
                scale=False,
                name=safe_name(f"{module_path}.norm2_context"),
            )
            self.ff_context = StableDiffusion3GELUFeedForward(
                dim, module_path=f"{module_path}.ff_context"
            )

    def build(self, input_shape):
        x_shape, context_shape, temb_shape = input_shape
        self.norm1.build((x_shape, temb_shape))
        self.norm1_context.build((context_shape, temb_shape))
        self.attn.build(x_shape, context_shape)
        if self.use_dual_attention:
            self.attn2.build(x_shape)
        self.norm2.build(x_shape)
        self.ff.build(x_shape)
        if not self.context_pre_only:
            self.norm2_context.build(context_shape)
            self.ff_context.build(context_shape)
        self.built = True

    def call(self, inputs):
        x, context, temb = inputs
        modulated = self.norm1([x, temb])
        normed_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulated[:5]
        if self.context_pre_only:
            normed_context = self.norm1_context([context, temb])
            attn_x = self.attn(normed_x, normed_context)
        else:
            normed_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                self.norm1_context([context, temb])
            )
            attn_x, attn_context = self.attn(normed_x, normed_context)
        x = x + gate_msa[:, None, :] * attn_x
        if self.use_dual_attention:
            normed_x2, gate_msa2 = modulated[5:]
            x = x + gate_msa2[:, None, :] * self.attn2(normed_x2)
        h = self.norm2(x) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.ff(h)
        if self.context_pre_only:
            return x
        context = context + c_gate_msa[:, None, :] * attn_context
        h = self.norm2_context(context) * (1.0 + c_scale_mlp[:, None, :])
        h = h + c_shift_mlp[:, None, :]
        context = context + c_gate_mlp[:, None, :] * self.ff_context(h)
        return [x, context]

    def compute_output_shape(self, input_shape):
        x_shape, context_shape, _ = input_shape
        if self.context_pre_only:
            return tuple(x_shape)
        return [tuple(x_shape), tuple(context_shape)]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "heads": self.heads,
                "module_path": self.module_path,
                "context_pre_only": self.context_pre_only,
                "qk_norm": self.qk_norm,
                "use_dual_attention": self.use_dual_attention,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3T5GatedFeedForward(layers.Layer):
    """The T5 v1.1 feed-forward (``T5DenseGatedActDense``) with its pre-norm and
    residual: ``x + wo(gelu_tanh(wi_0(norm(x))) * wi_1(norm(x)))``, bias-free
    projections and the T5 RMSNorm. The original T5's ``relu(wi(x))`` block lives
    in the t5 family; this gated form is what the SD 3 T5-XXL encoder uses. The
    sub-layers are named ``{prefix}_ln`` / ``{prefix}_wi_0`` / ``{prefix}_wi_1`` /
    ``{prefix}_wo`` so every weight path is unique across blocks.

    Args:
        embed_dim: Token width.
        mlp_dim: Inner width.
        eps: RMSNorm epsilon.
        prefix: Leaf-name prefix (``enc_3_ff``).
    """

    def __init__(self, embed_dim, mlp_dim, eps, prefix, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.eps = eps
        self.prefix = prefix
        self.layer_norm = T5LayerNorm(eps, name=f"{prefix}_ln")
        self.wi_0 = layers.Dense(mlp_dim, use_bias=False, name=f"{prefix}_wi_0")
        self.wi_1 = layers.Dense(mlp_dim, use_bias=False, name=f"{prefix}_wi_1")
        self.wo = layers.Dense(embed_dim, use_bias=False, name=f"{prefix}_wo")

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.wi_0.build(input_shape)
        self.wi_1.build(input_shape)
        self.wo.build(tuple(input_shape[:-1]) + (self.mlp_dim,))
        self.built = True

    def call(self, hidden_states):
        normed = self.layer_norm(hidden_states)
        # "gelu_new": the tanh-approximate GELU of the T5 v1.1 checkpoints
        inner = ops.gelu(self.wi_0(normed), approximate=True) * self.wi_1(normed)
        return hidden_states + self.wo(inner)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "eps": self.eps,
                "prefix": self.prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3T5GatedEncoderBlock(layers.Layer):
    """One T5 v1.1 encoder block: the t5 family's pre-norm self-attention with the
    shared relative position bias, then :class:`StableDiffusion3T5GatedFeedForward`.

    Args:
        embed_dim: Token width.
        key_value_dim: Width of one attention head.
        num_heads: Attention heads.
        mlp_dim: Feed-forward inner width.
        eps: RMSNorm epsilon.
        prefix: Leaf-name prefix (``enc_3``).
    """

    def __init__(
        self, embed_dim, key_value_dim, num_heads, mlp_dim, eps, prefix, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.eps = eps
        self.prefix = prefix
        self.self_attention = T5SelfAttentionLayer(
            embed_dim, key_value_dim, num_heads, eps, prefix=f"{prefix}_attn"
        )
        self.ff = StableDiffusion3T5GatedFeedForward(
            embed_dim, mlp_dim, eps, prefix=f"{prefix}_ff"
        )

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.ff.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_bias):
        hidden_states = self.self_attention(hidden_states, position_bias)
        return self.ff(hidden_states)

    def compute_output_spec(self, hidden_states, position_bias):
        return keras.KerasTensor(hidden_states.shape, dtype=hidden_states.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "key_value_dim": self.key_value_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "eps": self.eps,
                "prefix": self.prefix,
            }
        )
        return config
