"""Keras 3 layers for Qwen-Image's double-stream DiT.

Ports the building blocks of Diffusers ``transformer_qwenimage.py`` (RMSNorm,
timestep projection, 3D RoPE, joint text/image attention, and the dual-stream
transformer block). The full ``QwenImageTransformer2DModel`` lives elsewhere.

Module paths follow Diffusers so weight conversion can map leaves such as
``transformer_blocks.0.attn.to_q``, ``img_mod.1``, ``img_mlp.net.0.proj``, and
``time_text_embed.timestep_embedder.linear_1``.
"""

from __future__ import annotations

import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import active_attn_implementation, fused_attention
from zeromodels.models.stable_diffusion.stable_diffusion_layers import (
    safe_name,
    timestep_embedding,
)
from zeromodels.models.stable_diffusion_3.stable_diffusion_3_layers import (
    StableDiffusion3AdaLayerNorm,
    StableDiffusion3GELUFeedForward,
)

NORM_EPS = 1e-6
MASK_NEG = -1e4
ROPE_MAX_INDEX = 4096


def qwen_approximate_gelu(x):
    """Tanh-approximate GELU (Diffusers ``gelu-approximate`` / BERT).

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``. Prefer
    :class:`StableDiffusion3GELUFeedForward` (``ops.gelu(..., approximate=True)``)
    for the block MLPs; this helper is for call sites that need the formula
    explicitly.
    """
    return 0.5 * x * (1.0 + ops.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * ops.power(x, 3))))


def apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    """Apply Qwen RoPE to ``x`` of shape ``[B, S, H, D]``.

    ``freqs_cis`` is either rotation angles ``[S, D // 2]`` (preferred; matches
    the Diffusers Neuron path) or a ``(cos, sin)`` pair each ``[S, D]`` with
    adjacent-pair angles repeated. Equivalent to complex multiplication of
    adjacent feature pairs ``(x[..., 2k], x[..., 2k+1])``.

    Args:
        x: Query or key tensor ``[B, S, H, D]``.
        freqs_cis: Angles ``[S, D // 2]`` or ``(cos, sin)`` each ``[S, D]``.
        use_real: Kept for Diffusers API parity; only the real path is used.
        use_real_unbind_dim: ``-1`` (adjacent pairs, Qwen/Flux) or ``-2``.
    """
    del use_real  # real path only (portable across Keras backends)
    if isinstance(freqs_cis, (tuple, list)):
        cos, sin = freqs_cis
    else:
        cos = ops.repeat(ops.cos(freqs_cis), 2, axis=-1)
        sin = ops.repeat(ops.sin(freqs_cis), 2, axis=-1)
    # Broadcast over batch and heads: [S, D] -> [1, S, 1, D]
    cos = ops.expand_dims(ops.expand_dims(cos, 0), 2)
    sin = ops.expand_dims(ops.expand_dims(sin, 0), 2)

    x_dtype = x.dtype
    x_f = ops.cast(x, "float32")
    if use_real_unbind_dim == -1:
        pair = ops.reshape(x_f, ops.shape(x)[:-1] + (-1, 2))
        x_real, x_imag = pair[..., 0], pair[..., 1]
        x_rotated = ops.reshape(
            ops.stack([-x_imag, x_real], axis=-1), ops.shape(x)
        )
    elif use_real_unbind_dim == -2:
        pair = ops.reshape(x_f, ops.shape(x)[:-1] + (2, -1))
        x_real, x_imag = pair[..., 0, :], pair[..., 1, :]
        x_rotated = ops.concatenate([-x_imag, x_real], axis=-1)
    else:
        raise ValueError(
            f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
        )
    out = x_f * ops.cast(cos, "float32") + x_rotated * ops.cast(sin, "float32")
    return ops.cast(out, x_dtype)


def _rope_angles(index, dim, theta):
    """Outer product of positions with ``1 / theta^(i/dim)`` frequencies."""
    assert dim % 2 == 0
    freqs = np.outer(
        np.asarray(index, dtype=np.float64),
        1.0
        / np.power(
            float(theta),
            np.arange(0, dim, 2, dtype=np.float64) / dim,
        ),
    )
    return freqs.astype(np.float32)


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageRMSNorm(layers.Layer):
    """Diffusers ``RMSNorm`` (learned weight, ones init, ``eps=1e-6``).

    Normalizes over the last axis. Used for ``txt_norm`` and per-head Q/K norms
    (``attn.norm_q``, ``norm_k``, ``norm_added_q``, ``norm_added_k``).
    """

    def __init__(self, eps=NORM_EPS, module_path=None, **kwargs):
        if module_path is not None:
            kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.eps = eps
        self.module_path = module_path

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(int(input_shape[-1]),),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x32 = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x32), axis=-1, keepdims=True)
        x32 = x32 * ops.rsqrt(variance + self.eps)
        return ops.cast(x32, dtype) * self.weight

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "module_path": self.module_path})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTimestepProjEmbeddings(layers.Layer):
    """Diffusers ``QwenTimestepProjEmbeddings``.

    ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)``
    followed by ``TimestepEmbedding`` (``linear_1`` / SiLU / ``linear_2``) to
    ``embedding_dim``. The Diffusers ``scale=1000`` is applied by multiplying
    timesteps before :func:`timestep_embedding` (equivalent to scaling the
    sinusoidal arguments).

    Weight path: ``{module_path}.timestep_embedder.linear_{1,2}``.
    """

    def __init__(
        self,
        embedding_dim,
        module_path="time_text_embed",
        time_freq_dim=256,
        scale=1000.0,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.module_path = module_path
        self.time_freq_dim = time_freq_dim
        self.scale = scale
        embedder = f"{module_path}.timestep_embedder"
        self.linear_1 = layers.Dense(
            embedding_dim, name=safe_name(f"{embedder}.linear_1")
        )
        self.linear_2 = layers.Dense(
            embedding_dim, name=safe_name(f"{embedder}.linear_2")
        )

    def build(self, timestep_shape):
        # Sinusoidal projection is fixed-width; MLP builds from (B, time_freq_dim).
        freq_shape = (timestep_shape[0], self.time_freq_dim)
        self.linear_1.build(freq_shape)
        self.linear_2.build((timestep_shape[0], self.embedding_dim))
        self.built = True

    def call(self, timestep):
        # Diffusers Timesteps(scale=1000): emb = scale * (t[:, None] * freqs)
        t = ops.cast(timestep, "float32") * self.scale
        emb = timestep_embedding(
            t,
            self.time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        emb = self.linear_1(emb)
        emb = ops.silu(emb)
        return self.linear_2(emb)

    def compute_output_shape(self, timestep_shape):
        return (timestep_shape[0], self.embedding_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "module_path": self.module_path,
                "time_freq_dim": self.time_freq_dim,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageEmbedRope(layers.Layer):
    """Diffusers ``QwenEmbedRope`` with real-valued rotation angles.

    Precomputes positive/negative position frequencies as angles (not complex)
    for backend portability. ``call(img_h, img_w, txt_seq_len, frame=1)`` returns
    ``(vid_freqs, txt_freqs)`` each ``[S, sum(axes_dim) // 2]`` for
    :func:`apply_rotary_emb_qwen`.

    Defaults match Qwen-Image T2I: ``theta=10000``, ``axes_dim=(16, 56, 56)``,
    ``scale_rope=True``.
    """

    def __init__(
        self,
        theta=10000,
        axes_dim=(16, 56, 56),
        scale_rope=True,
        module_path="pos_embed",
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.theta = int(theta)
        self.axes_dim = tuple(int(d) for d in axes_dim)
        self.scale_rope = bool(scale_rope)
        self.module_path = module_path
        self.rope_dim = sum(self.axes_dim) // 2  # complex half-dim == angle width
        self._axis_halves = [d // 2 for d in self.axes_dim]

        pos_index = np.arange(ROPE_MAX_INDEX)
        # Diffusers: arange(4096).flip(0) * -1 - 1 -> [-4096, ..., -1]
        neg_index = np.flip(pos_index) * -1 - 1
        pos_freqs = np.concatenate(
            [_rope_angles(pos_index, d, self.theta) for d in self.axes_dim],
            axis=1,
        )
        neg_freqs = np.concatenate(
            [_rope_angles(neg_index, d, self.theta) for d in self.axes_dim],
            axis=1,
        )
        self._pos_freqs_np = pos_freqs
        self._neg_freqs_np = neg_freqs

    def build(self, input_shape=None):
        self.pos_freqs = self.add_weight(
            name="pos_freqs",
            shape=(ROPE_MAX_INDEX, self.rope_dim),
            initializer=keras.initializers.Constant(self._pos_freqs_np),
            trainable=False,
        )
        self.neg_freqs = self.add_weight(
            name="neg_freqs",
            shape=(ROPE_MAX_INDEX, self.rope_dim),
            initializer=keras.initializers.Constant(self._neg_freqs_np),
            trainable=False,
        )
        self.built = True

    def _split_axes(self, freqs):
        # Diffusers ``tensor.split(sizes, dim)``; Keras ``ops.split`` takes indices.
        h0, h1, h2 = self._axis_halves
        return (
            freqs[:, :h0],
            freqs[:, h0 : h0 + h1],
            freqs[:, h0 + h1 :],
        )

    def _video_freqs(self, frame, height, width, idx=0):
        frame, height, width, idx = int(frame), int(height), int(width), int(idx)
        seq_lens = frame * height * width
        freqs_pos = self._split_axes(self.pos_freqs)
        freqs_neg = self._split_axes(self.neg_freqs)

        freqs_frame = freqs_pos[0][idx : idx + frame]
        freqs_frame = ops.reshape(freqs_frame, (frame, 1, 1, self._axis_halves[0]))
        freqs_frame = ops.broadcast_to(
            freqs_frame, (frame, height, width, self._axis_halves[0])
        )

        if self.scale_rope:
            h_neg = height - height // 2
            w_neg = width - width // 2
            freqs_height = ops.concatenate(
                [freqs_neg[1][-h_neg:], freqs_pos[1][: height // 2]], axis=0
            )
            freqs_width = ops.concatenate(
                [freqs_neg[2][-w_neg:], freqs_pos[2][: width // 2]], axis=0
            )
        else:
            freqs_height = freqs_pos[1][:height]
            freqs_width = freqs_pos[2][:width]

        freqs_height = ops.reshape(
            freqs_height, (1, height, 1, self._axis_halves[1])
        )
        freqs_height = ops.broadcast_to(
            freqs_height, (frame, height, width, self._axis_halves[1])
        )
        freqs_width = ops.reshape(freqs_width, (1, 1, width, self._axis_halves[2]))
        freqs_width = ops.broadcast_to(
            freqs_width, (frame, height, width, self._axis_halves[2])
        )

        freqs = ops.concatenate(
            [freqs_frame, freqs_height, freqs_width], axis=-1
        )
        return ops.reshape(freqs, (seq_lens, self.rope_dim))

    def call(self, img_h, img_w, txt_seq_len, frame=1):
        """Return ``(vid_freqs, txt_freqs)`` angle tables for T2I RoPE.

        Args:
            img_h / img_w: Latent patch grid height / width (Python ints preferred).
            txt_seq_len: Text token count (matches encoder sequence length).
            frame: Temporal size (``1`` for images).
        """
        if not self.built:
            self.build(None)
        height, width = int(img_h), int(img_w)
        txt_seq_len = int(txt_seq_len)
        frame = int(frame)

        vid_freqs = self._video_freqs(frame, height, width, idx=0)
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + txt_seq_len]
        return vid_freqs, txt_freqs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "theta": self.theta,
                "axes_dim": list(self.axes_dim),
                "scale_rope": self.scale_rope,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageDoubleStreamAttention(layers.Layer):
    """Joint text/image attention (Diffusers ``Attention`` + ``QwenDoubleStreamAttnProcessor2_0``).

    Separate QKV for the image stream (``to_q`` / ``to_k`` / ``to_v``) and text
    stream (``add_q_proj`` / ``add_k_proj`` / ``add_v_proj``), RMSNorm on Q/K for
    both, RoPE on both, then joint attention over ``[text, image]`` (text-first,
    unlike SD3's image-first concat). Outputs project via ``to_out.0`` and
    ``to_add_out``.

    Typical Qwen-Image sizes: ``dim=3072``, ``heads=24``, ``head_dim=128``.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim=None,
        module_path="attn",
        eps=NORM_EPS,
        attn_implementation="fused",
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim or (dim // num_attention_heads)
        self.module_path = module_path
        self.eps = eps
        self.attn_implementation = attn_implementation
        self.scale = self.attention_head_dim**-0.5

        self.to_q = self._dense("to_q")
        self.to_k = self._dense("to_k")
        self.to_v = self._dense("to_v")
        self.to_out = self._dense("to_out.0")
        self.add_q_proj = self._dense("add_q_proj")
        self.add_k_proj = self._dense("add_k_proj")
        self.add_v_proj = self._dense("add_v_proj")
        self.to_add_out = self._dense("to_add_out")

        self.norm_q = QwenImageRMSNorm(eps, module_path=f"{module_path}.norm_q")
        self.norm_k = QwenImageRMSNorm(eps, module_path=f"{module_path}.norm_k")
        self.norm_added_q = QwenImageRMSNorm(
            eps, module_path=f"{module_path}.norm_added_q"
        )
        self.norm_added_k = QwenImageRMSNorm(
            eps, module_path=f"{module_path}.norm_added_k"
        )

    def _dense(self, leaf):
        return layers.Dense(self.dim, name=safe_name(f"{self.module_path}.{leaf}"))

    def build(
        self,
        hidden_states_shape,
        encoder_hidden_states_shape=None,
        image_rotary_emb_shape=None,
        encoder_hidden_states_mask_shape=None,
    ):
        del image_rotary_emb_shape, encoder_hidden_states_mask_shape
        if encoder_hidden_states_shape is None:
            encoder_hidden_states_shape = hidden_states_shape
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out):
            layer.build(hidden_states_shape)
        for layer in (
            self.add_q_proj,
            self.add_k_proj,
            self.add_v_proj,
            self.to_add_out,
        ):
            layer.build(encoder_hidden_states_shape)
        head_shape = (
            None,
            None,
            self.num_attention_heads,
            self.attention_head_dim,
        )
        for norm in (self.norm_q, self.norm_k, self.norm_added_q, self.norm_added_k):
            norm.build(head_shape)
        self.built = True

    def _to_heads(self, t):
        # [B, S, dim] -> [B, S, H, D]
        t = ops.reshape(
            t,
            (-1, ops.shape(t)[1], self.num_attention_heads, self.attention_head_dim),
        )
        return t

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb=None,
        encoder_hidden_states_mask=None,
    ):
        seq_txt = ops.shape(encoder_hidden_states)[1]

        img_q = self._to_heads(self.to_q(hidden_states))
        img_k = self._to_heads(self.to_k(hidden_states))
        img_v = self._to_heads(self.to_v(hidden_states))
        txt_q = self._to_heads(self.add_q_proj(encoder_hidden_states))
        txt_k = self._to_heads(self.add_k_proj(encoder_hidden_states))
        txt_v = self._to_heads(self.add_v_proj(encoder_hidden_states))

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        # Joint sequence order: [text, image] (Diffusers Qwen; not SD3).
        query = ops.concatenate([txt_q, img_q], axis=1)
        key = ops.concatenate([txt_k, img_k], axis=1)
        value = ops.concatenate([txt_v, img_v], axis=1)

        # fused_attention expects [B, H, S, D]
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        attention_mask = None
        if encoder_hidden_states_mask is not None:
            batch = ops.shape(hidden_states)[0]
            seq_img = ops.shape(hidden_states)[1]
            img_mask = ops.ones((batch, seq_img), dtype=encoder_hidden_states_mask.dtype)
            joint_mask = ops.concatenate(
                [encoder_hidden_states_mask, img_mask], axis=1
            )
            # Additive mask: keep=0, drop=MASK_NEG (Diffusers bool True=keep).
            keep = ops.cast(joint_mask, "float32")
            attention_mask = (1.0 - keep) * MASK_NEG
            attention_mask = attention_mask[:, None, None, :]

        out = fused_attention(
            query,
            key,
            value,
            self.scale,
            attention_mask=attention_mask,
            attn_implementation=active_attn_implementation()
            or self.attn_implementation,
        )
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, ops.shape(out)[1], self.dim))

        txt_out = self.to_add_out(out[:, :seq_txt])
        img_out = self.to_out(out[:, seq_txt:])
        return img_out, txt_out

    def compute_output_shape(self, input_shape, context_shape=None):
        if context_shape is None and isinstance(input_shape, (list, tuple)):
            if len(input_shape) >= 2:
                input_shape, context_shape = input_shape[0], input_shape[1]
        img_out = tuple(input_shape[:-1]) + (self.dim,)
        txt_out = tuple(context_shape[:-1]) + (self.dim,)
        return img_out, txt_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_attention_heads": self.num_attention_heads,
                "attention_head_dim": self.attention_head_dim,
                "module_path": self.module_path,
                "eps": self.eps,
                "attn_implementation": self.attn_implementation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTransformerBlock(layers.Layer):
    """One double-stream DiT block (Diffusers ``QwenImageTransformerBlock``).

    Image and text streams each get SiLU+Dense(``6 * dim``) modulation
    (``img_mod.1`` / ``txt_mod.1``), affine-free LayerNorm, joint attention,
    and a gated ``gelu-approximate`` MLP. ``zero_cond_t`` is not supported in
    this v1 port (always ``False``).

    ``call`` returns ``(encoder_hidden_states, hidden_states)`` — text then
    image — matching Diffusers.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim=None,
        module_path="transformer_blocks.0",
        eps=NORM_EPS,
        attn_implementation="fused",
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim or (dim // num_attention_heads)
        self.module_path = module_path
        self.eps = eps
        self.attn_implementation = attn_implementation

        # Diffusers Sequential(SiLU, Linear) -> weight leaf ``img_mod.1`` / ``txt_mod.1``.
        self.img_mod = layers.Dense(
            6 * dim, name=safe_name(f"{module_path}.img_mod.1")
        )
        self.txt_mod = layers.Dense(
            6 * dim, name=safe_name(f"{module_path}.txt_mod.1")
        )
        self.img_norm1 = layers.LayerNormalization(
            epsilon=eps,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.img_norm1"),
        )
        self.txt_norm1 = layers.LayerNormalization(
            epsilon=eps,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.txt_norm1"),
        )
        self.img_norm2 = layers.LayerNormalization(
            epsilon=eps,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.img_norm2"),
        )
        self.txt_norm2 = layers.LayerNormalization(
            epsilon=eps,
            center=False,
            scale=False,
            name=safe_name(f"{module_path}.txt_norm2"),
        )
        self.attn = QwenImageDoubleStreamAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            module_path=f"{module_path}.attn",
            eps=eps,
            attn_implementation=attn_implementation,
        )
        self.img_mlp = StableDiffusion3GELUFeedForward(
            dim, module_path=f"{module_path}.img_mlp"
        )
        self.txt_mlp = StableDiffusion3GELUFeedForward(
            dim, module_path=f"{module_path}.txt_mlp"
        )

    def build(
        self,
        hidden_states_shape,
        encoder_hidden_states_shape=None,
        temb_shape=None,
        encoder_hidden_states_mask_shape=None,
        image_rotary_emb_shape=None,
    ):
        del encoder_hidden_states_mask_shape, image_rotary_emb_shape
        if isinstance(hidden_states_shape, (list, tuple)) and temb_shape is None:
            # Explicit ``build((img, txt, temb))`` from a parent layer.
            if (
                len(hidden_states_shape) >= 3
                and hasattr(hidden_states_shape[0], "__len__")
            ):
                (
                    hidden_states_shape,
                    encoder_hidden_states_shape,
                    temb_shape,
                ) = (
                    hidden_states_shape[0],
                    hidden_states_shape[1],
                    hidden_states_shape[2],
                )
        if encoder_hidden_states_shape is None or temb_shape is None:
            raise ValueError(
                "QwenImageTransformerBlock.build expects image, text, and temb shapes."
            )
        self.img_mod.build(temb_shape)
        self.txt_mod.build(temb_shape)
        self.img_norm1.build(hidden_states_shape)
        self.txt_norm1.build(encoder_hidden_states_shape)
        self.img_norm2.build(hidden_states_shape)
        self.txt_norm2.build(encoder_hidden_states_shape)
        self.attn.build(hidden_states_shape, encoder_hidden_states_shape)
        self.img_mlp.build(hidden_states_shape)
        self.txt_mlp.build(encoder_hidden_states_shape)
        self.built = True

    def _modulate(self, x, mod_params):
        """``x * (1 + scale) + shift``, returning ``(modulated, gate)``."""
        shift, scale, gate = ops.split(mod_params, 3, axis=-1)
        modulated = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return modulated, gate

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        encoder_hidden_states_mask=None,
        image_rotary_emb=None,
    ):
        img_mod_params = self.img_mod(ops.silu(temb))
        txt_mod_params = self.txt_mod(ops.silu(temb))
        img_mod1, img_mod2 = ops.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = ops.split(txt_mod_params, 2, axis=-1)

        img_modulated, img_gate1 = self._modulate(
            self.img_norm1(hidden_states), img_mod1
        )
        txt_modulated, txt_gate1 = self._modulate(
            self.txt_norm1(encoder_hidden_states), txt_mod1
        )

        img_attn, txt_attn = self.attn(
            img_modulated,
            txt_modulated,
            image_rotary_emb=image_rotary_emb,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        hidden_states = hidden_states + img_gate1[:, None, :] * img_attn
        encoder_hidden_states = (
            encoder_hidden_states + txt_gate1[:, None, :] * txt_attn
        )

        img_modulated2, img_gate2 = self._modulate(
            self.img_norm2(hidden_states), img_mod2
        )
        hidden_states = hidden_states + img_gate2[:, None, :] * self.img_mlp(
            img_modulated2
        )

        txt_modulated2, txt_gate2 = self._modulate(
            self.txt_norm2(encoder_hidden_states), txt_mod2
        )
        encoder_hidden_states = (
            encoder_hidden_states
            + txt_gate2[:, None, :] * self.txt_mlp(txt_modulated2)
        )

        # Diffusers clips fp16 dual-stream outputs to the fp16 finite range.
        if str(encoder_hidden_states.dtype).endswith("float16"):
            encoder_hidden_states = ops.clip(
                encoder_hidden_states, -65504.0, 65504.0
            )
        if str(hidden_states.dtype).endswith("float16"):
            hidden_states = ops.clip(hidden_states, -65504.0, 65504.0)

        return encoder_hidden_states, hidden_states

    def compute_output_shape(self, input_shape):
        # Multi-input builds pass ``[img_shape, txt_shape, temb_shape]``; a lone
        # tensor shape is ``(batch, seq, dim)``.
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) >= 2
            and hasattr(input_shape[0], "__len__")
        ):
            img_shape, txt_shape = input_shape[0], input_shape[1]
            return tuple(txt_shape), tuple(img_shape)
        return tuple(input_shape), tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_attention_heads": self.num_attention_heads,
                "attention_head_dim": self.attention_head_dim,
                "module_path": self.module_path,
                "eps": self.eps,
                "attn_implementation": self.attn_implementation,
            }
        )
        return config


# Final-norm AdaLN: Diffusers ``AdaLayerNormContinuous`` uses (scale, shift)
# order with ``num_chunks=2`` — same as StableDiffusion3AdaLayerNorm.
QwenImageAdaLayerNormContinuous = StableDiffusion3AdaLayerNorm


__all__ = [
    "qwen_approximate_gelu",
    "apply_rotary_emb_qwen",
    "QwenImageRMSNorm",
    "QwenImageTimestepProjEmbeddings",
    "QwenImageEmbedRope",
    "QwenImageDoubleStreamAttention",
    "QwenImageTransformerBlock",
    "QwenImageAdaLayerNormContinuous",
    "StableDiffusion3AdaLayerNorm",
    "StableDiffusion3GELUFeedForward",
]
