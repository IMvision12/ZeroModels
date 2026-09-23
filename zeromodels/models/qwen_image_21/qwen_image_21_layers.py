from __future__ import annotations

import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention
from zeromodels.models.qwen_image.qwen_image_layers import (
    QwenImageRMSNorm,
    apply_rotary_emb_qwen,
)
from zeromodels.models.stable_diffusion.stable_diffusion_layers import (
    safe_name,
    timestep_embedding,
)

NORM_EPS = 1e-6
MASK_NEG = -1e4
IMG_TOKENS_PER_SLOT = 4


def rope_params(index, dim, theta):
    freqs = np.outer(
        np.asarray(index, dtype=np.float64),
        1.0 / np.power(float(theta), np.arange(0, dim, 2, dtype=np.float64) / dim),
    )
    return freqs.astype(np.float32)


def build_qwenimage21_rope_angles(img_shapes, image_pad_mask, axes_dim, theta=10000):
    """Return RoPE angles ``[S, sum(axes_dim)//2]`` for a joint text/image layout.

    Mirrors Diffusers ``QwenImage21Rope`` (real-valued angles, not complex).
    ``image_pad_mask`` is a 1-D bool/int array of length ``S`` with ``True`` at
    image-token positions; ``img_shapes`` is a list of ``(frame, height, width)``.
    """
    image_pad_mask = np.asarray(image_pad_mask, dtype=bool)
    total_len = int(image_pad_mask.shape[0])
    axes_dim = list(axes_dim)

    pos_index = np.arange(8192)
    neg_index = np.arange(1024)[::-1] * -1 - 1
    freqs = [
        np.concatenate([rope_params(pos_index, dim, theta), rope_params(neg_index, dim, theta)], axis=0)
        for dim in axes_dim
    ]

    frame_index, image_height_index, image_width_index = [], [], []
    cursor, position = 0, 0
    is_image = image_pad_mask.tolist()

    for _, height, width in img_shapes:
        block_start = is_image.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len

        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)

        image_height_index.extend(
            [h for h in range(-(height - height // 2), height // 2) for _ in range(width)]
        )
        image_width_index.extend(
            [w for _ in range(height) for w in range(-(width - width // 2), width // 2)]
        )

    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))

    frame_index = np.asarray(frame_index, dtype=np.int64)
    height_index = frame_index.copy()
    width_index = frame_index.copy()
    height_index[image_pad_mask] = np.asarray(image_height_index, dtype=np.int64)
    width_index[image_pad_mask] = np.asarray(image_width_index, dtype=np.int64)

    return np.concatenate(
        [freqs[0][frame_index], freqs[1][height_index], freqs[2][width_index]],
        axis=-1,
    )


def build_token_metadata(image_pad_mask, img_shapes):
    """Label joint-sequence tokens with image-block ids (Diffusers helper)."""
    image_pad_mask = np.asarray(image_pad_mask, dtype=bool)
    image_positions = np.flatnonzero(image_pad_mask)
    block_lengths = [int(math.prod(shape)) for shape in img_shapes]
    if sum(block_lengths) != image_positions.size:
        raise ValueError(
            f"img_shapes accounts for {sum(block_lengths)} image tokens but "
            f"image_pad_mask marks {image_positions.size}."
        )
    image_ids = np.full(image_pad_mask.shape, -1, dtype=np.int32)
    block_ids = np.repeat(np.arange(len(block_lengths), dtype=np.int32), block_lengths)
    image_ids[image_positions] = block_ids
    target_token_mask = np.zeros_like(image_pad_mask, dtype=bool)
    target_token_mask[image_positions[-block_lengths[-1] :]] = True
    return image_ids, target_token_mask


def build_block_causal_additive_mask(image_ids, key_valid=None):
    """Dense additive attention mask for block-causal attention.

    Allowed when ``(q >= kv) or same_image_block``, and ``key_valid[kv]``.
    Returns ``(1, 1, S, S)`` float32 mask with ``0`` / ``MASK_NEG``.
    """
    image_ids = np.asarray(image_ids, dtype=np.int32)
    seq = image_ids.shape[0]
    q = np.arange(seq)[:, None]
    kv = np.arange(seq)[None, :]
    same = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    allowed = (q >= kv) | same
    if key_valid is not None:
        key_valid = np.asarray(key_valid, dtype=bool)
        allowed = allowed & key_valid[None, :]
    mask = np.where(allowed, 0.0, MASK_NEG).astype(np.float32)
    return mask[None, None]


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21ZeroCenterRMSNorm(layers.Layer):
    """RMSNorm with zero-centered weight (effective scale ``weight + 1``)."""

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
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x32 = ops.cast(x, "float32")
        rrms = ops.rsqrt(ops.mean(ops.square(x32), axis=-1, keepdims=True) + self.eps)
        out = x32 * rrms * (ops.cast(self.weight, "float32") + 1.0)
        return ops.cast(out, dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "module_path": self.module_path})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21TimestepProjEmbeddings(layers.Layer):
    """Diffusers ``QwenImage21TimestepProjEmbeddings`` (bias-free MLP)."""

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
            embedding_dim, use_bias=False, name=safe_name(f"{embedder}.linear_1")
        )
        self.linear_2 = layers.Dense(
            embedding_dim, use_bias=False, name=safe_name(f"{embedder}.linear_2")
        )

    def build(self, timestep_shape):
        freq_shape = (timestep_shape[0], self.time_freq_dim)
        self.linear_1.build(freq_shape)
        self.linear_2.build((timestep_shape[0], self.embedding_dim))
        self.built = True

    def call(self, timestep):
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
class QwenImage21TextProjection(layers.Layer):
    """Diffusers ``QwenImage21TextProjection``."""

    def __init__(self, context_in_dim, hidden_size, eps=NORM_EPS, module_path="txt_in", **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.eps = eps
        self.module_path = module_path
        self.text_norm = QwenImage21ZeroCenterRMSNorm(
            eps=eps, module_path=f"{module_path}.text_norm"
        )
        self.in_layer = layers.Dense(
            hidden_size, use_bias=False, name=safe_name(f"{module_path}.in_layer")
        )
        self.out_layer = layers.Dense(
            hidden_size, use_bias=False, name=safe_name(f"{module_path}.out_layer")
        )

    def build(self, input_shape):
        self.text_norm.build(input_shape)
        self.in_layer.build(input_shape)
        self.out_layer.build((*input_shape[:-1], self.hidden_size))
        self.built = True

    def call(self, x):
        x = self.text_norm(x)
        x = self.in_layer(x)
        x = keras.activations.gelu(x, approximate=True)
        return self.out_layer(x)

    def compute_output_shape(self, input_shape):
        return (*tuple(input_shape)[:-1], self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "context_in_dim": self.context_in_dim,
                "hidden_size": self.hidden_size,
                "eps": self.eps,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21SwiGLUFeedForward(layers.Layer):
    """Diffusers ``QwenImage21SwiGLUFeedForward``."""

    def __init__(self, hidden_size, mlp_hidden_size, module_path, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.module_path = module_path
        self.proj = layers.Dense(
            mlp_hidden_size, use_bias=False, name=safe_name(f"{module_path}.proj")
        )
        self.gate_layer = layers.Dense(
            mlp_hidden_size, use_bias=False, name=safe_name(f"{module_path}.gate_layer")
        )
        self.out = layers.Dense(
            hidden_size, use_bias=False, name=safe_name(f"{module_path}.out")
        )

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.gate_layer.build(input_shape)
        self.out.build((*input_shape[:-1], self.mlp_hidden_size))
        self.built = True

    def call(self, x):
        return self.out(ops.silu(self.gate_layer(x)) * self.proj(x))

    def compute_output_shape(self, input_shape):
        return (*tuple(input_shape)[:-1], self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mlp_hidden_size": self.mlp_hidden_size,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21AdaLayerNormContinuous(layers.Layer):
    """Final adaptive LayerNorm (scale only, no shift)."""

    def __init__(self, embedding_dim, conditioning_embedding_dim, eps=NORM_EPS, module_path="norm_out", **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.conditioning_embedding_dim = conditioning_embedding_dim
        self.eps = eps
        self.module_path = module_path
        self.linear = layers.Dense(
            embedding_dim, use_bias=False, name=safe_name(f"{module_path}.linear")
        )
        self.norm = layers.LayerNormalization(
            axis=-1, epsilon=eps, center=False, scale=False, name=safe_name(f"{module_path}.norm")
        )

    def build(self, input_shapes):
        hidden_shape = input_shapes[0]
        cond_shape = input_shapes[1]
        self.norm.build(hidden_shape)
        self.linear.build(cond_shape)
        self.built = True

    def call(self, inputs):
        hidden_states, conditioning, target_token_mask = inputs
        scale = self.linear(ops.silu(ops.cast(conditioning, hidden_states.dtype)))
        scale = select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1.0 + scale)

    def compute_output_shape(self, input_shapes):
        return tuple(input_shapes[0])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "conditioning_embedding_dim": self.conditioning_embedding_dim,
                "eps": self.eps,
                "module_path": self.module_path,
            }
        )
        return config


def select_modulation_rows(params, target_token_mask):
    """Broadcast per-sample modulation over tokens (causal_condition aware)."""
    if target_token_mask is None:
        return ops.expand_dims(params, 1)
    real = ops.expand_dims(params[:-1], 1)
    zero = ops.expand_dims(params[-1:], 0)
    mask = ops.reshape(ops.cast(target_token_mask, "bool"), (1, -1, 1))
    return ops.where(mask, real, zero)


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Attention(layers.Layer):
    """Single-stream attention (bias-free QKV / out, per-head QK RMSNorm)."""

    def __init__(self, dim, heads, dim_head, eps=NORM_EPS, module_path=None, **kwargs):
        if module_path is not None:
            kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.module_path = module_path
        inner = heads * dim_head
        path = module_path or "attn"
        self.to_q = layers.Dense(inner, use_bias=False, name=safe_name(f"{path}.to_q"))
        self.to_k = layers.Dense(inner, use_bias=False, name=safe_name(f"{path}.to_k"))
        self.to_v = layers.Dense(inner, use_bias=False, name=safe_name(f"{path}.to_v"))
        self.to_out = layers.Dense(dim, use_bias=False, name=safe_name(f"{path}.to_out.0"))
        self.norm_q = QwenImageRMSNorm(eps=eps, module_path=f"{path}.norm_q")
        self.norm_k = QwenImageRMSNorm(eps=eps, module_path=f"{path}.norm_k")

    def build(self, input_shape):
        self.to_q.build(input_shape)
        self.to_k.build(input_shape)
        self.to_v.build(input_shape)
        head_shape = (*input_shape[:-1], self.heads, self.dim_head)
        self.norm_q.build(head_shape)
        self.norm_k.build(head_shape)
        self.to_out.build((*input_shape[:-1], self.heads * self.dim_head))
        self.built = True

    def call(self, hidden_states, rotary_emb=None, attention_mask=None):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        def unflatten(x):
            shape = ops.shape(x)
            return ops.reshape(x, (shape[0], shape[1], self.heads, self.dim_head))

        query = self.norm_q(unflatten(query))
        key = self.norm_k(unflatten(key))
        value = unflatten(value)

        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb, use_real=True)
            key = apply_rotary_emb_qwen(key, rotary_emb, use_real=True)

        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))
        scale = self.dim_head**-0.5
        out = fused_attention(query, key, value, scale, attention_mask=attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        shape = ops.shape(out)
        out = ops.reshape(out, (shape[0], shape[1], self.heads * self.dim_head))
        return self.to_out(out)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def compute_output_spec(self, hidden_states, rotary_emb=None, attention_mask=None):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "heads": self.heads,
                "dim_head": self.dim_head,
                "eps": self.eps,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21TransformerBlock(layers.Layer):
    """Single-stream block with shared external modulation."""

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=3,
        eps=NORM_EPS,
        module_path=None,
        **kwargs,
    ):
        if module_path is not None:
            kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.mlp_ratio = mlp_ratio
        self.eps = eps
        self.module_path = module_path
        path = module_path or "block"
        self.img_norm1 = layers.LayerNormalization(
            axis=-1, epsilon=eps, center=False, scale=False, name=safe_name(f"{path}.img_norm1")
        )
        self.attn = QwenImage21Attention(
            dim, num_attention_heads, attention_head_dim, eps=eps, module_path=f"{path}.attn"
        )
        self.img_norm2 = layers.LayerNormalization(
            axis=-1, epsilon=eps, center=False, scale=False, name=safe_name(f"{path}.img_norm2")
        )
        self.img_mlp = QwenImage21SwiGLUFeedForward(
            dim, dim * mlp_ratio, module_path=f"{path}.img_mlp"
        )

    def build(self, input_shape):
        self.img_norm1.build(input_shape)
        self.attn.build(input_shape)
        self.img_norm2.build(input_shape)
        self.img_mlp.build(input_shape)
        self.built = True

    def modulate(self, hidden_states, mod_params, target_token_mask):
        scale, gate = ops.split(mod_params, 2, axis=-1)
        scale = select_modulation_rows(scale, target_token_mask)
        gate = select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1.0 + scale), gate

    def call(self, hidden_states, modulation, rotary_emb=None, attention_mask=None, target_token_mask=None):
        mod1, mod2 = ops.split(modulation, 2, axis=-1)
        img_modulated, img_gate1 = self.modulate(
            self.img_norm1(hidden_states), mod1, target_token_mask
        )
        attn_output = self.attn(
            img_modulated, rotary_emb=rotary_emb, attention_mask=attention_mask
        )
        hidden_states = hidden_states + ops.tanh(img_gate1) * attn_output

        img_modulated2, img_gate2 = self.modulate(
            self.img_norm2(hidden_states), mod2, target_token_mask
        )
        hidden_states = hidden_states + ops.tanh(img_gate2) * self.img_mlp(img_modulated2)

        if keras.backend.standardize_dtype(hidden_states.dtype) == "float16":
            hidden_states = ops.clip(hidden_states, -65504.0, 65504.0)
        return hidden_states

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def compute_output_spec(
        self,
        hidden_states,
        modulation,
        rotary_emb=None,
        attention_mask=None,
        target_token_mask=None,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_attention_heads": self.num_attention_heads,
                "attention_head_dim": self.attention_head_dim,
                "mlp_ratio": self.mlp_ratio,
                "eps": self.eps,
                "module_path": self.module_path,
            }
        )
        return config
