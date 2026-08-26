import math

import keras
from keras import layers, ops


def ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Multi-scale deformable attention via manual bilinear sampling.

    Args:
        value: ``(B, n_heads, head_dim, N_total)``.
        value_spatial_shapes: list of ``(H, W)`` per level.
        sampling_locations: ``(B, Len_q, n_heads, L, P, 2)`` in ``[0, 1]``.
        attention_weights: ``(B, Len_q, n_heads, L*P)`` (already softmaxed).

    Returns:
        ``(B, Len_q, n_heads*head_dim)``.
    """
    B = ops.shape(value)[0]
    n_heads = int(value.shape[1])
    head_dim = int(value.shape[2])
    Len_q = ops.shape(sampling_locations)[1]
    L = len(value_spatial_shapes)
    P = int(sampling_locations.shape[4])

    sampling_grids = 2 * sampling_locations - 1
    sizes = [h * w for h, w in value_spatial_shapes]
    split_indices = []
    cum = 0
    for s in sizes[:-1]:
        cum += s
        split_indices.append(cum)
    value_list = ops.split(value, split_indices, axis=3)

    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        value_l = ops.reshape(value_list[lid], [B * n_heads, head_dim, H, W])
        value_l = ops.transpose(value_l, [0, 2, 3, 1])
        val_flat = ops.reshape(value_l, [B * n_heads, H * W, head_dim])

        grid_l = sampling_grids[:, :, :, lid, :, :]
        grid_l = ops.transpose(grid_l, [0, 2, 1, 3, 4])
        grid_l = ops.reshape(grid_l, [B * n_heads, Len_q, P, 2])

        grid_x = grid_l[..., 0]
        grid_y = grid_l[..., 1]
        W_f = ops.cast(W, grid_x.dtype)
        H_f = ops.cast(H, grid_y.dtype)
        ix = ((grid_x + 1) * W_f - 1) / 2.0
        iy = ((grid_y + 1) * H_f - 1) / 2.0

        ix0 = ops.cast(ops.floor(ix), "int32")
        iy0 = ops.cast(ops.floor(iy), "int32")
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        fx = ix - ops.cast(ix0, ix.dtype)
        fy = iy - ops.cast(iy0, iy.dtype)

        valid_00 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_01 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_10 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)
        valid_11 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)

        ix0_c = ops.clip(ix0, 0, W - 1)
        ix1_c = ops.clip(ix1, 0, W - 1)
        iy0_c = ops.clip(iy0, 0, H - 1)
        iy1_c = ops.clip(iy1, 0, H - 1)
        BN = B * n_heads

        def gather(iy, ix):
            idx = iy * W + ix
            idx_flat = ops.reshape(idx, [BN, Len_q * P])
            idx_flat = ops.repeat(ops.expand_dims(idx_flat, -1), head_dim, axis=-1)
            g = ops.take_along_axis(val_flat, idx_flat, axis=1)
            return ops.reshape(g, [BN, Len_q, P, head_dim])

        v00 = gather(iy0_c, ix0_c) * ops.expand_dims(valid_00, -1)
        v01 = gather(iy0_c, ix1_c) * ops.expand_dims(valid_01, -1)
        v10 = gather(iy1_c, ix0_c) * ops.expand_dims(valid_10, -1)
        v11 = gather(iy1_c, ix1_c) * ops.expand_dims(valid_11, -1)
        fx = ops.expand_dims(fx, -1)
        fy = ops.expand_dims(fy, -1)
        sampled = (
            v00 * (1 - fx) * (1 - fy)
            + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy
            + v11 * fx * fy
        )
        sampling_value_list.append(ops.transpose(sampled, [0, 3, 1, 2]))

    sampling_values = ops.stack(sampling_value_list, axis=-2)
    sampling_values = ops.reshape(
        sampling_values, [B * n_heads, head_dim, Len_q, L * P]
    )
    attn = ops.transpose(attention_weights, [0, 2, 1, 3])
    attn = ops.reshape(attn, [B * n_heads, 1, Len_q, L * P])
    output = ops.sum(sampling_values * attn, axis=-1)
    output = ops.reshape(output, [B, n_heads * head_dim, Len_q])
    return ops.transpose(output, [0, 2, 1])


def encode_sine_position(pos_tensor, num_pos_feats=128, temperature=10000):
    """Per-coordinate interleaved sin/cos embedding of normalized coords.

    ``pos_tensor`` is ``(..., n_coords)``; returns ``(..., n_coords*num_pos_feats)``
    with the x and y embeddings swapped (DETR convention) for >=2 coords.
    """
    scale = 2 * math.pi
    dim_t = ops.arange(num_pos_feats, dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)
    n = int(pos_tensor.shape[-1])
    embeddings = []
    for i in range(n):
        coord = pos_tensor[..., i]
        e = coord[..., None] * scale / dim_t
        e = ops.reshape(
            ops.stack([ops.sin(e[..., 0::2]), ops.cos(e[..., 1::2])], axis=-1),
            ops.shape(e),
        )
        embeddings.append(e)
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return ops.concatenate(embeddings, axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoMLPPredictionHead(layers.Layer):
    """Simple MLP (``num_layers`` dense layers, ReLU between) for box regression."""

    def __init__(self, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.dense_layers = [
            layers.Dense(d, name=f"layers_{i}") for i, d in enumerate(dims)
        ]

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = ops.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoContrastiveEmbedding(layers.Layer):
    """Vision-text contrastive scores padded to ``max_text_len`` (no weights)."""

    def __init__(self, max_text_len=256, **kwargs):
        super().__init__(**kwargs)
        self.max_text_len = max_text_len

    def call(self, vision_hidden, text_hidden, text_token_mask):
        output = ops.matmul(vision_hidden, ops.transpose(text_hidden, (0, 2, 1)))
        mask = text_token_mask[:, None, :]
        output = ops.where(mask, output, -1e8)
        text_len = int(output.shape[-1])
        if text_len < self.max_text_len:
            pad = ops.full(
                (
                    ops.shape(output)[0],
                    int(output.shape[1]),
                    self.max_text_len - text_len,
                ),
                -1e8,
                dtype=output.dtype,
            )
            output = ops.concatenate([output, pad], axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"max_text_len": self.max_text_len})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoDeformableAttention(layers.Layer):
    """Multi-scale deformable attention (Deformable DETR)."""

    def __init__(self, d_model, num_heads, n_levels, n_points, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // num_heads
        self.sampling_offsets = layers.Dense(
            num_heads * n_levels * n_points * 2, name="sampling_offsets"
        )
        self.attention_weights = layers.Dense(
            num_heads * n_levels * n_points, name="attention_weights"
        )
        self.value_proj = layers.Dense(d_model, name="value_proj")
        self.output_proj = layers.Dense(d_model, name="output_proj")

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        reference_points,
        spatial_shapes_list=None,
        position_embeddings=None,
        attention_mask=None,
    ):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        b = ops.shape(hidden_states)[0]
        lq = int(hidden_states.shape[1])
        s = int(encoder_hidden_states.shape[1])
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value * ops.cast(attention_mask[..., None], value.dtype)
        value = ops.reshape(value, (b, s, self.num_heads, self.head_dim))
        offsets = ops.reshape(
            self.sampling_offsets(hidden_states),
            (b, lq, self.num_heads, self.n_levels, self.n_points, 2),
        )
        attn = ops.reshape(
            self.attention_weights(hidden_states),
            (b, lq, self.num_heads, self.n_levels * self.n_points),
        )
        attn = ops.softmax(ops.cast(attn, "float32"), axis=-1)
        attn = ops.cast(
            ops.reshape(attn, (b, lq, self.num_heads, self.n_levels, self.n_points)),
            value.dtype,
        )
        num_coords = int(reference_points.shape[-1])
        if num_coords == 2:
            wh = ops.convert_to_tensor(
                [[w, h] for h, w in spatial_shapes_list], dtype="float32"
            )
            normalizer = ops.cast(wh, offsets.dtype)
            loc = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        else:
            loc = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        value_t = ops.transpose(value, (0, 2, 3, 1))
        aw = ops.reshape(attn, (b, lq, self.num_heads, self.n_levels * self.n_points))
        out = ms_deform_attn_core(value_t, spatial_shapes_list, loc, aw)
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoMultiheadAttention(layers.Layer):
    """Standard MHA (separate q/k/v + out_proj), additive mask."""

    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = layers.Dense(hidden_size, name="query")
        self.key = layers.Dense(hidden_size, name="key")
        self.value = layers.Dense(hidden_size, name="value")
        self.out_proj = layers.Dense(hidden_size, name="out_proj")

    def call(self, queries, keys, values, attention_mask=None):
        b = ops.shape(queries)[0]
        qn = int(queries.shape[1])
        kn = int(keys.shape[1])

        def split(t, n):
            return ops.transpose(
                ops.reshape(t, (b, n, self.num_heads, self.head_dim)), (0, 2, 1, 3)
            )

        q = split(self.query(queries), qn)
        k = split(self.key(keys), kn)
        v = split(self.value(values), kn)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn = attn + ops.cast(attention_mask, attn.dtype)
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        ctx = ops.matmul(attn, v)
        ctx = ops.reshape(ops.transpose(ctx, (0, 2, 1, 3)), (b, qn, self.hidden_size))
        return self.out_proj(ctx)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoTextEnhancerLayer(layers.Layer):
    """Text self-attention enhancer (half heads, half ffn)."""

    def __init__(self, d_model, num_heads, ffn_dim, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.eps = eps
        self.self_attn = GroundingDinoMultiheadAttention(
            d_model, num_heads, name="self_attn"
        )
        self.fc1 = layers.Dense(ffn_dim, name="fc1")
        self.fc2 = layers.Dense(d_model, name="fc2")
        self.layer_norm_before = layers.LayerNormalization(
            epsilon=eps, name="layer_norm_before"
        )
        self.layer_norm_after = layers.LayerNormalization(
            epsilon=eps, name="layer_norm_after"
        )

    def call(self, hidden_states, attention_mask=None, position_embeddings=None):
        q = k = (
            hidden_states
            if position_embeddings is None
            else hidden_states + position_embeddings
        )
        attn_out = self.self_attn(q, k, hidden_states, attention_mask=attention_mask)
        hidden_states = self.layer_norm_before(hidden_states + attn_out)
        residual = hidden_states
        hidden_states = self.fc2(ops.relu(self.fc1(hidden_states)))
        return self.layer_norm_after(hidden_states + residual)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoBiMultiHeadAttention(layers.Layer):
    """Bi-directional vision<->text cross-attention for the fusion layer."""

    def __init__(self, embed_dim, num_heads, vision_dim, text_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.vision_proj = layers.Dense(embed_dim, name="vision_proj")
        self.text_proj = layers.Dense(embed_dim, name="text_proj")
        self.values_vision_proj = layers.Dense(embed_dim, name="values_vision_proj")
        self.values_text_proj = layers.Dense(embed_dim, name="values_text_proj")
        self.out_vision_proj = layers.Dense(vision_dim, name="out_vision_proj")
        self.out_text_proj = layers.Dense(text_dim, name="out_text_proj")

    def reshape(self, t, n, b):
        return ops.reshape(
            ops.transpose(
                ops.reshape(t, (b, n, self.num_heads, self.head_dim)), (0, 2, 1, 3)
            ),
            (b * self.num_heads, n, self.head_dim),
        )

    def call(self, vision, text, vision_attention_mask=None, text_attention_mask=None):
        b = ops.shape(vision)[0]
        tgt = int(vision.shape[1])
        src = int(text.shape[1])
        vq = self.reshape(self.vision_proj(vision) * self.scale, tgt, b)
        tk = self.reshape(self.text_proj(text), src, b)
        vv = self.reshape(self.values_vision_proj(vision), tgt, b)
        tv = self.reshape(self.values_text_proj(text), src, b)

        attn = ops.matmul(vq, ops.transpose(tk, (0, 2, 1)))  # (b*nh, tgt, src)
        attn = attn - ops.max(attn)
        attn = ops.clip(attn, -50000, 50000)
        text_attn = ops.transpose(attn, (0, 2, 1))
        text_attn = text_attn - ops.max(text_attn, axis=-1, keepdims=True)
        text_attn = ops.clip(text_attn, -50000, 50000)
        if vision_attention_mask is not None:
            vm = ops.reshape(
                ops.broadcast_to(
                    vision_attention_mask[:, None, None, :],
                    (b, self.num_heads, 1, tgt),
                ),
                (b * self.num_heads, 1, tgt),
            )
            text_attn = ops.where(vm, -1e9, text_attn)
        text_attn = ops.softmax(ops.cast(text_attn, "float32"), axis=-1)
        if text_attention_mask is not None:
            tm = ops.reshape(
                ops.broadcast_to(
                    text_attention_mask[:, None, None, :],
                    (b, self.num_heads, 1, src),
                ),
                (b * self.num_heads, 1, src),
            )
            attn = ops.where(tm, -1e9, attn)
        vision_attn = ops.softmax(ops.cast(attn, "float32"), axis=-1)

        vision_out = ops.matmul(ops.cast(vision_attn, tv.dtype), tv)
        text_out = ops.matmul(ops.cast(text_attn, vv.dtype), vv)
        vision_out = ops.reshape(
            ops.transpose(
                ops.reshape(vision_out, (b, self.num_heads, tgt, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (b, tgt, self.embed_dim),
        )
        text_out = ops.reshape(
            ops.transpose(
                ops.reshape(text_out, (b, self.num_heads, src, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (b, src, self.embed_dim),
        )
        return self.out_vision_proj(vision_out), self.out_text_proj(text_out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoFusionLayer(layers.Layer):
    """Pre-norm bi-directional fusion with learned per-channel layer scale."""

    def __init__(self, d_model, embed_dim, num_heads, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.eps = eps
        self.layer_norm_vision = layers.LayerNormalization(
            epsilon=eps, name="layer_norm_vision"
        )
        self.layer_norm_text = layers.LayerNormalization(
            epsilon=eps, name="layer_norm_text"
        )
        self.attn = GroundingDinoBiMultiHeadAttention(
            embed_dim, num_heads, d_model, d_model, name="attn"
        )

    def build(self, input_shape):
        self.vision_param = self.add_weight(
            name="vision_param",
            shape=(self.d_model,),
            initializer=keras.initializers.Constant(1e-4),
            trainable=True,
        )
        self.text_param = self.add_weight(
            name="text_param",
            shape=(self.d_model,),
            initializer=keras.initializers.Constant(1e-4),
            trainable=True,
        )
        self.built = True

    def call(self, vision, text, vision_mask=None, text_mask=None):
        vision = self.layer_norm_vision(vision)
        text = self.layer_norm_text(text)
        delta_v, delta_t = self.attn(
            vision,
            text,
            vision_attention_mask=vision_mask,
            text_attention_mask=text_mask,
        )
        vision = vision + self.vision_param * delta_v
        text = text + self.text_param * delta_t
        return vision, text

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoDeformableLayer(layers.Layer):
    """Encoder image self-attention (deformable) + FFN, both post-norm."""

    def __init__(
        self, d_model, num_heads, n_levels, n_points, ffn_dim, eps=1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.ffn_dim = ffn_dim
        self.eps = eps
        self.self_attn = GroundingDinoDeformableAttention(
            d_model, num_heads, n_levels, n_points, name="self_attn"
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="self_attn_layer_norm"
        )
        self.fc1 = layers.Dense(ffn_dim, name="fc1")
        self.fc2 = layers.Dense(d_model, name="fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states,
        position_embeddings,
        reference_points,
        spatial_shapes_list=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            hidden_states,
            reference_points,
            spatial_shapes_list=spatial_shapes_list,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.fc2(ops.relu(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "ffn_dim": self.ffn_dim,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoEncoderLayer(layers.Layer):
    """One GD encoder layer: fusion -> text enhancer -> image deformable."""

    def __init__(
        self,
        d_model,
        encoder_heads,
        encoder_ffn_dim,
        n_levels,
        n_points,
        eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_heads = encoder_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.n_levels = n_levels
        self.n_points = n_points
        self.eps = eps
        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(
            d_model,
            encoder_heads // 2,
            encoder_ffn_dim // 2,
            eps,
            name="text_enhancer_layer",
        )
        self.fusion_layer = GroundingDinoFusionLayer(
            d_model, encoder_ffn_dim // 2, encoder_heads // 2, eps, name="fusion_layer"
        )
        self.deformable_layer = GroundingDinoDeformableLayer(
            d_model,
            encoder_heads,
            n_levels,
            n_points,
            encoder_ffn_dim,
            eps,
            name="deformable_layer",
        )

    def call(
        self,
        vision_features,
        text_features,
        vision_position_embedding,
        reference_points,
        spatial_shapes_list=None,
        text_position_embedding=None,
        text_self_attention_mask=None,
        key_padding_mask=None,
        text_attention_mask=None,
    ):
        vision_features, text_features = self.fusion_layer(
            vision_features,
            text_features,
            vision_mask=key_padding_mask,
            text_mask=text_attention_mask,
        )
        text_features = self.text_enhancer_layer(
            text_features,
            attention_mask=text_self_attention_mask,
            position_embeddings=text_position_embedding,
        )
        vision_features = self.deformable_layer(
            vision_features,
            vision_position_embedding,
            reference_points,
            spatial_shapes_list=spatial_shapes_list,
            attention_mask=None if key_padding_mask is None else ~key_padding_mask,
        )
        return vision_features, text_features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "encoder_heads": self.encoder_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoDecoderLayer(layers.Layer):
    """Decoder layer: self-attn -> text cross-attn -> image deformable cross-attn -> FFN."""

    def __init__(
        self,
        d_model,
        decoder_heads,
        decoder_ffn_dim,
        n_levels,
        n_points,
        eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.decoder_heads = decoder_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.n_levels = n_levels
        self.n_points = n_points
        self.eps = eps
        self.self_attn = GroundingDinoMultiheadAttention(
            d_model, decoder_heads, name="self_attn"
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="self_attn_layer_norm"
        )
        self.encoder_attn_text = GroundingDinoMultiheadAttention(
            d_model, decoder_heads, name="encoder_attn_text"
        )
        self.encoder_attn_text_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="encoder_attn_text_layer_norm"
        )
        self.encoder_attn = GroundingDinoDeformableAttention(
            d_model, decoder_heads, n_levels, n_points, name="encoder_attn"
        )
        self.encoder_attn_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="encoder_attn_layer_norm"
        )
        self.fc1 = layers.Dense(decoder_ffn_dim, name="fc1")
        self.fc2 = layers.Dense(d_model, name="fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states,
        position_embeddings,
        reference_points,
        vision_encoder_hidden_states,
        text_encoder_hidden_states,
        spatial_shapes_list=None,
        text_encoder_attention_mask=None,
        vision_encoder_attention_mask=None,
    ):
        residual = hidden_states
        q = k = hidden_states + position_embeddings
        hidden_states = self.self_attn(q, k, hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        q = hidden_states + position_embeddings
        hidden_states = self.encoder_attn_text(
            q,
            text_encoder_hidden_states,
            text_encoder_hidden_states,
            attention_mask=text_encoder_attention_mask,
        )
        hidden_states = self.encoder_attn_text_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states,
            vision_encoder_hidden_states,
            reference_points,
            spatial_shapes_list=spatial_shapes_list,
            position_embeddings=position_embeddings,
            attention_mask=vision_encoder_attention_mask,
        )
        hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.fc2(ops.relu(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "decoder_heads": self.decoder_heads,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "eps": self.eps,
            }
        )
        return config
