import math

import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def inverse_sigmoid(x, eps=1e-5):
    """Numerically safe ``logit``: ``log(x / (1 - x))`` with both terms clamped."""
    x = ops.clip(x, 0.0, 1.0)
    return ops.log(ops.maximum(x, eps) / ops.maximum(1.0 - x, eps))


def coordinate_sine_embedding(coords, num_pos_feats=128, temperature=10000):
    """Sinusoidal embedding of normalized coordinates, per coordinate.

    Each of the ``n`` coordinates in ``coords`` (``(..., n)``, values in
    ``[0, 1]``) is scaled by ``2 * pi`` and encoded with ``num_pos_feats``
    interleaved sin / cos features; the per-coordinate embeddings are
    concatenated. For two or more coordinates the x and y embeddings are swapped
    so the result follows the DETR ``[pos_y, pos_x, ...]`` convention.

    Returns:
        ``(..., n * num_pos_feats)``.
    """
    dim_t = ops.arange(num_pos_feats, dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)
    coords = ops.cast(coords, "float32")
    lead = tuple(ops.shape(coords)[:-1])
    embeddings = []
    for i in range(int(coords.shape[-1])):
        e = coords[..., i : i + 1] * (2 * math.pi) / dim_t
        e = ops.stack([ops.sin(e[..., 0::2]), ops.cos(e[..., 1::2])], axis=-1)
        embeddings.append(ops.reshape(e, lead + (num_pos_feats,)))
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return ops.concatenate(embeddings, axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDetrMLP(layers.Layer):
    """Feed-forward prediction head: ``num_layers`` Dense layers, ReLU between.

    Used for the decoder's ``ref_point_head`` (query position -> 2D reference
    point), ``query_scale`` (decoder content -> per-channel positional scale) and
    the detector's box head.

    Args:
        hidden_dim: Width of every hidden layer.
        output_dim: Width of the last layer.
        num_layers: Number of Dense layers.
        input_dim: Width of the input, used to build the layer eagerly.
            Defaults to ``hidden_dim``.
    """

    def __init__(self, hidden_dim, output_dim, num_layers, input_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.dense_layers = [
            layers.Dense(d, name=f"layers_{i}") for i, d in enumerate(dims)
        ]

    def build(self, input_shape=None):
        in_dim = self.input_dim
        for layer in self.dense_layers:
            layer.build((None, in_dim))
            in_dim = layer.units
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.relu(x)
        return x

    def compute_output_spec(self, x):
        return keras.KerasTensor(
            tuple(x.shape[:-1]) + (self.output_dim,), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "input_dim": self.input_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDetrDecoderLayer(layers.Layer):
    """One Conditional DETR decoder layer (post-norm).

    Self-attention whose queries and keys are the sum of separate content and
    position projections; then the conditional cross-attention, where each head's
    query is the concatenation of its content query and the (transformed) sine
    embedding of the query's 2D reference point, and each head's key the
    concatenation of the content key and the projected spatial position embedding
    (so the dot product splits into a content term and a spatial term). Only the
    first layer also adds the query position embedding to the content query (and
    the key position to the content key). Then a ReLU feed-forward block.

    Args:
        hidden_dim: Model width.
        num_heads: Number of attention heads.
        dim_feedforward: Feed-forward width.
        dropout_rate: Dropout after each sub-block.
        is_first: Whether this is the first decoder layer (owns
            ``ca_qpos_proj``).
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.1,
        is_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.is_first = is_first
        self.head_dim = hidden_dim // num_heads

        def dense(name, units=hidden_dim):
            return layers.Dense(units, name=name)

        self.sa_qcontent_proj = dense("sa_qcontent_proj")
        self.sa_qpos_proj = dense("sa_qpos_proj")
        self.sa_kcontent_proj = dense("sa_kcontent_proj")
        self.sa_kpos_proj = dense("sa_kpos_proj")
        self.sa_v_proj = dense("sa_v_proj")
        self.self_attn_out_proj = dense("self_attn_out_proj")
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_layer_norm"
        )

        self.ca_qcontent_proj = dense("ca_qcontent_proj")
        self.ca_qpos_proj = dense("ca_qpos_proj") if is_first else None
        self.ca_kcontent_proj = dense("ca_kcontent_proj")
        self.ca_kpos_proj = dense("ca_kpos_proj")
        self.ca_v_proj = dense("ca_v_proj")
        self.ca_qpos_sine_proj = dense("ca_qpos_sine_proj")
        self.encoder_attn_out_proj = dense("encoder_attn_out_proj")
        self.encoder_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="encoder_attn_layer_norm"
        )

        self.fc1 = dense("fc1", dim_feedforward)
        self.fc2 = dense("fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="final_layer_norm"
        )
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape=None):
        d = self.hidden_dim
        for layer in (
            self.sa_qcontent_proj,
            self.sa_qpos_proj,
            self.sa_kcontent_proj,
            self.sa_kpos_proj,
            self.sa_v_proj,
            self.self_attn_out_proj,
            self.ca_qcontent_proj,
            self.ca_qpos_proj,
            self.ca_kcontent_proj,
            self.ca_kpos_proj,
            self.ca_v_proj,
            self.ca_qpos_sine_proj,
            self.encoder_attn_out_proj,
            self.fc1,
            self.self_attn_layer_norm,
            self.encoder_attn_layer_norm,
            self.final_layer_norm,
        ):
            if layer is not None:
                layer.build((None, d))
        self.fc2.build((None, self.dim_feedforward))
        self.built = True

    def _split_heads(self, x, length, dim):
        b = ops.shape(x)[0]
        return ops.transpose(
            ops.reshape(x, (b, length, self.num_heads, dim)), (0, 2, 1, 3)
        )

    def _merge_heads(self, x, length):
        b = ops.shape(x)[0]
        return ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (b, length, -1))

    def call(
        self,
        hidden_states,
        query_position,
        query_sine_embed,
        memory,
        memory_position,
        training=None,
    ):
        num_queries = int(hidden_states.shape[1])
        memory_len = int(memory.shape[1])
        hd = self.head_dim

        # Self-attention: content + position projections for queries and keys.
        q = self.sa_qcontent_proj(hidden_states) + self.sa_qpos_proj(query_position)
        k = self.sa_kcontent_proj(hidden_states) + self.sa_kpos_proj(query_position)
        v = self.sa_v_proj(hidden_states)
        attn = fused_attention(
            self._split_heads(q, num_queries, hd),
            self._split_heads(k, num_queries, hd),
            self._split_heads(v, num_queries, hd),
            hd**-0.5,
        )
        attn = self.self_attn_out_proj(self._merge_heads(attn, num_queries))
        hidden_states = self.self_attn_layer_norm(
            hidden_states + self.dropout(attn, training=training)
        )

        # Conditional cross-attention: per-head [content | spatial] queries / keys.
        q = self.ca_qcontent_proj(hidden_states)
        k = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(memory_position)
        if self.is_first:
            q = q + self.ca_qpos_proj(query_position)
            k = k + k_pos
        q_sine = self.ca_qpos_sine_proj(query_sine_embed)
        q = ops.concatenate(
            [
                self._split_heads(q, num_queries, hd),
                self._split_heads(q_sine, num_queries, hd),
            ],
            axis=-1,
        )
        k = ops.concatenate(
            [
                self._split_heads(k, memory_len, hd),
                self._split_heads(k_pos, memory_len, hd),
            ],
            axis=-1,
        )
        attn = fused_attention(
            q, k, self._split_heads(v, memory_len, hd), (2 * hd) ** -0.5
        )
        attn = self.encoder_attn_out_proj(self._merge_heads(attn, num_queries))
        hidden_states = self.encoder_attn_layer_norm(
            hidden_states + self.dropout(attn, training=training)
        )

        # Feed-forward.
        ff = self.fc2(ops.relu(self.fc1(hidden_states)))
        return self.final_layer_norm(
            hidden_states + self.dropout(ff, training=training)
        )

    def compute_output_spec(self, hidden_states, *args, **kwargs):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "is_first": self.is_first,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDetrDecoder(layers.Layer):
    """The Conditional DETR decoder stack.

    Predicts one 2D reference point per object query from its position embedding
    (``ref_point_head`` + sigmoid), sine-embeds it, and runs the decoder layers;
    from the second layer on, the sine embedding is scaled channel-wise by
    ``query_scale(hidden_states)`` (the paper's transformation ``T``). The queries
    start at zero. A final LayerNorm is applied to the last layer's output.

    Call args:
        query_position: ``(B, num_queries, hidden_dim)`` learned query embeddings.
        memory: ``(B, HW, hidden_dim)`` encoder output.
        memory_position: ``(B, HW, hidden_dim)`` sine spatial embedding.

    Returns:
        ``(last_hidden_state, reference_points)``:
        ``(B, num_queries, hidden_dim)`` and ``(B, num_queries, 2)`` (sigmoid
        space, ``(x, y)``).
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.decoder_layers = [
            ConditionalDetrDecoderLayer(
                hidden_dim,
                num_heads,
                dim_feedforward,
                dropout_rate,
                is_first=(i == 0),
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.query_scale = ConditionalDetrMLP(
            hidden_dim, hidden_dim, 2, name="query_scale"
        )
        self.ref_point_head = ConditionalDetrMLP(
            hidden_dim, 2, 2, name="ref_point_head"
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-5, name="layernorm")

    def build(self, input_shape=None):
        for layer in self.decoder_layers:
            layer.build()
        self.query_scale.build()
        self.ref_point_head.build()
        self.layernorm.build((None, self.hidden_dim))
        self.built = True

    def call(self, query_position, memory, memory_position, training=None):
        reference_points = ops.sigmoid(self.ref_point_head(query_position))
        sine_embed = ops.cast(
            coordinate_sine_embedding(
                reference_points, num_pos_feats=self.hidden_dim // 2
            ),
            query_position.dtype,
        )
        hidden_states = ops.zeros_like(query_position)
        for i, layer in enumerate(self.decoder_layers):
            query_sine = (
                sine_embed if i == 0 else sine_embed * self.query_scale(hidden_states)
            )
            hidden_states = layer(
                hidden_states,
                query_position,
                query_sine,
                memory,
                memory_position,
                training=training,
            )
        return self.layernorm(hidden_states), reference_points

    def compute_output_spec(self, query_position, *args, **kwargs):
        shape = tuple(query_position.shape)
        return (
            keras.KerasTensor(shape, dtype=self.compute_dtype),
            keras.KerasTensor(shape[:-1] + (2,), dtype=self.compute_dtype),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDetrBoxDecode(layers.Layer):
    """Weightless box decoding relative to the query's reference point.

    ``sigmoid(delta + [logit(reference_xy), 0, 0])``: the box head predicts the
    center as an offset (in logit space) from the decoder's 2D reference point
    and the size directly. Returns ``(B, num_queries, 4)`` boxes as normalized
    ``(cx, cy, w, h)``.
    """

    def call(self, box_delta, reference_points):
        reference = ops.cast(inverse_sigmoid(reference_points), box_delta.dtype)
        center = box_delta[..., :2] + reference
        return ops.sigmoid(ops.concatenate([center, box_delta[..., 2:]], axis=-1))

    def compute_output_spec(self, box_delta, reference_points):
        return keras.KerasTensor(box_delta.shape, dtype=self.compute_dtype)
