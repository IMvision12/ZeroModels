import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineLearnableAffineBlock(layers.Layer):
    """Learnable affine transformation: ``scale * x + bias``.

    Applies element-wise scaling and bias with two scalar parameters,
    used as part of the Learnable Affine Block (LAB) in HGNetV2.

    Args:
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(1,),
            initializer=keras.initializers.Constant(1.0),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer=keras.initializers.Constant(0.0),
        )
        super().build(input_shape)

    def call(self, x):
        return self.scale * x + self.bias

    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, dtype=x.dtype)


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineDecoderParams(layers.Layer):
    """Holds learnable ``up`` and ``reg_scale`` parameters for FDR decoding.

    Passes its input through unchanged and additionally returns the
    non-uniform FDR weighting vector (``project``) and ``reg_scale`` as
    graph tensors. These are computed inside ``call``: from the *current*
    weight values at run time: rather than from the raw weight variables
    at graph-build time. This matters because ``reg_scale`` varies across
    D-FINE variants (e.g. 4.0 for large, 8.0 for xlarge) and is only set
    during weight transfer: evaluating the weighting function eagerly at
    build time would bake in the initializer value and ignore the
    transferred one.

    Args:
        max_num_bins: Number of FDR distribution bins.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(self, max_num_bins=32, **kwargs):
        super().__init__(**kwargs)
        self.max_num_bins = max_num_bins

    def build(self, input_shape):
        self.up = self.add_weight(
            name="up",
            shape=(1,),
            initializer=keras.initializers.Constant(0.5),
        )
        self.reg_scale = self.add_weight(
            name="reg_scale",
            shape=(1,),
            initializer=keras.initializers.Constant(4.0),
        )
        super().build(input_shape)

    def weighting_function(self):
        abs_up = ops.abs(self.up)
        abs_rs = ops.abs(self.reg_scale)
        upper_bound1 = ops.multiply(abs_up, abs_rs)
        upper_bound2 = ops.multiply(upper_bound1, 2.0)
        step = ops.power(upper_bound1 + 1.0, 2.0 / (self.max_num_bins - 2))

        values = [ops.reshape(ops.negative(upper_bound2), [1])]
        for i in range(self.max_num_bins // 2 - 1, 0, -1):
            values.append(ops.reshape(ops.negative(ops.power(step, i)) + 1.0, [1]))
        values.append(ops.zeros([1], dtype=self.up.dtype))
        for i in range(1, self.max_num_bins // 2):
            values.append(ops.reshape(ops.power(step, i) - 1.0, [1]))
        values.append(ops.reshape(upper_bound2, [1]))
        return ops.concatenate(values, axis=0)

    def call(self, x):
        project = self.weighting_function()
        reg_scale = self.reg_scale * 1.0
        return x, project, reg_scale

    def compute_output_spec(self, x):
        return (
            keras.KerasTensor(x.shape, dtype=x.dtype),
            keras.KerasTensor((self.max_num_bins + 1,), dtype=self.compute_dtype),
            keras.KerasTensor((1,), dtype=self.compute_dtype),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"max_num_bins": self.max_num_bins})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineScalarParam(layers.Layer):
    """Holds a single learnable scalar parameter.

    Acts as identity on its input but ensures the scalar parameter is
    part of the computation graph for gradient tracking.

    Args:
        init_value (float): Initial value for the scalar. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(self, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.param = self.add_weight(
            name="param",
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_value),
        )
        super().build(input_shape)

    def call(self, x):
        return x + 0.0 * self.param

    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, dtype=x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineSinePositionEmbedding(layers.Layer):
    """2-D sinusoidal positional encoding for D-FINE hybrid encoder.

    Generates non-learnable sine/cosine positional encodings from a 2-D
    spatial grid. The embedding dimension is split into four equal parts
    encoding height-sin, height-cos, width-sin, and width-cos.

    Args:
        embed_dim (int): Total embedding dimension. Must be divisible
            by 4. Defaults to 256.
        temperature (int): Temperature scaling factor. Defaults to 10000.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(self, embed_dim=256, temperature=10000, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.temperature = temperature

    def call(self, height, width):
        pos_dim = self.embed_dim // 4
        dim_t = ops.cast(ops.arange(pos_dim), "float32") / pos_dim
        dim_t = 1.0 / (self.temperature**dim_t)

        grid_w = ops.cast(ops.arange(width), "float32")
        grid_h = ops.cast(ops.arange(height), "float32")
        grid_w, grid_h = ops.meshgrid(grid_w, grid_h)

        out_w = ops.reshape(grid_w, [-1, 1]) * ops.reshape(dim_t, [1, -1])
        out_h = ops.reshape(grid_h, [-1, 1]) * ops.reshape(dim_t, [1, -1])

        pos = ops.concatenate(
            [ops.sin(out_h), ops.cos(out_h), ops.sin(out_w), ops.cos(out_w)],
            axis=-1,
        )
        return ops.expand_dims(pos, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "temperature": self.temperature,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineMultiHeadAttention(layers.Layer):
    """Multi-head attention for the D-FINE transformer.

    Implements scaled dot-product multi-head attention with separate
    query, key, and value projections followed by an output projection.

    Args:
        hidden_dim (int): Total model dimension.
        num_heads (int): Number of parallel attention heads.
        dropout_rate (float): Dropout rate on attention weights.
            Defaults to 0.0.
        block_prefix (str): Name prefix for sub-layers.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        block_prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_q_proj")
        self.k_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_k_proj")
        self.v_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_v_proj")
        self.out_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_out_proj")
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [-1, ops.shape(q)[1], self.num_heads, self.head_dim])
        k = ops.reshape(k, [-1, ops.shape(k)[1], self.num_heads, self.head_dim])
        v = ops.reshape(v, [-1, ops.shape(v)[1], self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(
            attn_output, [-1, ops.shape(attn_output)[1], self.hidden_dim]
        )
        return self.out_proj(attn_output)

    def compute_output_spec(self, query, key, value, **kwargs):
        if not self.q_proj.built:
            self.q_proj.build(query.shape)
        if not self.k_proj.built:
            self.k_proj.build(key.shape)
        if not self.v_proj.built:
            self.v_proj.build(value.shape)
        if not self.out_proj.built:
            self.out_proj.build(query.shape)
        return keras.KerasTensor(query.shape, dtype=query.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


def ms_deform_attn_core_variable(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
):
    """Multi-scale deformable attention core with variable points per level.

    Performs bilinear sampling at learned offset locations across multiple
    feature levels, weighted by attention scores. Uses
    ``torch.nn.functional.grid_sample`` on the torch backend for exact
    numerical parity with the reference, and falls back to a pure-ops
    implementation on JAX and TensorFlow.

    Args:
        value: Tensor of shape ``(B, seq_len, n_heads, head_dim)``.
        value_spatial_shapes: List of ``(H, W)`` tuples per feature level.
        sampling_locations: Tensor ``(B, Len_q, n_heads, sum(points), 2)``
            with coordinates in ``[0, 1]``.
        attention_weights: Tensor ``(B, Len_q, n_heads, sum(points))``
            after softmax.
        num_points_list: List of integers, sampling points per level.

    Returns:
        Tensor of shape ``(B, Len_q, n_heads * head_dim)``.
    """
    B = ops.shape(value)[0]
    n_heads = ops.shape(value)[2]
    head_dim = ops.shape(value)[3]
    Len_q = ops.shape(sampling_locations)[1]
    value = ops.transpose(value, [0, 2, 3, 1])
    value = ops.reshape(value, [B * n_heads, head_dim, -1])

    sampling_grids = 2 * sampling_locations - 1

    sizes = [h * w for h, w in value_spatial_shapes]
    split_indices = []
    cum = 0
    for s in sizes[:-1]:
        cum += s
        split_indices.append(cum)
    value_list = ops.split(value, split_indices, axis=2)

    npoints_cumulative = []
    cum = 0
    for np_ in num_points_list[:-1]:
        cum += np_
        npoints_cumulative.append(cum)

    sampling_grids = ops.transpose(sampling_grids, [0, 2, 1, 3, 4])
    sampling_grids = ops.reshape(
        sampling_grids, [B * n_heads, Len_q, sum(num_points_list), 2]
    )
    grids_per_level = ops.split(sampling_grids, npoints_cumulative, axis=2)

    use_torch_grid_sample = keras.config.backend() == "torch"
    if use_torch_grid_sample:
        import torch.nn.functional as F

    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        value_l = ops.reshape(value_list[lid], [B * n_heads, head_dim, H, W])
        grid_l = grids_per_level[lid]

        if use_torch_grid_sample:
            sampled = F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            P = num_points_list[lid]
            value_l_t = ops.transpose(value_l, [0, 2, 3, 1])
            val_flat = ops.reshape(value_l_t, [B * n_heads, H * W, head_dim])

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

            valid_00 = ops.cast(
                (ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype
            )
            valid_01 = ops.cast(
                (ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype
            )
            valid_10 = ops.cast(
                (ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype
            )
            valid_11 = ops.cast(
                (ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype
            )

            ix0_c = ops.clip(ix0, 0, W - 1)
            ix1_c = ops.clip(ix1, 0, W - 1)
            iy0_c = ops.clip(iy0, 0, H - 1)
            iy1_c = ops.clip(iy1, 0, H - 1)

            def _gather(val_flat, iy, ix, BN, Len_q, P, H, W, head_dim):
                idx = iy * W + ix
                idx_flat = ops.reshape(idx, [BN, Len_q * P])
                idx_flat = ops.expand_dims(idx_flat, axis=-1)
                idx_flat = ops.repeat(idx_flat, head_dim, axis=-1)
                gathered = ops.take_along_axis(val_flat, idx_flat, axis=1)
                return ops.reshape(gathered, [BN, Len_q, P, head_dim])

            BN = B * n_heads
            v00 = _gather(val_flat, iy0_c, ix0_c, BN, Len_q, P, H, W, head_dim)
            v01 = _gather(val_flat, iy0_c, ix1_c, BN, Len_q, P, H, W, head_dim)
            v10 = _gather(val_flat, iy1_c, ix0_c, BN, Len_q, P, H, W, head_dim)
            v11 = _gather(val_flat, iy1_c, ix1_c, BN, Len_q, P, H, W, head_dim)

            v00 = v00 * ops.expand_dims(valid_00, axis=-1)
            v01 = v01 * ops.expand_dims(valid_01, axis=-1)
            v10 = v10 * ops.expand_dims(valid_10, axis=-1)
            v11 = v11 * ops.expand_dims(valid_11, axis=-1)

            fx = ops.expand_dims(fx, axis=-1)
            fy = ops.expand_dims(fy, axis=-1)

            w00 = (1 - fx) * (1 - fy)
            w01 = fx * (1 - fy)
            w10 = (1 - fx) * fy
            w11 = fx * fy

            sampled = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
            sampled = ops.transpose(sampled, [0, 3, 1, 2])

        sampling_value_list.append(sampled)

    sampling_values = ops.concatenate(sampling_value_list, axis=-1)

    attn = ops.transpose(attention_weights, [0, 2, 1, 3])
    attn = ops.reshape(attn, [B * n_heads, 1, Len_q, sum(num_points_list)])

    output = ops.sum(sampling_values * attn, axis=-1)
    output = ops.reshape(output, [B, n_heads * head_dim, Len_q])
    output = ops.transpose(output, [0, 2, 1])
    return output


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineMultiScaleDeformableAttention(layers.Layer):
    """Multi-scale deformable attention for D-FINE decoder cross-attention.

    Supports a variable number of sampling points per feature level via
    ``num_points_list``. Unlike RT-DETR, this layer has no value_proj or
    output_proj: the encoder hidden states are used directly as values.

    Args:
        hidden_dim (int): Model dimension. Defaults to 256.
        n_levels (int): Number of feature levels. Defaults to 3.
        n_heads (int): Number of attention heads. Defaults to 8.
        num_points_list (list[int]): Sampling points per level.
        offset_scale (float): Offset scale factor. Defaults to 0.5.
        spatial_shapes (list[tuple]): List of ``(H, W)`` tuples per level.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(
        self,
        hidden_dim=256,
        n_levels=3,
        n_heads=8,
        num_points_list=None,
        offset_scale=0.5,
        spatial_shapes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.num_points_list = num_points_list or [4] * n_levels
        self.offset_scale = offset_scale
        self.spatial_shapes = spatial_shapes or []

        total_points = n_heads * sum(self.num_points_list)
        self.sampling_offsets = layers.Dense(
            total_points * 2,
            name="sampling_offsets",
        )
        self.attention_weights_proj = layers.Dense(
            total_points,
            name="attention_weights",
        )

    def build(self, input_shape):
        num_points_scale = []
        for np_ in self.num_points_list:
            for _ in range(np_):
                num_points_scale.append(1.0 / np_)
        self.num_points_scale = self.add_weight(
            name="num_points_scale",
            shape=(sum(self.num_points_list),),
            initializer=keras.initializers.Constant(num_points_scale),
            trainable=False,
        )
        super().build(input_shape)

    def call(
        self,
        query,
        reference_points,
        input_flatten,
    ):
        input_spatial_shapes = self.spatial_shapes
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        head_dim = self.hidden_dim // self.n_heads
        total_pts = sum(self.num_points_list)

        value = ops.reshape(input_flatten, [N, -1, self.n_heads, head_dim])

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets,
            [N, Len_q, self.n_heads, total_pts, 2],
        )

        attention_weights = self.attention_weights_proj(query)
        attention_weights = ops.reshape(
            attention_weights,
            [N, Len_q, self.n_heads, total_pts],
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)

        num_coords = reference_points.shape[-1]
        if num_coords == 4:
            scale = ops.reshape(self.num_points_scale, [1, 1, 1, total_pts, 1])
            scale = ops.cast(scale, sampling_offsets.dtype)
            offset = (
                sampling_offsets
                * scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"reference_points last dim must be 4, got {num_coords}")

        output = ms_deform_attn_core_variable(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "n_levels": self.n_levels,
                "n_heads": self.n_heads,
                "num_points_list": self.num_points_list,
                "offset_scale": self.offset_scale,
                "spatial_shapes": self.spatial_shapes,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineDecoderLayer(layers.Layer):
    """Single D-FINE decoder layer with gateway-gated cross-attention.

    Combines self-attention, multi-scale deformable cross-attention with
    gateway gating, and a feedforward network. The gateway replaces the
    simple residual + layernorm after cross-attention: it concatenates
    the residual and cross-attention output, applies a gated linear unit,
    and then layer normalizes.

    Args:
        hidden_dim (int): Model dimension. Defaults to 256.
        num_heads (int): Number of attention heads. Defaults to 8.
        dim_feedforward (int): FFN intermediate dimension. Defaults to 1024.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        activation (str): FFN activation function. Defaults to ``"relu"``.
        n_levels (int): Number of feature levels. Defaults to 3.
        num_points_list (list[int]): Sampling points per level.
        offset_scale (float): Offset scale factor. Defaults to 0.5.
        spatial_shapes (list[tuple]): List of ``(H, W)`` tuples per level.
        block_prefix (str): Name prefix for sub-layers.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - D-FINE: https://arxiv.org/abs/2410.13842
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        dim_feedforward=1024,
        dropout_rate=0.0,
        activation="relu",
        n_levels=3,
        num_points_list=None,
        offset_scale=0.5,
        spatial_shapes=None,
        block_prefix="decoder_layers_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_levels = n_levels
        self.num_points_list = num_points_list or [4] * n_levels
        self.offset_scale = offset_scale
        self.spatial_shapes = spatial_shapes or []
        self.block_prefix = block_prefix

        bp = block_prefix
        self.self_attn = DFineMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,
            block_prefix=f"{bp}_self_attn",
            name=f"{bp}_self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_self_attn_layer_norm",
        )

        self.encoder_attn = DFineMultiScaleDeformableAttention(
            hidden_dim=self.hidden_dim,
            n_levels=self.n_levels,
            n_heads=self.num_heads,
            num_points_list=self.num_points_list,
            offset_scale=self.offset_scale,
            spatial_shapes=self.spatial_shapes,
            name=f"{bp}_encoder_attn",
        )

        self.gateway_gate = layers.Dense(
            hidden_dim * 2,
            name=f"{bp}_gateway_gate",
        )
        self.gateway_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_gateway_norm",
        )

        self.fc1 = layers.Dense(
            self.dim_feedforward,
            name=f"{bp}_fc1",
        )
        self.fc1_act = layers.Activation(self.activation)
        self.fc2 = layers.Dense(self.hidden_dim, name=f"{bp}_fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_final_layer_norm",
        )

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        query_pos,
        reference_points,
        training=None,
    ):
        q = k = hidden_states + query_pos
        residual = hidden_states
        attn_out = self.self_attn(q, k, hidden_states, training=training)
        hidden_states = self.self_attn_layer_norm(residual + attn_out)

        residual = hidden_states
        cross_input = hidden_states + query_pos
        cross_out = self.encoder_attn(
            cross_input,
            reference_points,
            encoder_hidden_states,
        )

        gate_input = ops.concatenate([residual, cross_out], axis=-1)
        gates = ops.sigmoid(self.gateway_gate(gate_input))
        gate1 = gates[..., : self.hidden_dim]
        gate2 = gates[..., self.hidden_dim :]
        hidden_states = self.gateway_norm(gate1 * residual + gate2 * cross_out)

        residual = hidden_states
        ff_out = self.fc2(self.fc1_act(self.fc1(hidden_states)))
        hidden_states = residual + ff_out
        hidden_states = ops.clip(hidden_states, -65504, 65504)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    def compute_output_spec(self, *args, **kwargs):
        return keras.KerasTensor(args[0].shape, dtype=args[0].dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "n_levels": self.n_levels,
                "num_points_list": self.num_points_list,
                "offset_scale": self.offset_scale,
                "spatial_shapes": self.spatial_shapes,
                "block_prefix": self.block_prefix,
            }
        )
        return config
