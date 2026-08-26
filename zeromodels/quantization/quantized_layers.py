import keras
from keras import layers, ops

from .fp8_quantize import Fp8Quantizer, fp8_supported
from .int4_quantize import Int4Quantizer
from .int8_quantize import Int8Quantizer
from .mxfp4_quantize import MXFP4Quantizer, dequantize_mxfp4


class _LoadProxy:
    """A float stand-in surfaced in a quantized layer's ``weights`` during a
    no-float load (see :func:`~zeromodels.quantization.quantize_and_load`).

    It holds **no storage of its own**. A weight-transfer converter sees the
    layer's logical float kernel (right ``shape`` / ``path`` / ``name``), and its
    ``assign`` quantizes the incoming float straight into the layer's
    pre-allocated int kernel + scale, so the full float weight is never
    materialized on the device.

    ``assign_fn`` targets a specific bank when a layer has more than one (a fused
    MoE expert layer surfaces one proxy per bank); it defaults to the single-bank
    ``layer.assign_float_weight``.
    """

    def __init__(self, layer, shape, name, path, assign_fn=None):
        self._layer = layer
        self.shape = tuple(shape)
        self.name = name
        self.path = path
        self._assign_fn = assign_fn

    def assign(self, value):
        if self._assign_fn is not None:
            self._assign_fn(value)
        else:
            self._layer.assign_float_weight(value)


def get_quantizer(mode, group_size=32):
    """Build the :class:`~zeromodels.base.BaseQuantizer` for ``mode``."""
    if mode == "int8":
        return Int8Quantizer()
    if mode == "int4":
        return Int4Quantizer(group_size)
    if mode == "fp8":
        if not fp8_supported():
            raise ValueError(
                "fp8 quantization requires the torch or jax backend "
                "(tensorflow lacks float8 casts)."
            )
        return Fp8Quantizer()
    if mode == "mxfp4":
        return MXFP4Quantizer()
    raise ValueError(f"mode must be 'int8', 'int4', 'fp8', or 'mxfp4', got {mode!r}")


def einsum_contracting_axes(equation):
    """Kernel axes summed over in an ``EinsumDense`` equation.

    e.g. ``"abc,cde->abde"`` -> ``(0,)`` (``c``). Returns the kernel positions
    whose label appears in the input but not the output. Tolerates ``"..."``.
    """
    equation = equation.replace(" ", "")
    lhs, out_spec = equation.split("->")
    input_spec, kernel_spec = lhs.split(",")
    return tuple(
        i for i, ch in enumerate(kernel_spec) if ch in input_spec and ch not in out_spec
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class QuantizedDense(layers.Layer):
    """Weight-only int8 / int4 / fp8 / mxfp4 drop-in for ``keras.layers.Dense``.

    The kernel ``(in, out)`` is stored quantized (contracting ``axis=0``) and
    dequantized on the fly in ``call`` (the matmul runs in the activation dtype),
    so the model at rest is ~4x (int8 / fp8) or ~8x (int4 / mxfp4) smaller.
    Backend-agnostic; built from a trained ``Dense`` via :meth:`from_dense`.
    """

    def __init__(self, units, mode="int8", use_bias=True, group_size=32, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.mode = mode
        self.use_bias = use_bias
        self.group_size = group_size
        self.quantizer = get_quantizer(mode, group_size)
        self._loading = False
        self._loaded = False

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.in_features = in_dim
        spec = self.quantizer.storage_spec((in_dim, self.units), axis=0)
        (kernel_shape, kernel_dtype) = spec["kernel"]
        (scale_shape, scale_dtype) = spec["scale"]
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            dtype=kernel_dtype,
            initializer="zeros",
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            shape=scale_shape,
            dtype=scale_dtype,
            initializer="ones",
            trainable=False,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer="zeros", trainable=False
            )

    def call(self, inputs):
        kernel = self.quantizer.dequantize(
            self.kernel, self.scale, axis=0, dtype=inputs.dtype
        )
        y = ops.matmul(inputs, kernel)
        if self.use_bias:
            y = y + ops.cast(self.bias, y.dtype)
        return y

    def assign_float_weight(self, value):
        """Quantize a float kernel into this layer's int storage (no-float load)."""
        q, scale = self.quantizer.quantize(ops.convert_to_tensor(value), axis=0)
        self.kernel.assign(q)
        self.scale.assign(scale)
        self._loaded = True

    @property
    def weights(self):
        # During a no-float load, surface one float-shaped proxy for the kernel
        # (the converter assigns into it -> quantize) plus the real float bias.
        # The int kernel + scale stay hidden so the converter never sees them.
        if self._loading and self.built:
            proxy = _LoadProxy(
                self, (self.in_features, self.units), "kernel", self.kernel.path
            )
            return [proxy] + ([self.bias] if self.use_bias else [])
        return super().weights

    @classmethod
    def from_dense(cls, dense, mode, group_size=32):
        layer = cls(
            units=dense.units,
            mode=mode,
            use_bias=dense.use_bias,
            group_size=group_size,
            name=dense.name,
        )
        layer.build((None, int(dense.kernel.shape[0])))
        q, scale = layer.quantizer.quantize(dense.kernel, axis=0)
        layer.kernel.assign(q)
        layer.scale.assign(scale)
        if dense.use_bias:
            layer.bias.assign(dense.bias)
        return layer

    def to_dense(self):
        """Reconstruct a float ``keras.layers.Dense`` from the quantized weights."""
        kernel = self.quantizer.dequantize(self.kernel, self.scale, axis=0)
        dense = layers.Dense(self.units, use_bias=self.use_bias, name=self.name)
        dense.build((None, int(ops.shape(kernel)[0])))
        dense.kernel.assign(ops.cast(kernel, dense.kernel.dtype))
        if self.use_bias:
            dense.bias.assign(ops.cast(self.bias, dense.bias.dtype))
        return dense

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "mode": self.mode,
                "use_bias": self.use_bias,
                "group_size": self.group_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QuantizedEinsumDense(layers.Layer):
    """Weight-only int8 / int4 / fp8 / mxfp4 drop-in for ``keras.layers.EinsumDense``.

    Attention/projection kernels in some ports are N-D ``EinsumDense`` tensors
    (e.g. ``(hidden, heads, head_dim)``) whose contracting axis is not 0. This
    quantizes along the axes derived from the equation (a tuple for int8/fp8; a
    single axis is required for packed int4) and dequantizes on the fly. Built
    from a trained ``EinsumDense`` via :meth:`from_einsum_dense`.
    """

    def __init__(
        self,
        equation,
        output_shape,
        kernel_shape,
        mode="int8",
        bias_axes=None,
        bias_shape=None,
        group_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        self.partial_output_shape = tuple(output_shape)
        self.kernel_shape = tuple(kernel_shape)
        self.mode = mode
        self.bias_axes = bias_axes
        self.bias_shape = tuple(bias_shape) if bias_shape is not None else None
        self.group_size = group_size
        self.quantizer = get_quantizer(mode, group_size)
        self.axis = einsum_contracting_axes(equation)

    def build(self, input_shape=None):
        spec = self.quantizer.storage_spec(self.kernel_shape, axis=self.axis)
        (kernel_shape, kernel_dtype) = spec["kernel"]
        (scale_shape, scale_dtype) = spec["scale"]
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            dtype=kernel_dtype,
            initializer="zeros",
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            shape=scale_shape,
            dtype=scale_dtype,
            initializer="ones",
            trainable=False,
        )
        if self.bias_shape is not None:
            self.bias = self.add_weight(
                name="bias",
                shape=self.bias_shape,
                initializer="zeros",
                trainable=False,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        kernel = self.quantizer.dequantize(
            self.kernel, self.scale, axis=self.axis, dtype=inputs.dtype
        )
        y = ops.einsum(self.equation, inputs, kernel)
        if self.bias is not None:
            y = y + ops.cast(self.bias, y.dtype)
        return y

    @classmethod
    def from_einsum_dense(cls, einsum_dense, mode, group_size=32):
        bias = getattr(einsum_dense, "bias", None)
        layer = cls(
            equation=einsum_dense.equation,
            output_shape=einsum_dense.partial_output_shape,
            kernel_shape=tuple(einsum_dense.kernel.shape),
            mode=mode,
            bias_axes=einsum_dense.bias_axes,
            bias_shape=tuple(bias.shape) if bias is not None else None,
            group_size=group_size,
            name=einsum_dense.name,
        )
        layer.build()
        q, scale = layer.quantizer.quantize(einsum_dense.kernel, axis=layer.axis)
        layer.kernel.assign(q)
        layer.scale.assign(scale)
        if bias is not None:
            layer.bias.assign(bias)
        return layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "equation": self.equation,
                "output_shape": list(self.partial_output_shape),
                "kernel_shape": list(self.kernel_shape),
                "mode": self.mode,
                "bias_axes": self.bias_axes,
                "bias_shape": list(self.bias_shape) if self.bias_shape else None,
                "group_size": self.group_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QuantizedEmbedding(layers.Layer):
    """Weight-only int8 drop-in for ``keras.layers.Embedding``.

    The table is stored int8 with a per-row (per-token) scale (contracting
    ``axis=1``); the lookup gathers int8 rows and dequantizes only the gathered
    slice (the float table is never materialized). Used for both int8 and int4
    model modes: embeddings stay int8 (the 4-bit savings live in Dense weights).
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.quantizer = Int8Quantizer()
        self._loading = False
        self._loaded = False

    def build(self, input_shape=None):
        spec = self.quantizer.storage_spec((self.input_dim, self.output_dim), axis=1)
        (kernel_shape, kernel_dtype) = spec["kernel"]
        (scale_shape, scale_dtype) = spec["scale"]
        self.quantized_embeddings = self.add_weight(
            name="embeddings",
            shape=kernel_shape,
            dtype=kernel_dtype,
            initializer="zeros",
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            shape=scale_shape,
            dtype=scale_dtype,
            initializer="ones",
            trainable=False,
        )

    def call(self, inputs):
        ids = ops.cast(inputs, "int32")
        q = ops.take(self.quantized_embeddings, ids, axis=0)
        s = ops.take(self.scale, ids, axis=0)
        return ops.cast(q, self.compute_dtype) * ops.cast(s, self.compute_dtype)

    @property
    def embeddings(self):
        """Dequantized float table. Lets tied-output models that read
        ``token_embedding.embeddings`` directly (for the ``hidden @ embeddingsᵀ``
        logit projection) keep working after the embedding is quantized."""
        return self.quantizer.dequantize(
            self.quantized_embeddings, self.scale, axis=1, dtype=self.compute_dtype
        )

    def assign_float_weight(self, value):
        """Quantize a float table into this layer's int storage (no-float load)."""
        q, scale = self.quantizer.quantize(ops.convert_to_tensor(value), axis=1)
        self.quantized_embeddings.assign(q)
        self.scale.assign(scale)
        self._loaded = True

    @property
    def weights(self):
        # No-float load: surface a single float-shaped proxy named `embeddings`
        # for the converter to fill; the int table + scale stay hidden.
        if self._loading and self.built:
            proxy = _LoadProxy(
                self,
                (self.input_dim, self.output_dim),
                "embeddings",
                self.quantized_embeddings.path,
            )
            return [proxy]
        return super().weights

    @classmethod
    def from_embedding(cls, embedding):
        layer = cls(
            input_dim=embedding.input_dim,
            output_dim=embedding.output_dim,
            name=embedding.name,
        )
        layer.build()
        q, scale = layer.quantizer.quantize(embedding.embeddings, axis=1)
        layer.quantized_embeddings.assign(q)
        layer.scale.assign(scale)
        return layer

    def to_embedding(self):
        """Reconstruct a float ``keras.layers.Embedding`` from the quantized table."""
        table = self.quantizer.dequantize(self.quantized_embeddings, self.scale, axis=1)
        emb = layers.Embedding(self.input_dim, self.output_dim, name=self.name)
        emb.build(None)
        emb.embeddings.assign(ops.cast(table, emb.embeddings.dtype))
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QuantizedExperts(layers.Layer):
    """Weight-only quantized drop-in for a fused SwiGLU MoE expert bank.

    Replaces a ``...Experts`` layer that stores ``gate_up_proj`` ``(E, 2I, H)``
    and ``down_proj`` ``(E, H, I)`` as fused weights and runs the experts with
    ``einsum``. Both banks are quantized along their **contracting (last) axis**
    via the shared int8 / int4 / fp8 / mxfp4 quantizers and dequantized on the fly.
    ``activation`` is the gate nonlinearity (``"silu"`` for most MoE LLMs,
    ``"gelu"`` for Gemma-style).
    """

    def __init__(
        self,
        num_experts,
        embed_dim,
        mlp_dim,
        mode="int8",
        group_size=32,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode == "fp8" and not fp8_supported():
            raise ValueError(
                "fp8 quantization requires the torch or jax backend "
                "(tensorflow lacks float8 casts)."
            )
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.mode = mode
        self.group_size = group_size
        self.activation = activation
        self.quantizer = get_quantizer(mode, group_size)
        self._loading = False
        self._loaded = False
        self._gate_up_loaded = False
        self._down_loaded = False

    def build(self, input_shape=None):
        e, i, h = self.num_experts, self.mlp_dim, self.embed_dim
        gate_up = self.quantizer.storage_spec((e, 2 * i, h), axis=-1)
        down = self.quantizer.storage_spec((e, h, i), axis=-1)
        self.gate_up_q = self.add_weight(
            name="gate_up_proj",
            shape=gate_up["kernel"][0],
            dtype=gate_up["kernel"][1],
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_scale = self.add_weight(
            name="gate_up_scale",
            shape=gate_up["scale"][0],
            dtype=gate_up["scale"][1],
            initializer="ones",
            trainable=False,
        )
        self.down_q = self.add_weight(
            name="down_proj",
            shape=down["kernel"][0],
            dtype=down["kernel"][1],
            initializer="zeros",
            trainable=False,
        )
        self.down_scale = self.add_weight(
            name="down_scale",
            shape=down["scale"][0],
            dtype=down["scale"][1],
            initializer="ones",
            trainable=False,
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        act = ops.gelu if self.activation == "gelu" else ops.silu
        gate_up_w = self.quantizer.dequantize(
            self.gate_up_q, self.gate_up_scale, axis=-1, dtype=hidden_states.dtype
        )
        gate_up = ops.einsum("th,eoh->teo", hidden_states, gate_up_w)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        down_w = self.quantizer.dequantize(
            self.down_q, self.down_scale, axis=-1, dtype=hidden_states.dtype
        )
        expert_out = ops.einsum("tei,ehi->teh", act(gate) * up, down_w)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

    def assign_gate_up(self, value):
        """Quantize the fused ``gate_up_proj`` bank into int storage (no-float)."""
        q, scale = self.quantizer.quantize(ops.convert_to_tensor(value), axis=-1)
        self.gate_up_q.assign(q)
        self.gate_up_scale.assign(scale)
        self._gate_up_loaded = True
        self._loaded = self._gate_up_loaded and self._down_loaded

    def assign_down(self, value):
        """Quantize the fused ``down_proj`` bank into int storage (no-float)."""
        q, scale = self.quantizer.quantize(ops.convert_to_tensor(value), axis=-1)
        self.down_q.assign(q)
        self.down_scale.assign(scale)
        self._down_loaded = True
        self._loaded = self._gate_up_loaded and self._down_loaded

    @property
    def weights(self):
        # During a no-float load, surface one float-shaped proxy per fused bank
        # (the converter assigns into each -> quantize). The int storage stays
        # hidden so the converter never sees it.
        if self._loading and self.built:
            e, i, h = self.num_experts, self.mlp_dim, self.embed_dim
            return [
                _LoadProxy(
                    self,
                    (e, 2 * i, h),
                    "gate_up_proj",
                    self.gate_up_q.path,
                    self.assign_gate_up,
                ),
                _LoadProxy(
                    self, (e, h, i), "down_proj", self.down_q.path, self.assign_down
                ),
            ]
        return super().weights

    @classmethod
    def from_experts(cls, experts, mode, group_size=32, activation="silu"):
        layer = cls(
            experts.num_experts,
            experts.embed_dim,
            experts.mlp_dim,
            mode=mode,
            group_size=group_size,
            activation=activation,
            name=experts.name,
        )
        layer.build()
        gate_up_q, gate_up_scale = layer.quantizer.quantize(
            experts.gate_up_proj, axis=-1
        )
        down_q, down_scale = layer.quantizer.quantize(experts.down_proj, axis=-1)
        layer.gate_up_q.assign(gate_up_q)
        layer.gate_up_scale.assign(gate_up_scale)
        layer.down_q.assign(down_q)
        layer.down_scale.assign(down_scale)
        return layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "mode": self.mode,
                "group_size": self.group_size,
                "activation": self.activation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssMXFP4Experts(layers.Layer):
    """MXFP4-quantized GPT-OSS expert bank, matching the official packed weights.

    The MXFP4 counterpart of ``GptOssExperts``: it lives here beside the other
    quantized layers (not in the model file) so the model stays
    quantization-agnostic, the way transformers keeps its ``Mxfp4GptOssExperts``
    in ``integrations/mxfp4.py`` rather than ``modeling_gpt_oss.py``. Unlike the
    generic :class:`QuantizedExperts`, this one carries GPT-OSS's exact clamped
    gated-SiLU and per-expert biases and consumes the checkpoint's native
    ``_blocks`` / ``_scales`` layout directly.

    ``gate_up_proj`` and ``down_proj`` are stored in MXFP4 exactly as OpenAI ships
    them, uint8 nibble ``_blocks`` (two 4-bit codebook indices per byte) plus a
    uint8 e8m0 ``_scales`` exponent per 32-value block, ~4x smaller than bf16.
    They are dequantized on the fly in ``call`` and consumed in their natural
    ``(E, 2I, H)`` / ``(E, H, I)`` layout, so no transpose is needed and the block
    tensors map one-to-one onto the checkpoint. Biases stay full precision. On
    single-token decode only the top-k routed experts are dequantized.

    Args:
        num_experts: Number of experts ``E``.
        embed_dim: Model width ``H`` (must be a multiple of 32).
        mlp_dim: Per-expert hidden width ``I`` (must be a multiple of 32).
        num_experts_per_tok: Top-k experts routed per token (the sparse decode
            path dequantizes exactly this many).
    """

    def __init__(
        self, num_experts, embed_dim, mlp_dim, num_experts_per_tok=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.alpha = 1.702
        self.limit = 7.0

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        if h % 32 or i % 32:
            raise ValueError(
                f"MXFP4 packs 32-value blocks, so embed_dim ({h}) and mlp_dim ({i}) "
                f"must be multiples of 32."
            )
        # gate_up_proj (E, 2I, H) packed along H; down_proj (E, H, I) packed along I.
        # Shapes match OpenAI's *_blocks / *_scales exactly (direct checkpoint copy).
        self.gate_up_proj_blocks = self.add_weight(
            name="gate_up_proj_blocks",
            shape=(e, 2 * i, h // 32, 16),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_proj_scales = self.add_weight(
            name="gate_up_proj_scales",
            shape=(e, 2 * i, h // 32),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_proj_bias = self.add_weight(
            name="gate_up_proj_bias", shape=(e, 2 * i), initializer="zeros"
        )
        self.down_proj_blocks = self.add_weight(
            name="down_proj_blocks",
            shape=(e, h, i // 32, 16),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.down_proj_scales = self.add_weight(
            name="down_proj_scales",
            shape=(e, h, i // 32),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.down_proj_bias = self.add_weight(
            name="down_proj_bias", shape=(e, h), initializer="zeros"
        )
        self.built = True

    def _activate(self, gate_up, groups):
        # gate_up (T, groups, 2I) -> gated (T, groups, I): GPT-OSS's clamped
        # gated-SiLU on the interleaved gate/up halves.
        gate_up = ops.reshape(gate_up, (-1, groups, self.mlp_dim, 2))
        gate = ops.minimum(gate_up[..., 0], self.limit)
        up = ops.clip(gate_up[..., 1], -self.limit, self.limit)
        return (up + 1.0) * (gate * ops.sigmoid(gate * self.alpha))

    def call(self, hidden_states, routing_weights):
        # Prefill routes many tokens at once, and each routed expert dequantizes to
        # a large bf16 tensor, so process the tokens in fixed-size chunks to bound
        # the per-forward dequant peak. (transformers avoids the materialized dequant
        # entirely with a fused Triton kernel, or chunks the *weight* dequant at load
        # via rows_per_chunk; token-chunking the forward is the backend-agnostic
        # equivalent here.) A single-token decode (tokens <= chunk) runs in one pass;
        # tokens is static at trace time, and a whole-tensor pass is the fallback.
        tokens = hidden_states.shape[0]
        chunk = max(1, 16 // self.num_experts_per_tok)
        if tokens is not None and tokens > chunk:
            return ops.concatenate(
                [
                    self._route(
                        hidden_states[i : i + chunk], routing_weights[i : i + chunk]
                    )
                    for i in range(0, tokens, chunk)
                ],
                axis=0,
            )
        return self._route(hidden_states, routing_weights)

    def _route(self, hidden_states, routing_weights):
        # Evaluate only the experts each token routes to when that is cheaper than
        # dequantizing the whole bank (top_k vs all num_experts). Identical result
        # either way: the non-routed experts carry zero routing weight.
        tokens = hidden_states.shape[0]
        if tokens is not None and tokens * self.num_experts_per_tok < self.num_experts:
            return self._call_sparse(hidden_states, routing_weights)
        return self._call_dense(hidden_states, routing_weights)

    def _call_dense(self, hidden_states, routing_weights):
        dtype = hidden_states.dtype
        gate_up_proj = dequantize_mxfp4(
            self.gate_up_proj_blocks, self.gate_up_proj_scales, dtype
        )  # (E, 2I, H)
        down_proj = dequantize_mxfp4(
            self.down_proj_blocks, self.down_proj_scales, dtype
        )  # (E, H, I)
        gate_up = (
            ops.einsum("th,eih->tei", hidden_states, gate_up_proj)
            + self.gate_up_proj_bias
        )
        gated = self._activate(gate_up, self.num_experts)  # (T, E, I)
        expert_out = (
            ops.einsum("tei,ehi->teh", gated, down_proj) + self.down_proj_bias
        )  # (T, E, H)
        return ops.einsum("te,teh->th", routing_weights, expert_out)  # (T, H)

    def _call_sparse(self, hidden_states, routing_weights):
        # Gather + dequantize only the top_k routed experts per token. routing_weights
        # (T, E) has exactly top_k nonzero entries per row, so top_k recovers the
        # routed indices + weights with a static (T, k) shape.
        dtype = hidden_states.dtype
        weights, idx = ops.top_k(routing_weights, self.num_experts_per_tok)  # (T, k)
        gate_up_proj = dequantize_mxfp4(
            ops.take(self.gate_up_proj_blocks, idx, axis=0),
            ops.take(self.gate_up_proj_scales, idx, axis=0),
            dtype,
        )  # (T, k, 2I, H)
        gate_up = ops.einsum("th,tkjh->tkj", hidden_states, gate_up_proj) + ops.take(
            self.gate_up_proj_bias, idx, axis=0
        )  # (T, k, 2I)
        gated = self._activate(gate_up, self.num_experts_per_tok)  # (T, k, I)
        down_proj = dequantize_mxfp4(
            ops.take(self.down_proj_blocks, idx, axis=0),
            ops.take(self.down_proj_scales, idx, axis=0),
            dtype,
        )  # (T, k, H, I)
        expert_out = ops.einsum("tki,tkhi->tkh", gated, down_proj) + ops.take(
            self.down_proj_bias, idx, axis=0
        )  # (T, k, H)
        return ops.einsum("tk,tkh->th", weights, expert_out)  # (T, H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_experts_per_tok": self.num_experts_per_tok,
            }
        )
        return config
