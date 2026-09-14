import contextlib
import contextvars
import functools

import keras
from keras import ops

VALID_ATTN_IMPL = ("sdpa", "fused", "flash")
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"

# The attention implementation explicitly chosen for the current build or
# forward (``None``: nothing chosen, each layer falls back to its own default,
# ``"sdpa"`` unless a model family says otherwise). This is a ContextVar, not a
# plain module global: ``Model.from_weights`` activates a model's implementation
# only for the scope of that model's build (so a functional graph bakes the
# right branch) and each of that model's forwards / generation steps (so an
# imperative decode reads it), always restoring the previous value. A per-model
# choice therefore never leaks to another model, a later forward, or another
# thread.
_ATTN_IMPLEMENTATION = contextvars.ContextVar("attn_implementation", default=None)


@contextlib.contextmanager
def use_attn_implementation(attn_implementation):
    """Activate ``attn_implementation`` for the enclosed build or forward.

    ``None`` clears the choice (layers use their default, ``"sdpa"`` unless the
    model family picks another, such as the SD 3 MMDiT's ``"fused"``). The
    previous value is always restored on exit, so a per-model implementation
    never leaks past its scope.
    """
    impl = attn_implementation
    if impl is not None and impl not in VALID_ATTN_IMPL:
        raise ValueError(
            f"attn_implementation must be one of {VALID_ATTN_IMPL}, got {impl!r}"
        )
    token = _ATTN_IMPLEMENTATION.set(impl)
    try:
        yield
    finally:
        _ATTN_IMPLEMENTATION.reset(token)


def active_attn_implementation():
    """The implementation explicitly activated for the current build or forward
    (by ``Model.from_weights`` or :func:`use_attn_implementation`), or ``None``.
    """
    return _ATTN_IMPLEMENTATION.get()


def with_model_attn_implementation(method):
    """Decorate a model method so it runs under the model's captured implementation.

    A no-op when the model carries no explicit ``_attn_implementation`` (so a
    sub-model built inside another model's scope inherits the outer choice).
    Used on imperative entry points (generation) that run attention outside the
    functional graph, where the baked-at-build branch does not apply.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        impl = getattr(self, "_attn_implementation", None)
        if impl is None:
            return method(self, *args, **kwargs)
        with use_attn_implementation(impl):
            return method(self, *args, **kwargs)

    return wrapper


def _fused_op_available():
    backend = keras.config.backend()
    if backend == "jax":
        try:
            from jax.nn import dot_product_attention  # noqa: F401

            return True
        except ImportError:
            return False
    if backend == "torch":
        try:
            from torch.backends.cuda import can_use_flash_attention  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def fused_attention(
    query,
    key,
    value,
    scale,
    attention_mask=None,
    soft_cap=None,
    dropout=None,
    training=None,
    attn_implementation=None,
):
    """Scaled dot-product attention with a selectable implementation.

    Computes ``softmax(soft_cap(QKᵀ · scale) + mask) V``. The implementation is
    chosen by ``attn_implementation`` (falling back to the implementation active
    for the current build/forward, which ``Model.from_weights`` sets per model):

    * ``"sdpa"`` -- hand-written matmul/softmax math. Portable across every
      backend, dtype and device. This is the default.
    * ``"fused"`` -- :func:`keras.ops.dot_product_attention` with the backend's
      own kernel selection: torch's ``scaled_dot_product_attention`` (flash /
      memory-efficient / math, so the logits are never materialized on a GPU
      when a fused kernel applies; masks allowed), XLA on JAX. Falls back to
      the ``"sdpa"`` math on TensorFlow, with attention dropout or a logit
      soft-cap.
    * ``"flash"`` -- :func:`keras.ops.dot_product_attention` with
      ``flash_attention=True`` (the real flash kernel). Used only when the
      backend supports it and there is no attention dropout or logit soft-cap;
      otherwise it transparently falls back to the ``"sdpa"`` math (and the
      flash op itself raises if the GPU/dtype cannot support flash).

    All tensors are ``(batch, num_heads, seq, head_dim)`` with the key/value
    heads already repeated to ``num_heads`` (GQA expansion is the caller's
    responsibility). The result is returned in the same layout.

    Args:
        query: ``(batch, num_heads, q_len, head_dim)``.
        key: ``(batch, num_heads, kv_len, head_dim)``.
        value: ``(batch, num_heads, kv_len, head_dim)``.
        scale: Query/key scaling factor (e.g. ``head_dim**-0.5``).
        attention_mask: Additive mask broadcastable to
            ``(batch, num_heads, q_len, kv_len)``, or ``None``.
        soft_cap: Optional tanh logit soft-cap value (e.g. Gemma's ``50.0``);
            ``None`` disables it. Forces the ``"sdpa"`` path.
        dropout: Optional ``keras.layers.Dropout`` applied to the attention
            probabilities. Only active during training with a positive rate, in
            which case the ``"sdpa"`` path is used so it can be applied.
        training: Whether the call is in training mode.
        attn_implementation: ``"sdpa"`` / ``"fused"`` / ``"flash"`` / ``None``
            (use the implementation active in the current context, else
            ``"sdpa"``).

    Returns:
        ``(batch, num_heads, q_len, head_dim)``.
    """
    impl = (
        attn_implementation or _ATTN_IMPLEMENTATION.get() or DEFAULT_ATTN_IMPLEMENTATION
    )
    if impl not in VALID_ATTN_IMPL:
        raise ValueError(
            f"attn_implementation must be one of {VALID_ATTN_IMPL}, got {impl!r}"
        )

    use_dropout = (
        bool(training) and dropout is not None and getattr(dropout, "rate", 0.0) > 0.0
    )
    use_fused_op = (
        impl in ("fused", "flash")
        and _fused_op_available()
        and not use_dropout
        and soft_cap is None
    )
    # Match the additive mask to the compute dtype: torch / jax auto-promote a
    # float32 mask against bf16 logits, but tensorflow raises on the mismatch.
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, query.dtype)
    if use_fused_op:
        q = ops.transpose(query, (0, 2, 1, 3))
        k = ops.transpose(key, (0, 2, 1, 3))
        v = ops.transpose(value, (0, 2, 1, 3))
        out = ops.dot_product_attention(
            q,
            k,
            v,
            bias=attention_mask,
            scale=scale,
            flash_attention=True if impl == "flash" else None,
        )
        return ops.transpose(out, (0, 2, 1, 3))

    logits = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * scale
    if soft_cap is not None:
        logits = soft_cap * ops.tanh(logits / soft_cap)
    if attention_mask is not None:
        logits = logits + attention_mask
    probs = ops.cast(ops.softmax(ops.cast(logits, "float32"), axis=-1), query.dtype)
    if use_dropout:
        probs = dropout(probs, training=True)
    return ops.matmul(probs, value)
