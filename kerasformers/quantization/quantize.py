import contextlib
import inspect
import warnings

import keras
from keras import layers, ops

from .quant_config import resolve_config
from .quantized_layers import (
    QuantizedDense,
    QuantizedEinsumDense,
    QuantizedEmbedding,
    QuantizedExperts,
)


def quantize_model(model, config="int8", group_size=32):
    """In-place weight-only int8 / int4 / fp8 / mxfp4 quantization of a model.

    Walks the layer tree and swaps every eligible built ``keras.layers.Dense``
    for a :class:`QuantizedDense` and ``keras.layers.Embedding`` for a
    :class:`QuantizedEmbedding`, freeing the float weights. Activations stay
    float (weight-only), so it is backend-agnostic and needs no special kernels.

    Args:
        model: A built model (weights already loaded).
        config: A :class:`QuantizationConfig`, a bare mode (``"int8"`` /
            ``"int4"`` / ``"fp8"`` / ``"mxfp4"``), or a named scheme
            (``"int4-g128"``, ...). The config controls per-layer precision,
            skipped layers, and whether embeddings are quantized. ``"mxfp4"``
            (OCP 4-bit float, the format GPT-OSS ships) needs each quantized
            kernel's contracting dim to be a multiple of 32.
        group_size: int4 block size when ``config`` is a bare mode string.

    Returns:
        The model, quantized. Subclassed models are quantized in place;
        functional models return a NEW cloned model (use the return value). The
        resolved :class:`QuantizationConfig` is stored as
        ``model._quantization_config``.
    """
    config = resolve_config(config, group_size)
    if _is_functional(model):
        model = quantize_functional(model, config)
    else:
        _quantize_layer(model, config, "")
    model._quantization_config = config
    return model


def quantize_skeleton(model, config="int8", group_size=32):
    """Swap **unbuilt** ``Dense`` / ``Embedding`` for unbuilt quantized layers.

    Run this on a freshly constructed (not-yet-built) subclassed model, then do
    one forward: the layers build **integer** storage directly, so the full float
    model is never materialized (the transformers "replace on the meta device,
    quantize on load" idea). Load the quantized weights afterwards. This is the
    no-float path that lets a checkpoint larger than RAM be loaded quantized.
    """
    config = resolve_config(config, group_size)
    if _is_functional(model):
        raise ValueError(
            "quantize_skeleton is for unbuilt subclassed models; functional graphs "
            "build eagerly. Use quantize_model on the built functional model."
        )
    _skeleton_layer(model, config, "")
    model._quantization_config = config
    return model


def _skeleton_layer(layer, config, path):
    quant = (QuantizedDense, QuantizedEinsumDense, QuantizedEmbedding, QuantizedExperts)
    for name, value in list(_named_children(layer).items()):
        if name.startswith("_") or isinstance(value, quant):
            continue
        child_path = _child_path(path, value, name)
        if isinstance(value, layers.Dense) and not value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                _swap(
                    layer,
                    name,
                    value,
                    QuantizedDense(
                        value.units,
                        mode=mode,
                        use_bias=value.use_bias,
                        group_size=config.group_size,
                        name=value.name,
                    ),
                )
        elif isinstance(value, layers.Embedding) and not value.built:
            if config.quantize_embeddings and config.mode_for(child_path) is not None:
                _swap(
                    layer,
                    name,
                    value,
                    QuantizedEmbedding(
                        value.input_dim, value.output_dim, name=value.name
                    ),
                )
        elif _is_experts_skeleton(value) and not value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                _swap(
                    layer,
                    name,
                    value,
                    QuantizedExperts(
                        value.num_experts,
                        value.embed_dim,
                        value.mlp_dim,
                        mode=mode,
                        group_size=config.group_size,
                        activation=_expert_activation(value),
                        name=value.name,
                    ),
                )
        elif isinstance(value, layers.Layer):
            _skeleton_layer(value, config, child_path)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, layers.Layer):
                    _skeleton_layer(item, config, _child_path(path, item, name))


def _walk_layers(layer):
    yield layer
    for name, value in _named_children(layer).items():
        if name.startswith("_"):
            continue
        if isinstance(value, layers.Layer):
            yield from _walk_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, layers.Layer):
                    yield from _walk_layers(item)


@contextlib.contextmanager
def _floats_loading(model):
    touched = [
        layer
        for layer in _walk_layers(model)
        if isinstance(layer, (QuantizedDense, QuantizedEmbedding, QuantizedExperts))
    ]
    for layer in touched:
        layer._loading = True
    try:
        yield
    finally:
        for layer in touched:
            layer._loading = False


def quantize_and_load(model, config, transfer_fn, state_dict, group_size=32):
    """No-float quantized load: stream a float checkpoint straight into int storage.

    Builds an integer skeleton (:func:`quantize_skeleton`) on the **unbuilt**
    subclassed ``model``, then runs the model's own ``transfer_fn``, but with the
    quantized layers surfacing float proxies, so each source tensor is quantized
    into int storage as it is assigned and the full float model is never built.
    Dense and Embedding weights are quantized; layers the skeleton leaves float
    (e.g. ``EinsumDense`` / MoE experts, the skipped head) load as float.

    Args:
        model: A freshly constructed (unbuilt) subclassed model.
        config: A :class:`QuantizationConfig`, mode, or scheme.
        transfer_fn: The model's weight-transfer callable
            ``(model, state_dict) -> None`` (e.g. ``cls.transfer_from_hf``). It
            must build the model if unbuilt and iterate ``model.weights`` (the
            standard kerasformers converter pattern).
        state_dict: The flat ``{name: array}`` source checkpoint.
        group_size: int4 block size when ``config`` is a bare mode string.

    Returns:
        The model, quantized in place, with ``model._quantization_config`` set.
    """
    config = resolve_config(config, group_size)
    quantize_skeleton(model, config, group_size)
    with _floats_loading(model):
        transfer_fn(model, state_dict)
    unfilled = [
        layer.name
        for layer in _walk_layers(model)
        if isinstance(layer, (QuantizedDense, QuantizedEmbedding, QuantizedExperts))
        and layer.built
        and not getattr(layer, "_loaded", False)
    ]
    if unfilled:
        raise RuntimeError(
            "No-float quantized load left quantized layers unfilled "
            f"({unfilled[:5]}{'...' if len(unfilled) > 5 else ''}): this model's "
            "converter does not assign all weights through model.weights, so the "
            "float proxies for these layers were never written. A subclassed model "
            "must build and iterate model.weights in its converter to be loadable "
            "in quantized form."
        )
    model._quantization_config = config
    return model


def dequantize_model(model):
    """Revert a quantized model back to float ``Dense`` / ``Embedding`` layers.

    Subclassed models are reverted in place; functional models return a NEW
    cloned model. (Quantized MoE experts in subclassed models stay quantized:
    they still run correctly via ``QuantizedExperts``.)
    """
    if _is_functional(model):
        return _dequantize_functional(model)
    _dequantize_layer(model)
    model._quantization_config = None
    return model


def _named_children(layer):
    items = dict(vars(layer))
    modules = getattr(layer, "_modules", None)
    if modules:
        for k, v in modules.items():
            items.setdefault(k, v)
    return items


def _child_path(path, value, name):
    leaf = getattr(value, "name", None) or name
    return f"{path}/{leaf}" if path else leaf


def _is_fused_experts(layer):
    return (
        hasattr(layer, "gate_up_proj")
        and hasattr(layer, "down_proj")
        and hasattr(layer, "num_experts")
        and getattr(layer, "built", False)
    )


def _is_experts_skeleton(layer):
    # An *unbuilt* fused-experts bank: gate_up_proj / down_proj don't exist yet,
    # so detect by the routed-experts call signature. Excludes the MoE wrapper
    # (its call is ``call(hidden_states)``, no ``routing_weights``) and an
    # already-quantized bank.
    return (
        not isinstance(layer, QuantizedExperts)
        and hasattr(layer, "num_experts")
        and hasattr(layer, "embed_dim")
        and hasattr(layer, "mlp_dim")
        and "routing_weights" in inspect.signature(layer.call).parameters
    )


def _expert_activation(layer):
    return "gelu" if "gemma" in type(layer).__name__.lower() else "silu"


def _swap(parent, name, old, new):
    # Tracker locks state after build; unlock to drop the old layer and register
    # the new one, then restore the PRIOR lock state. Preserving it matters for
    # the skeleton path: an unbuilt parent (e.g. an MoE block that adds its own
    # router weight in build()) is still unlocked, and re-locking here would break
    # its later build(). A built parent is re-locked, as keras expects.
    locked = parent._tracker.locked
    parent._tracker.unlock()
    parent._tracker.untrack(old)
    setattr(parent, name, new)
    if locked:
        parent._tracker.lock()


@contextlib.contextmanager
def _build_scope(target_path):
    # Recreate the name-scope stack so a layer built OUTSIDE a forward pass (the
    # in-place swap) gets the SAME weight paths it would get if built during its
    # parent's call. Without this, swapped layers lose their parent prefix: all
    # `block_*/q` collapse to bare `q`, which collides in the sharded
    # `.weights.json` format and desyncs the saved model from a skeleton reload
    # (the skeleton builds during a forward, so it keeps the full path).
    parent = target_path.rsplit("/", 1)[0] if target_path and "/" in target_path else ""
    if not parent:
        yield
        return
    from keras.src.backend.common.name_scope import name_scope

    grandparent, leaf = parent.rsplit("/", 1) if "/" in parent else ("", parent)
    with name_scope(leaf, override_parent=grandparent or None):
        yield


def _swap_to_quantized_dense(parent, name, dense, mode, group_size):
    # Attach the (unbuilt) quantized layer, THEN build it within the original
    # layer's name scope, so its weights take the full graph path
    # (parent/.../name/kernel), matching the skeleton path and keeping save/load
    # consistent. (Building standalone would give bare, collision-prone paths.)
    in_dim = int(dense.kernel.shape[0])
    target_path = dense.path
    quantized = QuantizedDense(
        dense.units,
        mode=mode,
        use_bias=dense.use_bias,
        group_size=group_size,
        name=dense.name,
    )
    _swap(parent, name, dense, quantized)
    with _build_scope(target_path):
        quantized.build((None, in_dim))
    q, scale = quantized.quantizer.quantize(dense.kernel, axis=0)
    quantized.kernel.assign(q)
    quantized.scale.assign(scale)
    if dense.use_bias:
        quantized.bias.assign(dense.bias)


def _swap_to_quantized_embedding(parent, name, embedding):
    target_path = embedding.path
    quantized = QuantizedEmbedding(
        embedding.input_dim, embedding.output_dim, name=embedding.name
    )
    _swap(parent, name, embedding, quantized)
    with _build_scope(target_path):
        quantized.build()
    q, scale = quantized.quantizer.quantize(embedding.embeddings, axis=1)
    quantized.quantized_embeddings.assign(q)
    quantized.scale.assign(scale)


def _quantize_layer(layer, config, path):
    for name, value in list(_named_children(layer).items()):
        if name.startswith("_"):
            continue
        if isinstance(
            value,
            (
                QuantizedDense,
                QuantizedEinsumDense,
                QuantizedEmbedding,
                QuantizedExperts,
            ),
        ):
            continue
        child_path = _child_path(path, value, name)
        if isinstance(value, layers.Dense) and value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                _swap_to_quantized_dense(layer, name, value, mode, config.group_size)
        elif isinstance(value, layers.EinsumDense) and value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                with _build_scope(value.path):
                    quantized = QuantizedEinsumDense.from_einsum_dense(
                        value, mode, config.group_size
                    )
                _swap(layer, name, value, quantized)
        elif isinstance(value, layers.Embedding) and value.built:
            if config.quantize_embeddings and config.mode_for(child_path) is not None:
                _swap_to_quantized_embedding(layer, name, value)
        elif _is_fused_experts(value):
            mode = config.mode_for(child_path)
            if mode is not None:
                with _build_scope(value.path):
                    quantized = QuantizedExperts.from_experts(
                        value, mode, config.group_size, _expert_activation(value)
                    )
                _swap(layer, name, value, quantized)
        elif isinstance(value, layers.Layer):
            _quantize_layer(value, config, child_path)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (layers.Dense, layers.Embedding)) and item.built:
                    warnings.warn(
                        f"Not quantizing {type(item).__name__} '{item.name}' held "
                        f"in list '{name}' (list-element swap unsupported); store "
                        f"it as a direct attribute to quantize it.",
                        stacklevel=2,
                    )
                elif isinstance(item, layers.Layer):
                    _quantize_layer(item, config, _child_path(path, item, name))


def _dequantize_layer(layer):
    for name, value in list(_named_children(layer).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, QuantizedDense):
            _swap(layer, name, value, value.to_dense())
        elif isinstance(value, QuantizedEmbedding):
            _swap(layer, name, value, value.to_embedding())
        elif isinstance(value, layers.Layer):
            _dequantize_layer(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, layers.Layer):
                    _dequantize_layer(item)


def _is_functional(model):
    try:
        from kerasformers.base import BaseModel

        if isinstance(model, BaseModel):
            return True
    except Exception:
        pass
    try:
        from keras.src.models.functional import Functional

        return isinstance(model, Functional)
    except Exception:
        return False


def _transfer_unquantized(src, dst):
    # Copy float weights for layers that clone_fn left unquantized. Skip any
    # layer that is quantized on either side (its weights are baked by clone_fn).
    quant = (
        QuantizedDense,
        QuantizedEinsumDense,
        QuantizedEmbedding,
        QuantizedExperts,
    )
    for layer in dst.layers:
        if not layer.weights or isinstance(layer, quant):
            continue
        if getattr(layer, "_quantization_config", None) is not None:
            continue  # a recursively-quantized sub-model: weights already baked
        try:
            orig = src.get_layer(layer.name)
        except ValueError:
            continue
        if isinstance(orig, quant) or not orig.weights:
            continue
        layer.set_weights([ops.convert_to_numpy(w) for w in orig.weights])


def quantize_functional(model, config="int8", group_size=32):
    """Quantize a functional model by cloning its graph with quantized layers.

    In-place swap can't rewire a functional graph, so this rebuilds it via
    ``keras.models.clone_model``, replacing eligible ``Dense`` / ``Embedding``
    with their quantized variants (weights baked in) and copying the remaining
    float weights. Returns a NEW model.
    """
    config = resolve_config(config, group_size)

    def clone_fn(layer):
        if _is_functional(layer):
            # Nested Functional sub-model (e.g. encoder / decoder): recurse so its
            # own graph (and its nested blocks) get quantized too. If it can't be
            # cloned (e.g. a weight-capturing `Lambda` lm_head), keep it float.
            try:
                return quantize_functional(layer, config)
            except Exception:
                return layer
        if isinstance(layer, layers.Dense) and layer.built:
            mode = config.mode_for(layer.name)
            if mode is not None:
                return QuantizedDense.from_dense(layer, mode, config.group_size)
        elif isinstance(layer, layers.EinsumDense) and layer.built:
            mode = config.mode_for(layer.name)
            if mode is not None:
                return QuantizedEinsumDense.from_einsum_dense(
                    layer, mode, config.group_size
                )
        elif isinstance(layer, layers.Embedding) and layer.built:
            if config.quantize_embeddings and config.mode_for(layer.name) is not None:
                return QuantizedEmbedding.from_embedding(layer)
        try:
            return layer.__class__.from_config(layer.get_config())
        except Exception:
            # Un-cloneable layer (e.g. a python-lambda Lambda for an activation /
            # scale / mask): reuse the original instance. These are stateless ops,
            # so sharing them with the source graph is safe.
            return layer

    try:
        clone = keras.models.clone_model(model, clone_function=clone_fn)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not quantize functional model '{model.name}': cloning its graph "
            f"failed ({type(e).__name__}: {str(e)[:160]})."
        ) from e
    _transfer_unquantized(model, clone)
    # clone_fn only swaps top-level graph nodes; quantize Dense / EinsumDense /
    # Embedding / experts nested inside custom block layers (attention, MLP, ...)
    # via the in-place swap, so functional transformers are fully covered.
    nested = (
        QuantizedDense,
        QuantizedEinsumDense,
        QuantizedEmbedding,
        QuantizedExperts,
    )
    for layer in clone.layers:
        if isinstance(layer, nested):
            continue
        if getattr(layer, "_quantization_config", None) is not None:
            continue  # already quantized as a nested sub-model
        _quantize_layer(layer, config, layer.name)
    clone._quantization_config = config
    return clone


def _dequantize_functional(model):
    def clone_fn(layer):
        if isinstance(layer, QuantizedDense):
            return layer.to_dense()
        if isinstance(layer, QuantizedEmbedding):
            return layer.to_embedding()
        return layer.__class__.from_config(layer.get_config())

    clone = keras.models.clone_model(model, clone_function=clone_fn)
    _transfer_unquantized(model, clone)
    clone._quantization_config = None
    return clone
