import numpy as np

# keras var-name -> (HF module suffix, is a Dense kernel that needs transposing)
NORM_SUFFIX = {
    "gamma": ".weight",
    "beta": ".bias",
    "moving_mean": ".running_mean",
    "moving_variance": ".running_var",
}
# leaf add_weight()s whose name is itself a module segment in the HF key
LEAF_WEIGHTS = {"rel_pos_emb", "depthwise_conv"}


def to_numpy(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t).astype("float32")


def keras_path_to_hf(path):
    """Map a keras weight path to its ``encoder.*`` HF key + a transform tag.

    Transform tags: ``dense`` (transpose a 2-D Dense kernel), ``depthwise``
    (``(inner, 1, k) -> (k, inner)``) or ``direct`` (copy as-is)."""
    parts = path.split("/")
    for i, p in enumerate(parts):
        if p in ("input_linear", "out", "out_mid") or p.startswith("layers_"):
            parts = parts[i:]
            break
    var = parts[-1]
    body = [
        f"layers.{p.split('_', 1)[1]}" if p.startswith("layers_") else p
        for p in parts[:-1]
    ]
    module = ".".join(body)

    if var == "kernel":
        return f"encoder.{module}.weight", "dense"
    if var == "bias":
        return f"encoder.{module}.bias", "direct"
    if var in NORM_SUFFIX:
        return f"encoder.{module}{NORM_SUFFIX[var]}", "direct"
    if var == "depthwise_conv":
        return f"encoder.{module}.{var}.weight", "depthwise"
    if var in LEAF_WEIGHTS:  # rel_pos_emb
        return f"encoder.{module}.{var}.weight", "direct"
    raise KeyError(f"Unmapped keras weight: {path}")


def transfer_granite_speech5_weights(keras_model, state_dict):
    """Load an ``ibm-granite/granite-speech-5.0-*-turboctc`` state dict into the
    keras GraniteSpeech5 encoder / CTC model. The CTC head is tied to
    ``encoder.out`` (a single ``out`` layer in keras serves both the mid-layer
    self-conditioning and the final logits)."""
    missing = []
    for w in keras_model.weights:
        hf_key, kind = keras_path_to_hf(w.path)
        if hf_key not in state_dict:
            missing.append((w.path, hf_key))
            continue
        arr = to_numpy(state_dict[hf_key])
        if kind == "dense":
            arr = arr.T
        elif kind == "depthwise":
            arr = np.transpose(arr[:, 0, :], (1, 0))  # (inner, 1, k) -> (k, inner)
        if tuple(arr.shape) != tuple(w.shape):
            raise ValueError(
                f"Shape mismatch for {w.path} <- {hf_key}: {arr.shape} vs {tuple(w.shape)}"
            )
        w.assign(arr)

    if missing:
        raise KeyError(
            f"{len(missing)} keras weights had no matching HF key, e.g. {missing[:3]}"
        )
