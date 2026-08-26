# fp8

Per-channel `float8_e4m3fn`, one byte per weight: the **same footprint as
[int8](quantization_int8.md)** with a floating-point grid instead of a uniform one. Worth
choosing when the weights are heavy-tailed.

> **torch and jax only.** TensorFlow lacks the float8 casts needed to store the weights, so
> `"fp8"` raises a clear error there. Use `"int8"` for a tf-portable 4x option.

## Usage

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate
from zeromodels.quantization import quantize_model

# load and quantize in one call
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="fp8")

# or quantize a model you already have, in place
quantize_model(model, "fp8")
```

Check the backend before committing to it:

```python
from zeromodels.quantization.fp8_quantize import fp8_supported

print(fp8_supported())
```

```
True
```

## Fp8Config

```python
Fp8Config(skip_modules=("lm_head",), quantize_embeddings=True, overrides=None)
```

The declarative recipe for fp8. It is `QuantizationConfig` with `mode="fp8"` fixed and
`group_size` dropped, since fp8 is per-channel rather than block-wise, so its arguments are
identical to [`Int8Config`](quantization_int8.md#int8config).

**Parameters**

- **skip_modules** (`tuple` of `str`, *optional*, defaults to `("lm_head",)`): name substrings; any layer whose path contains one is left in float.
- **quantize_embeddings** (`bool`, *optional*, defaults to `True`): quantize `Embedding` layers. They are quantized **int8** regardless of the model mode.
- **overrides** (`dict`, *optional*): `{name_substring: mode}`, per-layer precision, checked **before** `mode`.

Resolution order for any layer is `skip_modules` first, then `overrides`, then the mode.

## Using the config

Pass it anywhere a scheme string is accepted, `from_weights` included:

```python
from zeromodels.quantization import Fp8Config, quantize_model

cfg = Fp8Config(skip_modules=("lm_head",))

model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization=cfg)
quantize_model(model, cfg)
```

Because fp8 and int8 are the same size, `overrides` here is about **grid shape**, not
footprint: send the layers whose weights have wide dynamic range to fp8 and leave the rest
uniform.

```python
from zeromodels.quantization import QuantizationConfig

QuantizationConfig(mode="int8", overrides={"attn": "fp8"})
```

The backend restriction follows whatever ends up fp8, so a config like the one above still
fails on TensorFlow even though its default mode is int8.

Since the two schemes cost the same bytes, the choice between them is worth **measuring per
model** rather than deciding by rule: fp8 tends to win on heavy-tailed weights and int8 on
well-behaved ones.

See [Quantization](quantization.md) for the shared machinery, and
[int8](quantization_int8.md) / [int4](quantization_int4.md) for the other schemes.
