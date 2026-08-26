# mxfp4

OCP **MXFP4**: 4-bit floating point (e2m1) packed two values per `uint8`, with a shared
**e8m0** (power-of-two) scale per 32-value block. A floating-point 4-bit grid, ~8x smaller
than fp32 (~4x smaller than bf16): a weight-only `quantize_model` scheme you can apply to
any model whose kernels have 32-multiple contracting dims.

## Usage

Like [int8](quantization_int8.md) / [int4](quantization_int4.md) /
[fp8](quantization_fp8.md), `"mxfp4"` is a weight-only scheme: pass it to `from_weights`
or `quantize_model` and Dense / EinsumDense / fused-expert kernels are stored packed and
dequantized on the fly. Runs on **all three backends** (pure `keras.ops`).

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate
from zeromodels.quantization import quantize_model, Mxfp4Config

# load + quantize in one call
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="mxfp4")  # ~8x smaller

# or quantize a built model in place
quantize_model(model, "mxfp4")
quantize_model(model, Mxfp4Config(skip_modules=("lm_head",)))  # per-method config
```

**Constraint:** MXFP4 packs *fixed* 32-value blocks, so each quantized kernel's
**contracting dimension must be a multiple of 32** (true of essentially every
transformer width / head-dim). A kernel that violates it raises a clear error; exclude it
with a `skip_modules` pattern or use `int4` (which allows any even dim). Embeddings stay
int8, like int4 (the 4-bit savings live in the Dense weights).

Accuracy is comparable to int4 (~0.98 cosine): both are ~4-bit. MXFP4 uses a *float*
(e2m1) grid with a power-of-two block scale, so it shines on weights that were **trained**
in it and is a solid round-to-nearest option elsewhere. This is not calibrated PTQ (no
GPTQ / AWQ).

## Mxfp4Config

```python
Mxfp4Config(skip_modules=("lm_head",), quantize_embeddings=True, overrides=None)
```

The declarative recipe for mxfp4. It is `QuantizationConfig` with `mode="mxfp4"` fixed;
the 32-value block size is intrinsic to the format, so there is no `group_size` knob.

**Parameters**

- **skip_modules** (`tuple` of `str`, *optional*, defaults to `("lm_head",)`): name substrings; any layer whose path contains one is left in float.
- **quantize_embeddings** (`bool`, *optional*, defaults to `True`): quantize `Embedding` layers. They are quantized **int8**, not mxfp4: a 4-bit token table costs more accuracy than it saves bytes.
- **overrides** (`dict`, *optional*): `{name_substring: mode}`, per-layer precision, checked **before** `mode`.

Resolution order for any layer is `skip_modules` first, then `overrides`, then the mode.
Pass the config anywhere a scheme string is accepted, `from_weights` included:

```python
from zeromodels.quantization import Mxfp4Config, quantize_model

cfg = Mxfp4Config(skip_modules=("lm_head", "router"))

model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization=cfg)
quantize_model(model, cfg)
```

## Primitives

`zeromodels/quantization/mxfp4_quantize.py` holds the pure-`keras.ops`,
backend-agnostic pack / unpack and the `BaseQuantizer` that wraps them:

```python
from zeromodels.quantization import (
    MXFP4Quantizer,
    quantize_to_mxfp4,
    dequantize_mxfp4,
)

blocks, scales = quantize_to_mxfp4(w)  # float -> packed (uint8 blocks + e8m0 scales)
w_approx = dequantize_mxfp4(blocks, scales)  # packed -> float

q = MXFP4Quantizer()  # the quantize_model building block
packed, scale = q.quantize(w, axis=0)  # single packed axis (moved to the end)
w_approx = q.dequantize(packed, scale, axis=0, dtype="float32")
```

- **`dequantize_mxfp4(blocks, scales, dtype="float32")`**: nibble -> FP4 codebook, times
  `2^(e8m0 - 127)`. A bit-exact port of HF's `convert_moe_packed_tensors` (validated max
  \|Δ\| **0.0**), so packed weights decode to exactly the reference values.
- **`quantize_to_mxfp4(w)`**: the inverse. Picks the e8m0 block scale
  (`floor(log2(amax)) - 2`, OCP MXFP4) and rounds each value to the nearest FP4 grid
  point. A value-exact inverse on the lattice (round-trip **0.0**).
- **`MXFP4Quantizer`**: the `BaseQuantizer` used by `quantize_model` — `quantize` /
  `dequantize` / `storage_spec` along an arbitrary contracting axis, `uint8` kernel +
  `uint8` (e8m0) scale.

Both run on **every backend including CPU**, with no GPU triton kernel required.

## Footprint

4 bits per weight, so ~4x smaller than a bf16 copy and ~8x smaller than fp32. Weight-only,
so this is a memory win, not a speed one: the dequant runs each forward.

See [Quantization](quantization.md) for the shared int8 / int4 / fp8 / mxfp4 machinery,
and [int8](quantization_int8.md) / [int4](quantization_int4.md) / [fp8](quantization_fp8.md)
for the other schemes.
