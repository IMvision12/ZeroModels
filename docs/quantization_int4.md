# int4

Block-wise symmetric int4, packed two values per byte: roughly **8x smaller**. This is what
gets a checkpoint onto a GPU that int8 cannot.

## Usage

```python
from kerasformers.models.qwen3 import Qwen3TextGenerate
from kerasformers.quantization import quantize_model

# load and quantize in one call
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int4")

# a named scheme picks the block size
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int4-g128")

# or quantize a model you already have, in place
quantize_model(model, "int4")
quantize_model(model, "int4", group_size=64)
```

Three named schemes cover the common cases: `"int4"` (block size 32), `"int4-g64"`, and
`"int4-g128"`.

`quantization="int4"` builds the model at `load_dtype` (bf16) and quantizes it after, so
peak memory during the load is the bf16 model, then drops to the int4 size:

```python
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int4")
```

To load a checkpoint that does not fit in float at all, use a repo that ships already
quantized in a native packed format (e.g. GPT-OSS mxfp4), whose packed weights load without
a float intermediate.

## Int4Config

```python
Int4Config(
    group_size=32, skip_modules=("lm_head",), quantize_embeddings=True, overrides=None
)
```

The declarative recipe for int4. It is `QuantizationConfig` with `mode="int4"` fixed;
unlike the other two schemes it keeps `group_size`, which is the knob that matters here.

**Parameters**

- **group_size** (`int`, *optional*, defaults to `32`): block size along the contracting axis. Each block gets its own scale, so smaller means more scales: more accurate, less compression. If it does not divide the axis evenly, the largest divisor below it is used instead.
- **skip_modules** (`tuple` of `str`, *optional*, defaults to `("lm_head",)`): name substrings; any layer whose path contains one is left in float.
- **quantize_embeddings** (`bool`, *optional*, defaults to `True`): quantize `Embedding` layers. They are quantized **int8**, not int4, in either mode.
- **overrides** (`dict`, *optional*): `{name_substring: mode}`, per-layer precision, checked **before** `mode`.

Resolution order for any layer is `skip_modules` first, then `overrides`, then the mode.

## Using the config

Pass it anywhere a scheme string is accepted, `from_weights` included:

```python
from kerasformers.quantization import Int4Config, quantize_model

cfg = Int4Config(group_size=128)

model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization=cfg)
quantize_model(model, cfg)
```

### Choosing group_size

| `group_size` | Trade |
|---|---|
| 32 (default) | Most scales, most accurate, least compression. |
| 64 | Middle ground. |
| 128 | Fewest scales, best compression. Usually the better production choice. |

The accuracy cost of larger blocks is mild while the size difference compounds across a
whole model, which is why `int4-g128` is generally worth preferring over the default.

### Mixing precisions

int4 is a real accuracy step down from int8, so holding the sensitive layers higher is
common. Anything not matched keeps the int4 default:

```python
from kerasformers.quantization import QuantizationConfig

QuantizationConfig(mode="int4", group_size=128, overrides={"decoder_layer_0": "int8"})
```

The effect is visible on the layers themselves after quantizing:

```python
print(type(model.q_proj.quantizer).__name__, model.q_proj.quantizer.group_size)
print(type(model.embed_tokens.quantizer).__name__)
```

```
Int4Quantizer 128
Int8Quantizer
```

The embedding carrying an `Int8Quantizer` is expected: a 4-bit token table costs more
accuracy than it saves bytes, so the 4-bit savings are confined to the Dense weights.

See [Quantization](quantization.md) for the shared machinery, and
[int8](quantization_int8.md) / [fp8](quantization_fp8.md) for the other schemes.
