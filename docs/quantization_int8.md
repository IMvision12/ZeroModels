# int8

Per-channel symmetric int8, roughly **4x smaller**. The default scheme, and the only one
besides int4 that runs on all three backends.

## Usage

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate
from zeromodels.quantization import quantize_model

# load and quantize in one call
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int8")

# or quantize a model you already have, in place
quantize_model(model, "int8")
```

`quantize_model` swaps every built `Dense`, `EinsumDense`, `Embedding`, and fused MoE expert
bank for its quantized counterpart and frees the float weights. Functional models cannot be
mutated in place, so those are **cloned**: use the returned model.

```python
qmodel = quantize_model(vit_model, "int8")  # functional -> returns a NEW model
```

## Int8Config

```python
Int8Config(skip_modules=("lm_head",), quantize_embeddings=True, overrides=None)
```

The declarative recipe for int8, used when the bare `"int8"` string is not enough control.
It is `QuantizationConfig` with `mode="int8"` fixed and `group_size` dropped, since block
size means nothing for a per-channel scheme.

**Parameters**

- **skip_modules** (`tuple` of `str`, *optional*, defaults to `("lm_head",)`): name substrings; any layer whose path contains one is left in float. The output head is the most accuracy-sensitive layer in a decoder, so it is skipped by default.
- **quantize_embeddings** (`bool`, *optional*, defaults to `True`): quantize `Embedding` layers. Set `False` to keep the token table in float.
- **overrides** (`dict`, *optional*): `{name_substring: mode}`, per-layer precision, checked **before** `mode`. Use it to raise or lower precision on specific layers.

Resolution order for any layer is `skip_modules` first, then `overrides`, then the mode. A
layer matched by `skip_modules` stays float even if `overrides` also names it.

## Using the config

Pass it anywhere a scheme string is accepted, `from_weights` included:

```python
from zeromodels.quantization import Int8Config, quantize_model

cfg = Int8Config(skip_modules=("lm_head", "q_proj"))

model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization=cfg)
quantize_model(model, cfg)
```

Keeping an extra layer in float, when one measures badly:

```python
Int8Config(skip_modules=("lm_head", "q_proj"))
```

Keeping the token embedding table in float, worth trying when a model with a large
vocabulary loses quality:

```python
Int8Config(quantize_embeddings=False)
```

Mixing precisions, here dropping most of the model to int4 while holding the first decoder
block at int8:

```python
from zeromodels.quantization import QuantizationConfig

QuantizationConfig(mode="int4", group_size=128, overrides={"decoder_layer_0": "int8"})
```

The effect is visible on the layers themselves after quantizing:

```python
print(type(model.q_proj).__name__, type(model.lm_head).__name__)
```

```
QuantizedDense Dense
```

Embeddings are quantized int8 with a per-row scale. That holds in int4 mode too: the 4-bit
savings live in the Dense weights, so `QuantizedEmbedding` carries an `Int8Quantizer`
either way.

See [Quantization](quantization.md) for the shared machinery, and
[int4](quantization_int4.md) / [fp8](quantization_fp8.md) for the other schemes.
