# Quantization (int8 / int4 / fp8 / mxfp4)

zeromodels ships its **own** weight-only int8 / int4 / fp8 / mxfp4 quantization in
`zeromodels/quantization/`: a from-scratch, backend-agnostic implementation
(pure `keras.ops`), not Keras's built-in `model.quantize`. It shrinks a model
~4× (int8 / fp8) or ~8× (int4 / mxfp4) so larger checkpoints fit in memory. int8 /
int4 / mxfp4 run on TensorFlow / Torch / JAX; fp8 (float8-e4m3) is torch / jax only.

## Choosing a scheme

| Scheme | Size | Cosine | Backends | Pick it when |
|---|---|---|---|---|
| [**int8**](quantization_int8.md) | ~3.8× smaller | ~0.9999 | all three | The default. Near-free accuracy, use it whenever it fits. |
| [**int4**](quantization_int4.md) | ~5.8–8× smaller | ~0.98 | all three | int8 does not fit. Block-wise, `group_size` is the knob. |
| [**fp8**](quantization_fp8.md) | ~3.8× smaller | ~0.9994 | torch / jax | Same size as int8, better on heavy-tailed weights. Measure both. |
| [**mxfp4**](quantization_mxfp4.md) | ~8× smaller | ~0.98 | all three | 4-bit *float* (OCP e2m1). The format GPT-OSS ships; contracting dims must be multiples of 32. |

Each page covers that scheme's math, storage layout, measured accuracy, and its own config
class. The rest of this page is the machinery they share.

> **[MXFP4](quantization_mxfp4.md)** does double duty: it is both a general
> `quantize_model(model, "mxfp4")` scheme (applied to any model whose kernels have
> 32-multiple contracting dims) **and** the native on-disk format of GPT-OSS, whose
> experts are shipped packed and dequantized on the fly with the same primitives.

## Quick start

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate

# load + quantize in one call
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int8")  # ~4x smaller
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int4")  # ~8x smaller
model = Qwen3TextGenerate.from_weights(
    "qwen3-4b", quantization="fp8"
)  # ~4x (torch/jax)

# or quantize a model you already built/loaded
from zeromodels.quantization import quantize_model

quantize_model(model, "int8")  # in place
quantize_model(model, "int4", group_size=64)  # int4 block size (default 32)
quantize_model(model, "fp8")  # float8 e4m3 (torch / jax)
quantize_model(model, "mxfp4")  # OCP 4-bit float, fixed 32-value blocks
```

`quantization=` is wired through `from_weights` for every model (Hub Keras repos,
bare LLM/VLM variants, **and** `hf:` repos).

**Memory during a quantized load.** `quantization=` builds the model at `load_dtype`
(bf16 by default) and swaps in quantized layers after, freeing the floats. Peak memory
is therefore about the **bf16** model, then drops to the quantized size:

```python
model = Qwen3TextGenerate.from_weights("qwen3-4b", quantization="int4")
```

To load a checkpoint that does not fit in float at all, use a repo that ships **already
quantized** in a native packed format (e.g. GPT-OSS's mxfp4), whose packed weights load
directly without a float intermediate (see [`KfQuantizer`](#loading-a-pre-quantized-repo-kfquantizer)).

## Production usage

Pass a `QuantizationConfig` for fine control: named schemes, mixed precision,
and skipping accuracy-sensitive layers:

```python
from zeromodels.quantization import quantize_model, QuantizationConfig

cfg = QuantizationConfig(
    mode="int4",
    group_size=128,
    skip_modules=("lm_head",),  # keep these layers in float
    quantize_embeddings=True,
    overrides={"decoder_layer_0": "int8"},  # per-layer precision
)
quantize_model(model, cfg)
quantize_model(model, "int4-g128")  # or a named scheme
```

**Save / load / revert.** A quantized model saves and reloads itself quantized through
ordinary Keras save (the quantization is carried in `get_config` and re-applied in
`from_config`, see [`KfQuantizer`](#loading-a-pre-quantized-repo-kfquantizer)):

```python
import keras
from zeromodels.quantization import dequantize_model, get_kf_quantizer

# Full save: the quantization round-trips automatically.
model.save("model.keras")
model = keras.saving.load_model("model.keras")  # rebuilt quantized, weights loaded

# Weights-only (.weights.h5) carries values, not structure, so the target must already
# be quantized before load_weights. From a Hub repo that is automatic (kf_config's
# quantization_config drives it):
model = Qwen3TextGenerate.from_weights("zeromodels/qwen3-4b-int8")

# Into a hand-built model, apply the quantizer first, then load_weights. For a
# functional model preprocess_model returns a NEW (cloned) quantized model, so use it:
model.save_weights("model.weights.h5")
skeleton = Qwen3TextGenerate.from_weights("qwen3-4b", load_weights=False)
skeleton = get_kf_quantizer({"quant_method": "int8"}).preprocess_model(skeleton)
skeleton.load_weights("model.weights.h5")

dequantize_model(model)  # revert to float layers
```

**MoE and functional models:** `quantize_model` also quantizes **fused MoE
experts** (the `gate_up_proj` / `down_proj` banks of Qwen/GLM/DeepSeek-MoE, along
the contracted axis) and **functional / vision** models (ViT, CLIP, …). A
functional graph can't be mutated in place, so it is **cloned**: use the
returned model:

```python
qmodel = quantize_model(vit_model, "int8")  # functional -> returns a NEW model
```

## Loading a pre-quantized repo (`KfQuantizer`)

Everything above *applies* quantization to a float model. A repo can also **ship
already quantized** and declare it in `kf_config.json` with a transformers-style
block:

```json
"quantization_config": { "quant_method": "mxfp4" }
```

`from_weights` reads that block and **auto-applies** the matching quantizer, so a
quantized repo loads with **no flag**:

```python
# reads quantization_config -> loads bf16 dense + mxfp4 experts, by default
model = GptOssTextGenerate.from_weights("zeromodels/gpt-oss-20b")
```

The models stay **quantization-agnostic** (no per-model flags): the model builds the
plain float architecture, and a `KfQuantizer` swaps in the quantized layers **before
the weights load**, exactly like transformers'
`HfQuantizer._process_model_before_weight_loading`. `KfQuantizer` is a **second
level** above the tensor-level `BaseQuantizer`:

| level | class | job |
|---|---|---|
| tensor | `BaseQuantizer` (`Int8Quantizer`, `MXFP4Quantizer`, …) | quantize / dequantize one weight along an axis |
| model | `KfQuantizer` (transformers' `HfQuantizer` analog) | read `quantization_config`, swap modules before load |

`get_kf_quantizer(block)` dispatches on `quant_method`:

- `mxfp4` -> `Mxfp4KfQuantizer` (GPT-OSS native: swaps the float `GptOssExperts` for
  the packed `GptOssMXFP4Experts`).
- `int8` / `int4` / `fp8` -> `WeightOnlyKfQuantizer` (generic: swaps `Dense` /
  `Embedding` for their quantized layers via `quantize_model`, which clones a
  functional model, so `preprocess_model` returns the prepared model to load into).

**Save round-trips itself.** The applied quantization is stamped on the model
(`model._quantization_config`) and carried through `get_config` / `from_config`, so a
quantized model **saves and reloads itself quantized** via an ordinary Keras save,
no export step and no re-quantization:

```python
model.save("m.keras")
reloaded = keras.saving.load_model("m.keras")  # rebuilds bf16 + mxfp4 automatically
```

Functional models round-trip natively: Keras serializes each layer, quantized ones
included, on its own, and the recorded `quantization_config` rides along in `get_config` /
`from_config` so `from_config` re-applies the quantizer before the weights load.

## How it works

Weight-only quantization: the weights are stored quantized and **dequantized on
the fly** inside each layer's `call`, so the matmul still runs in the activation
dtype. No special int kernels are needed, which is why it is fully
backend-agnostic.

- **[int8](quantization_int8.md)**: per-channel symmetric absmax, one float scale
  per output channel over the contracting axis, `scale = max|w| / 127`.
- **[int4](quantization_int4.md)**: block-wise symmetric absmax, `scale = max|w| / 7`
  per block of `group_size`, packed two values per byte.
- **[fp8](quantization_fp8.md)**: per-channel absmax cast into the native
  `float8_e4m3fn` dtype, `scale = max|w| / 448`. torch / jax only.
- **[mxfp4](quantization_mxfp4.md)**: OCP 4-bit float (e2m1) in fixed 32-value
  blocks, each with a shared e8m0 (power-of-two) `uint8` scale; values pack two
  per byte. Same ~8× as int4, but a *float* grid and a `uint8` (not fp32) scale.
- **Embeddings**: int8 with a per-row scale; the lookup gathers int8 rows and
  dequantizes only the gathered slice (for the `int4` and `mxfp4` model modes,
  embeddings stay int8: the 4-bit savings live in the Dense weights).

**N-D kernels.** Quantization is along the **contracting axis**, not a hardcoded
`axis=0`, so the same quantizers serve 2-D `Dense` kernels, N-D `EinsumDense`
kernels (axis derived from the equation: a tuple for int8/fp8, a single axis for
packed int4 / mxfp4), per-row embeddings (`axis=1`), and fused MoE expert banks
(`axis=-1`). Scales keep the reduced axes as size 1 so they broadcast over any
rank with no reshape.

**Robustness.** Scales use an epsilon floor (`max(amax / MAX, ε)`) rather than an
exact-zero test (handles zero and denormal channels), and `dequantize` takes the
compute `dtype`, so `mixed_bfloat16` graphs don't upcast through float32.

`quantize_model` walks the layer tree and **swaps** every built `Dense` →
[`QuantizedDense`](https://github.com/IMvision12/ZeroModels/blob/main/zeromodels/quantization/quantized_layers.py), `EinsumDense`
→ `QuantizedEinsumDense`, `Embedding` → `QuantizedEmbedding`, and fused experts →
`QuantizedExperts`, freeing the float weights, then records the resolved
`QuantizationConfig` on the model. The swap unlocks the keras layer tracker,
untracks the float layer, and registers the quantized one, enumerating both
`__dict__` and (on the torch backend, where keras `Layer` is an `nn.Module`)
`_modules`, so it finds sub-layers on every backend.

## Components

The package mirrors keras's `Quantizer` / `AbsMaxQuantizer` structure: a base
class plus one file per scheme:

| Symbol | File | Role |
|---|---|---|
| `BaseQuantizer` | `base/base_quantization.py` | base class (also `zeromodels.base.BaseQuantizer`): `quantize(weight, axis)` / `dequantize(packed, scale, axis, dtype)` / `storage_spec(weight_shape, axis)` + `get_config` / `from_config`; ships `normalize_axes` / `single_axis` |
| `Int8Quantizer` | `int8_quantize.py` | per-channel int8 quantizer (quantize / dequantize methods) |
| `Int4Quantizer` | `int4_quantize.py` | block-wise packed int4 quantizer (any axis via moveaxis; module `effective_group_size`) |
| `Fp8Quantizer` | `fp8_quantize.py` | per-channel float8-e4m3 quantizer (module `fp8_supported`; torch / jax) |
| `MXFP4Quantizer` | `mxfp4_quantize.py` | OCP MXFP4 (e2m1) quantizer, single packed axis, `uint8` e8m0 scale (also `quantize_to_mxfp4` / `dequantize_mxfp4` pack / unpack) |
| `QuantizedDense` / `QuantizedEinsumDense` / `QuantizedEmbedding` / `QuantizedExperts` | `quantized_layers.py` | weight-only drop-in layers (each holds a quantizer); `QuantizedExperts` = fused MoE expert bank, contracting-axis quantized |
| `GptOssMXFP4Experts` | `quantized_layers.py` | GPT-OSS MoE expert bank kept in MXFP4 (native on-disk format), dequantized in `call` (top-k sparse on decode) |
| `QuantizationConfig` / `Int8Config` / `Int4Config` / `Fp8Config` / `Mxfp4Config` / `SCHEMES` | `quant_config.py` | recipe (mode, group_size, skip_modules, quantize_embeddings, overrides) + per-method configs + named presets |
| `quantize_model` / `quantize_functional` | `quantize.py` | model surgery: clone (functional models) / in-place (non-functional) |
| `quantize_skeleton` / `quantize_and_load` | `quantize.py` | legacy no-float path for **unbuilt** (non-functional) models; unused now that every model is functional |
| `KfQuantizer` / `Mxfp4KfQuantizer` / `WeightOnlyKfQuantizer` / `get_kf_quantizer` | `kf_quantizer.py` | model-level quantizers (transformers `HfQuantizer` analog): read a repo's `quantization_config` and swap in packed / int layers before load; dispatched by `quant_method` |
| `dequantize_model` | `quantize.py` | revert quantized layers back to float |

A `QuantizedDense` holds an `Int8Quantizer` / `Int4Quantizer` / `Fp8Quantizer` /
`MXFP4Quantizer` (via `get_quantizer(mode, group_size)`) and uses it for
`storage_spec` (build), `quantize` (from a float `Dense`), and `dequantize` (in
`call`).

## Will it fit? (memory sizing)

Weight-only quantization is about **fitting** a model, so the practical question
is bytes-per-parameter:

| precision | bytes / param | ~max params in 80 GB (weights only) |
|---|---|---|
| bf16 (float) | 2.0 | ~40B |
| int8 | ~1.0 | ~80B |
| int4 (g128) | ~0.55 | ~145B |

int4 adds the per-block fp32 scales (a few percent; a smaller `group_size` means
more scales, slightly larger). Leave **~20 % headroom** for the KV cache and
activations, so the *practical* ceilings on one 80 GB H100 are roughly **32B
bf16 / 64B int8 / ~115B int4**.

**MoE counts total, not active.** Sparse experts cut *compute* per token, but
every expert must be resident: size by total parameters, not active ones.

Worked examples (int4, ≈ 0.55 B/param):

| model | int4 weights | single 80 GB H100? |
|---|---|---|
| 70B dense | ~38 GB | yes |
| 120B (GPT-OSS-120B class) | ~66 GB | yes (tight) |
| 355B (GLM-4.5) | ~195 GB | no: ~3 GPUs |
| 744B (GLM-5.x) | ~410 GB | no: ~5–6 GPUs |

> **Load time.** `quantization=` builds the model at `load_dtype` (bf16) and quantizes
> after, so peak ≈ the **bf16** model (params × 2), then drops to the quantized size. A
> checkpoint that does not fit in float must therefore ship already quantized in a native
> packed format (e.g. GPT-OSS mxfp4), which loads packed without a float intermediate.

## Caveats (honest)

- **Portable weight-only = memory, not speed.** The default Keras path
  dequantizes weights to float every `call`, so it reduces footprint rather than
  latency.
- **Build-then-quantize.** `quantization=` builds the float architecture (at
  `load_dtype`) and swaps in quantized layers after, freeing the floats. Peak memory is
  the float model, not the quantized size, so a checkpoint larger than your float budget
  must ship already quantized in a native packed format (mxfp4), which loads packed
  without a float intermediate. Loading a repo whose `kf_config.json` declares a
  weight-only `quantization_config` follows the same build-then-quantize path (the
  `KfQuantizer` swaps in quantized layers before the packed weights load).
- **Coverage.** `Dense`, `EinsumDense`, `Embedding`, and fused-SwiGLU MoE expert
  banks (`gate_up_proj`/`down_proj`) are quantized; other custom weight layouts
  stay float. A `Dense`/`Embedding` stored inside a Python list (rare:
  zeromodels uses attributes) is skipped with a warning. `dequantize_model`
  reverts `Dense`/`Embedding`; quantized `EinsumDense` / experts stay quantized
  (they still run correctly). Tied-output LLMs that read `token_embedding.embeddings`
  for the logit projection keep working: `QuantizedEmbedding` exposes a
  dequantizing `embeddings` property.
- **Functional models are fully covered**, including Denses nested in custom
  blocks and nested `Functional` sub-models (encoder/decoder): after cloning the
  graph, the in-place swap descends into each block and recurses into sub-models.
  Functional **encoder-decoder ASR** (Whisper / Speech2Text / Moonshine) is the
  exception: it's *partially* quantized (cloneable parts like the encoder), but
  the decoder's weight-capturing `Lambda` lm_head can't be cloned so it stays
  float, and `clone_model` returns a plain `Functional` (dropping cached-
  generation methods), so quantized ASR is forward-only, not for `generate()`.
- **fp8 is torch / jax only.** TensorFlow lacks the float8 casts, so `"fp8"`
  raises a clear error there: use `"int8"` for a tf-portable ~4× option. fp8's
  `float8_e4m3fn` storage also does not round-trip through Keras `.weights.h5` (the
  variables come back empty on load), so an fp8-quantized model runs but cannot be saved
  to / loaded from `.weights.h5`; use int8 for a savable ~4× scheme.
- **No calibrated PTQ (GPTQ / AWQ).** This is round-to-nearest weight
  quantization; calibration-based methods for higher int4 accuracy are not
  included.
