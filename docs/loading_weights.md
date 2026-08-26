---
hide:
  - navigation
---

# Loading Weights

Every model and preprocessor loads through one call:

```python
model = SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")
```

Behind that one call sit **three ways** weights actually reach the model. You rarely choose
between them explicitly, but knowing which one a model uses explains what you see on first
load, and how long it takes.

| # | Way | What happens | Used by |
|---|---|---|---|
| 1 | [**HuggingFace Hub**](#1-hub-keras-weights) | `zeromodels/<variant>`. The repo's `zm_config.json` rebuilds the model and `model.weights.h5` (or a sharded `.weights.json`) loads with no conversion. | Vision, detection, segmentation, depth, speech, text encoders, CLIP-family, classification backbones |
| 2 | [**On the fly**](#2-on-the-fly-conversion) | A bare variant whose entry carries an `hf_id`. Upstream safetensors are downloaded and converted in process. | The LLMs and VLMs: Qwen, Llama, Gemma, DeepSeek, GLM, Mistral, ... |
| 3 | [**`hf:` prefix**](#3-the-hf-prefix) | Any Hub repo, named explicitly. Same conversion machinery as way 2, but you pick the repo. | Fine-tunes and community weights, for any architecture |

`from_weights` dispatches on the identifier shape: a string with `/` is way 1, a bare
variant is way 2, and an `hf:` prefix is way 3.

```python
model = SegFormerSemanticSegment.from_weights(
    "zeromodels/segformer_b0_ade_512"
)  # 1: Hub Keras
model = Qwen3TextGenerate.from_weights("qwen3-4b")  # 2: on the fly
model = SegFormerSemanticSegment.from_weights("hf:<user>/my-finetune")  # 3: hf:
```

## 1. Hub Keras weights

**Used by the vision, detection, segmentation, depth, speech, text-encoder, and
classification models.** Official checkpoints live under the
[zeromodels](https://huggingface.co/zeromodels) org on Hugging Face. Each repo is
self-describing:

- `zm_config.json` — model class + flat architecture fields
- `model.weights.h5` (or sharded `model.weights.json` + shard files)
- optionally `zm_preprocessor.json` — processor / tokenizer settings

```python
model = SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")
processor = SegFormerImageProcessor.from_weights("zeromodels/segformer_b0_ade_512")
```

Nothing is converted at load time: the weight file is downloaded once (with a tqdm
progress bar), cached, and handed to Keras's own `load_weights`. That makes this the
fastest path, which is why it is the default wherever the checkpoint is small enough to
mirror as Keras weights.

Sharded Hub repos work the same way: a `.weights.json` index pulls every shard it lists
from the same repo before loading.

> Prefer the Hub repo id (`zeromodels/<variant>`) over a bare variant name for these
> families. Bare ids only work for models that still keep an in-package
> `BASE_WEIGHT_CONFIG` with an `hf_id` (the LLMs / VLMs below).

## 2. On-the-fly conversion

**Used by every LLM and VLM in the library.** These variants carry an `hf_id`, so
`from_weights` downloads the original safetensors from the Hub and converts them in
memory:

```python
model = Qwen3TextGenerate.from_weights("qwen3-4b")
```

The reason is **size**. A pre-converted `.weights.h5` for a 4B model is ~8 GB and a 120B
one is hundreds; mirroring that as zeromodels Hub files would duplicate weights the
upstream Hub already serves. Converting on arrival costs CPU time instead of storage, and
the Hub download is cached like any other.

The tradeoff is that **the conversion runs on every load**. To pay it once:

```python
model = Qwen3TextGenerate.from_weights("qwen3-4b", cache_converted=True)
```

That stores the converted result under `$ZEROMODELS_HOME/converted` and rebuilds from it
next time, skipping both the download and the conversion. See [Caching](#caching).

Some families are **gated**. Accept the license on the upstream Hub repo and authenticate:

```shell
huggingface-cli login          # or: export HF_TOKEN=...
```

Every model page whose weights load this way carries a red banner saying so.

## 3. The `hf:` prefix

Way 2 picks the upstream repo for you from a bare variant. Prefix any Hub repo with `hf:`
and you pick it yourself; it is fetched, converted, and loaded in the same call, through
the same machinery. There is no offline conversion step and no intermediate file to
manage.

```python
model = Qwen2TextGenerate.from_weights("hf:Qwen/Qwen2-1.5B-Instruct")
tokenizer = Qwen2Tokenizer.from_weights("hf:Qwen/Qwen2-1.5B-Instruct")
```

What that does:

1. Downloads `config.json` and checks its `model_type` against the class.
2. Maps that config into constructor arguments (`config_from_hf`) and builds the model.
3. Downloads the safetensors and assigns every tensor into the Keras layers
   (`transfer_from_hf`), transposing, splitting, and fusing as each architecture requires.

This runs through `huggingface_hub` only. **`transformers` and `torch` are never
imported**, so the conversion happens wherever you are running, on any backend.

Because step 2 reads the repo's own config, a fine-tune with a different class count,
vocabulary, or image size needs no extra arguments: it is read off the checkpoint. That is
what makes community weights work.

```python
model = SegFormerSemanticSegment.from_weights("hf:<user>/segformer-my-dataset")
```

Point a class at the wrong checkpoint and it fails immediately rather than deep inside the
transfer:

```python
SegFormerSemanticSegment.from_weights("hf:openai/clip-vit-base-patch16")
```

```
ValueError: SegFormerSemanticSegment can only load HF models whose config.json model_type
is segformer, but 'openai/clip-vit-base-patch16' has model_type='clip'. This zeromodels
class is the wrong destination for that checkpoint.
```

Classification backbones can also convert from **timm-style** repos, which carry no
`model_type`. There the variant is inferred from the repo name, and `variant=` overrides it
when a fine-tune does not follow the timm naming convention:

```python
model = ResNetImageClassify.from_weights("hf:timm/resnet50.a1_in1k")
model = ResNetImageClassify.from_weights("hf:<user>/my-resnet", variant="resnet50")
```

Official zeromodels classification weights still prefer the Hub Keras path:

```python
model = ResNetImageClassify.from_weights("zeromodels/resnet50_a1_in1k")
```

> **Load the processor from the same source as the model.** A fine-tune can ship a
> different tokenizer, label set, or normalization; mismatching them fails quietly with
> wrong output rather than loudly with an error.

## Variants and Hub repo ids

Each model page lists the short variant ids in its Model Variants table. For Hub Keras
families, load them as `zeromodels/<variant>`:

```python
model = SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")
```

For LLMs / VLMs, the bare variant is enough — it resolves through the class's
`BASE_WEIGHT_CONFIG` to an upstream `hf_id`.

The architecture travels with the checkpoint (`zm_config.json` on Hub Keras repos, or the
upstream `config.json` for on-the-fly / `hf:` loads), so you do not pass `hidden_size`,
`num_layers`, or `num_classes` unless you are overriding them on purpose.

## Caching

Downloads are cached, so the second load of the same weights is local:

- Hub Keras `.h5` / `.json` files land in `~/.downloads` (streamed with tqdm).
- Hub metadata and safetensors use the standard `huggingface_hub` cache, shared with
  anything else on the machine that has pulled the same repo.

Conversion itself is **not** cached by default, which is what makes way 2 slower on the
second load than way 1. For a checkpoint you load repeatedly, `cache_converted=True` stores
the converted result under `$ZEROMODELS_HOME/converted` (default
`~/.cache/zeromodels/converted`) and rebuilds from it next time, skipping both download
and conversion:

```python
model = Qwen3TextGenerate.from_weights("qwen3-4b", cache_converted=True)  # way 2
model = Qwen3TextGenerate.from_weights(
    "hf:Qwen/Qwen3-8B", cache_converted=True
)  # way 3
```

It works for both conversion paths. Way 1 has nothing to cache beyond the downloaded file.

The cache key includes the source identity, the backend and dtype, and the quantization
recipe, so it cannot hand back a stale or differently configured model. For an `hf:` id the
source identity is the resolved **commit SHA**, so a repo that moves invalidates the entry.
A miss falls back to the normal path silently. On an ephemeral machine (Colab, CI) point
`ZEROMODELS_HOME` at persistent storage or the cache buys you nothing.

## Loading big checkpoints

Two independent flags, composable, for checkpoints that do not comfortably fit:

| Flag | Trades against | Effect |
|---|---|---|
| `load_dtype="bfloat16"` | Device memory | Builds under a bf16 policy so a bf16 checkpoint stays ~2 bytes/param instead of being upcast to fp32. |
| `quantization="int8"` | Device memory | Weight-only quantization of Dense and Embedding layers, roughly 4x, or 8x for `"int4"`. The model builds at `load_dtype` and quantizes after, so peak memory during the load is the float model. See [Quantization](quantization.md). |

```python
model = Qwen3TextGenerate.from_weights(
    "hf:Qwen/Qwen3-8B",
    load_dtype="bfloat16",
    quantization="int8",
)
```

These are memory optimizations, not speed ones. Weight-only quantization in particular
dequantizes on the fly, so it buys footprint, not throughput.

## Architecture only, and partial loads

`load_weights=False` builds the architecture with random initialization. For an `hf:` id
or a Hub Keras repo the config is still fetched to size the model, but the weight files are
not downloaded.

```python
model = SegFormerSemanticSegment.from_weights(
    "zeromodels/segformer_b0_ade_512", load_weights=False
)
```

`skip_mismatch=True` loads everything whose shape agrees and leaves the rest at its
initializer, which is how you keep a pretrained backbone while swapping the head:

```python
model = SegFormerSemanticSegment.from_weights(
    "zeromodels/segformer_b0_ade_512", num_classes=7, skip_mismatch=True
)
print(model.output_shape)
```

```
List of objects that could not be loaded:
[<Conv2D name=head_classifier, built=True>]
(None, 512, 512, 7)
```

Either way it tells you what it skipped. The Hub Keras `.h5` path reports it as the Keras
warning above; the converter path (`hf:` ids and on-the-fly safetensors) prints its own
line instead:

```
[from_weights] skip_mismatch: left 2 weight(s) at their initialized values due to
shape mismatch (e.g. a resized head): [...]
```

Read that list. It is your only signal that the head really was the only thing left
untrained.

## Being explicit

`from_weights` dispatches to three classmethods you can also call directly, when you would
rather the source be visible at the call site than encoded in the identifier:

```python
model = SegFormerSemanticSegment.from_hub_repo("zeromodels/segformer_b0_ade_512")
model = Qwen3TextGenerate.from_variant("qwen3-4b")
model = SegFormerSemanticSegment.from_hf("nvidia/segformer-b0-finetuned-ade-512-512")
```

Full signatures are in [Main Classes](main_classes.md#loading-weights).
