# Gemma 3

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The third Gemma generation, ported to pure Keras 3, and the first multimodal one. The
1B variants are text-only; 4B and larger pair the text decoder with a SigLIP vision
tower and an average-pool projector, so the same class handles text or image+text.

What changed from Gemma 2:

- **Vision** (4B+): a SigLIP tower plus `Gemma3MultiModalProjector`, which 4x4
  average-pools patch features into a fixed 256 soft tokens per image.
- **5:1 attention pattern**: five local (`sliding_window`) layers to each global one,
  set by `sliding_window_pattern`.
- **Dual RoPE bases**: global layers use `rope_theta` (1e6), local layers
  `rope_local_theta` (1e4), with optional `rope_scaling_factor` for long context.

Links:

- Paper: [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786)
- HF docs: [transformers/model_doc/gemma3](https://huggingface.co/docs/transformers/model_doc/gemma3)

See also [gemma.md](gemma.md), [gemma2.md](gemma2.md), [gemma4.md](gemma4.md).

## Variants

Load any of these with `from_weights("<variant>")`. `-pt` is the pretrained (base)
checkpoint, `-it` is instruction-tuned (use the chat template).

| Variant | Hub | Vision |
|---|---|---|
| `gemma-3-1b-pt` | [`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt) | text only |
| `gemma-3-1b-it` | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) | text only |
| `gemma-3-4b-pt` | [`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt) | yes |
| `gemma-3-4b-it` | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | yes |
| `gemma-3-12b-pt` | [`google/gemma-3-12b-pt`](https://huggingface.co/google/gemma-3-12b-pt) | yes |
| `gemma-3-12b-it` | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) | yes |
| `gemma-3-27b-pt` | [`google/gemma-3-27b-pt`](https://huggingface.co/google/gemma-3-27b-pt) | yes |
| `gemma-3-27b-it` | [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | yes |

## API

### `Gemma3Model`

The backbone: text decoder, plus the SigLIP tower and projector when
`vision_num_layers > 0`. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `262144` | token vocabulary size |
| `embed_dim` | `1152` | model width |
| `mlp_dim` | `6912` | GeGLU inner width |
| `num_layers` | `26` | decoder blocks |
| `num_heads` | `4` | query heads |
| `num_kv_heads` | `1` | key/value heads |
| `head_dim` | `256` | per-head width |
| `query_pre_attn_scalar` | `256.0` | query scaling divisor before attention |
| `sliding_window` | `512` | local attention span |
| `sliding_window_pattern` | `6` | one global layer every N (5 local : 1 global) |
| `norm_eps` | `1e-6` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base on global layers |
| `rope_local_theta` | `10000.0` | rotary base on local layers |
| `rope_scaling_factor` | `None` | long-context rope scaling |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |
| `vision_embed_dim` | `1152` | SigLIP tower width |
| `vision_mlp_dim` | `4304` | SigLIP MLP width |
| `vision_num_layers` | `0` | SigLIP depth; `0` builds a text-only model |

### `Gemma3ConditionalGenerate`

`Gemma3Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and
adds `.generate()` for text or image+text to text. Same constructor arguments as
`Gemma3Model`.

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    **prefill_inputs,
)
```

Image inputs ride along as `**prefill_inputs` (`pixel_values`), which the processor
produces for you.

### `Gemma3TextGenerate`

The text-only head for the text-only Gemma 3 checkpoints (`gemma-3-1b`, `gemma-3-270m`,
which have no SigLIP tower). Same behaviour as `Gemma3ConditionalGenerate` minus the
`pixel_values` prefill; it shares the backbone weights, so a text-only repo loads under
either head. This is the shared-decoder half of the two-class API: `Gemma3TextGenerate`
(text) and `Gemma3ConditionalGenerate` (multimodal).

```python
from zeromodels.models.gemma3 import Gemma3TextGenerate, Gemma3Tokenizer

model = Gemma3TextGenerate.from_weights("zeromodels/gemma-3-1b-it")
tokenizer = Gemma3Tokenizer.from_weights("zeromodels/gemma-3-1b-it")
outputs = model.generate(**tokenizer("The capital of France is"), max_new_tokens=32)
```

### `Gemma3VisionModel`

The SigLIP vision tower: biased conv patch embed and learned position embeddings into
pre-LN encoder blocks. Built automatically by `Gemma3Model`; you rarely construct it
directly.

```python
Gemma3VisionModel(
    embed_dim,
    mlp_dim,
    num_layers,
    num_heads,
    image_size=896,
    patch_size=14,
    norm_eps=1e-6,
)
```

### `Gemma3MultiModalProjector`

Maps vision features to text-embedding space: 4x4 average pool, soft-token RMSNorm,
then a matmul with the learned projection.

```python
Gemma3MultiModalProjector(
    vision_dim, text_dim, patches_per_image=64, tokens_per_side=16, norm_eps=1e-6
)
```

### `Gemma3ImageProcessor`

Bicubic resize to a fixed square, rescale to `[0, 1]`, normalize.

```python
Gemma3ImageProcessor(size=896, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
```

### `Gemma3Tokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend.

```python
Gemma3Tokenizer(hf_id=None, tokenizer_file=None)
```

### `Gemma3Processor`

Image plus text to model inputs. Renders the chat template, preprocesses images, and
expands each image placeholder into `mm_tokens_per_image` soft tokens.

```python
Gemma3Processor(
    hf_id=None, mm_tokens_per_image=256, tokenizer=None, image_processor=None
)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo for the tokenizer files |
| `mm_tokens_per_image` | `256` | soft tokens each image expands to |
| `tokenizer` | `None` | override the default `Gemma3Tokenizer` |
| `image_processor` | `None` | override the default `Gemma3ImageProcessor` |

## End-to-end example

### Single input (text only)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gemma3 import Gemma3TextGenerate, Gemma3Tokenizer

model = Gemma3TextGenerate.from_weights("gemma-3-1b-it")
tokenizer = Gemma3Tokenizer.from_weights("gemma-3-1b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Single input (image + text)

Use `Gemma3Processor` with a 4B or larger variant:

```python
from PIL import Image
from zeromodels.models.gemma3 import Gemma3ConditionalGenerate, Gemma3Processor

model = Gemma3ConditionalGenerate.from_weights("gemma-3-4b-it")
processor = Gemma3Processor.from_weights("gemma-3-4b-it")

image = Image.open("photo.jpg")
inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Batch

```python
prompts = [
    "The capital of France is",
    "In one sentence, what is a transformer?",
    "Write a haiku about GPUs.",
]
inputs = tokenizer(prompts)  # {"input_ids": (3, seq), "attention_mask": (3, seq)}
outputs = model.generate(**inputs, max_new_tokens=64)

for text in tokenizer.batch_decode(outputs):
    print(text)
```

### Backbone only

```python
from zeromodels.models.gemma3 import Gemma3Model

backbone = Gemma3Model.from_weights("gemma-3-1b-pt")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

```python
model = Gemma3ConditionalGenerate.from_weights("hf:google/gemma-3-4b-it")
```

### Lower memory

The 12B and 27B checkpoints benefit from bf16 or weight-only quantization. See
[quantization.md](quantization.md):

```python
model = Gemma3ConditionalGenerate.from_weights(
    "gemma-3-27b-it", quantization="int8", load_dtype="bfloat16"
)
```
