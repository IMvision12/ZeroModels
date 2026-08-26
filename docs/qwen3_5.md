# Qwen3.5 (Qwen3-Next)

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> and <code>tokenizer.json</code> plus the
Keras weights: <code>model.weights.h5</code>, or a sharded <code>model.weights.json</code> +
shards for the larger checkpoints). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen3.5 (Qwen3-Next) hybrid-attention model, ported to pure Keras 3. It
interleaves Gated-DeltaNet linear-attention layers with periodic full-attention layers
(`full_attention_interval`), keeping partial rotary embeddings and QK-RMSNorm on the
full-attention path. The dense checkpoints are vision-language models: a ViT vision
tower feeds soft tokens into the text decoder. Two heads drive generation, the two-class
API used across the library: `Qwen3_5TextGenerate` (text-only) and
`Qwen3_5ConditionalGenerate` (image + text). A multimodal repo loads under
`Qwen3_5ConditionalGenerate`; `Qwen3_5TextGenerate` reads the same checkpoint and keeps
just the text backbone, dropping the vision tower.


See also [qwen3.md](qwen3.md), [qwen3_next.md](qwen3_next.md).

Collection: [Qwen3.5](https://huggingface.co/collections/zeromodels/qwen35-6a7e5421737d73e63669ebb9)

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-base` suffix marks the base
(non-instruction-tuned) checkpoints. Qwen3.5 is Apache 2.0. The MoE sizes live on
[qwen3_5_moe.md](qwen3_5_moe.md).

| Variant | Hub |
|---|---|
| `qwen3.5-0.8b` | [`zeromodels/qwen3.5-0.8b`](https://huggingface.co/zeromodels/qwen3.5-0.8b) |
| `qwen3.5-0.8b-base` | [`zeromodels/qwen3.5-0.8b-base`](https://huggingface.co/zeromodels/qwen3.5-0.8b-base) |
| `qwen3.5-2b` | [`zeromodels/qwen3.5-2b`](https://huggingface.co/zeromodels/qwen3.5-2b) |
| `qwen3.5-2b-base` | [`zeromodels/qwen3.5-2b-base`](https://huggingface.co/zeromodels/qwen3.5-2b-base) |
| `qwen3.5-4b` | [`zeromodels/qwen3.5-4b`](https://huggingface.co/zeromodels/qwen3.5-4b) |
| `qwen3.5-4b-base` | [`zeromodels/qwen3.5-4b-base`](https://huggingface.co/zeromodels/qwen3.5-4b-base) |
| `qwen3.5-9b` | [`zeromodels/qwen3.5-9b`](https://huggingface.co/zeromodels/qwen3.5-9b) |
| `qwen3.5-9b-base` | [`zeromodels/qwen3.5-9b-base`](https://huggingface.co/zeromodels/qwen3.5-9b-base) |
| `qwen3.5-27b` | [`zeromodels/qwen3.5-27b`](https://huggingface.co/zeromodels/qwen3.5-27b) |
| `qwen3.8-27b` | [`zeromodels/qwen3.8-27b`](https://huggingface.co/zeromodels/qwen3.8-27b) |

`qwen3.8-27b` is Alibaba's Qwen3.8-27B, which shares this same Qwen3.5 architecture (it
loads under `zeromodels.models.qwen3_5`): the full vision-language checkpoint, driven by
`Qwen3_5ConditionalGenerate` for image + text or `Qwen3_5TextGenerate` for text-only.

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3.5-4B")` or `from_weights("hf:Qwen/Qwen3.8-27B")`, which
convert them in process (pass `cache_converted=True` to keep the result). See
[Loading Weights](loading_weights.md).

## API

### `Qwen3_5Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `248320` | token vocabulary size |
| `embed_dim` | `1024` | model width |
| `mlp_dim` | `3584` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `8` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `10000000.0` | rotary base frequency |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |
| `full_attention_interval` | `4` |  |
| `linear_conv_kernel_dim` | `4` |  |
| `linear_key_head_dim` | `128` |  |
| `linear_value_head_dim` | `128` |  |
| `linear_num_key_heads` | `16` |  |
| `linear_num_value_heads` | `16` |  |

### `Qwen3_5TextGenerate`

`Qwen3_5Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3_5Model`.

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

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |

### `Qwen3_5ConditionalGenerate`

The multimodal head, over the `Qwen3_5VLModel` backbone (text decoder + ViT vision tower)
plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds
`.generate()`. Text-only when no image tensors are passed, multimodal when they are; the
prefill runs the vision tower and scatters the resulting soft tokens onto the image
placeholders, then decoding is text-only over the hybrid per-layer cache. Pair it with
`Qwen3_5Processor`.

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    pixel_values=None,
    image_grid_thw=None,
    pixel_values_videos=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` / `seed` | `None` | sampling strategy (greedy when unset) / seed |
| `pixel_values` / `image_grid_thw` | `None` | flattened image patches + their `(t, h, w)` grid |
| `pixel_values_videos` | `None` | flattened video patches |

### `Qwen3_5Processor`

Image + text processor: a 16px-patch Qwen image processor paired with the Qwen3.5
tokenizer. Call it on a chat conversation (text and `image` items) and it returns
`{"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}`, ready to splat into
`generate`. Decode with `.decode(ids)` / `.batch_decode(ids)`.

### `Qwen3_5Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3_5Tokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull the tokenizer files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string, a list of strings (a batch), or a chat-message list, which is
routed through `apply_chat_template` automatically. Decode with `.decode(ids)` for one
sequence or `.batch_decode(ids)` for a batch.

## End-to-end example

### Single input (text only)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.qwen3_5 import Qwen3_5TextGenerate, Qwen3_5Tokenizer

model = Qwen3_5TextGenerate.from_weights("zeromodels/qwen3.5-0.8b")
tokenizer = Qwen3_5Tokenizer.from_weights("zeromodels/qwen3.5-0.8b")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Single input (image + text)

```python
from zeromodels.models.qwen3_5 import Qwen3_5ConditionalGenerate, Qwen3_5Processor

model = Qwen3_5ConditionalGenerate.from_weights("zeromodels/qwen3.5-4b")
processor = Qwen3_5Processor.from_weights("zeromodels/qwen3.5-4b")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://.../cat.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
inputs = processor(conversation)
# inputs -> input_ids, attention_mask, pixel_values, image_grid_thw
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Batch

Pass a list of strings. The tokenizer pads them and `generate` runs the batch
together:

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
from zeromodels.models.qwen3_5 import Qwen3_5Model

backbone = Qwen3_5Model.from_weights("zeromodels/qwen3.5-0.8b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3_5TextGenerate.from_weights("hf:Qwen/Qwen3.5-0.8B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3_5TextGenerate.from_weights(
    "zeromodels/qwen3.5-0.8b", quantization="int8", load_dtype="bfloat16"
)
```
