# Qwen3

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> and <code>tokenizer.json</code> plus the
Keras weights: <code>model.weights.h5</code>, or a sharded <code>model.weights.json</code> +
shards for the larger checkpoints). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen3 dense decoder-only LLM, ported to pure Keras 3. It drops Qwen2's
QKV biases and adds QK-RMSNorm (queries and keys are normalized before
attention), keeping RMSNorm, SwiGLU and grouped-query attention otherwise.

Links:

- HF collection: [Qwen3](https://huggingface.co/collections/zeromodels/qwen3-6a7d3fcc4e56b32e86f5b2c4)
- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3)

See also [qwen2.md](qwen2.md), [qwen3_moe.md](qwen3_moe.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-base` suffix marks the base
(non-instruction-tuned) checkpoints, and the `-2507` variants are the July 2025 refresh
(same 4B architecture, longer context). Qwen3 is Apache 2.0. The MoE sizes live on
[qwen3_moe.md](qwen3_moe.md).

| Variant | Hub |
|---|---|
| `qwen3-0.6b` | [`zeromodels/qwen3-0.6b`](https://huggingface.co/zeromodels/qwen3-0.6b) |
| `qwen3-0.6b-base` | [`zeromodels/qwen3-0.6b-base`](https://huggingface.co/zeromodels/qwen3-0.6b-base) |
| `qwen3-1.7b` | [`zeromodels/qwen3-1.7b`](https://huggingface.co/zeromodels/qwen3-1.7b) |
| `qwen3-1.7b-base` | [`zeromodels/qwen3-1.7b-base`](https://huggingface.co/zeromodels/qwen3-1.7b-base) |
| `qwen3-4b` | [`zeromodels/qwen3-4b`](https://huggingface.co/zeromodels/qwen3-4b) |
| `qwen3-4b-base` | [`zeromodels/qwen3-4b-base`](https://huggingface.co/zeromodels/qwen3-4b-base) |
| `qwen3-4b-instruct-2507` | [`zeromodels/qwen3-4b-instruct-2507`](https://huggingface.co/zeromodels/qwen3-4b-instruct-2507) |
| `qwen3-4b-thinking-2507` | [`zeromodels/qwen3-4b-thinking-2507`](https://huggingface.co/zeromodels/qwen3-4b-thinking-2507) |
| `qwen3-8b` | [`zeromodels/qwen3-8b`](https://huggingface.co/zeromodels/qwen3-8b) |
| `qwen3-8b-base` | [`zeromodels/qwen3-8b-base`](https://huggingface.co/zeromodels/qwen3-8b-base) |
| `qwen3-14b` | [`zeromodels/qwen3-14b`](https://huggingface.co/zeromodels/qwen3-14b) |
| `qwen3-14b-base` | [`zeromodels/qwen3-14b-base`](https://huggingface.co/zeromodels/qwen3-14b-base) |
| `qwen3-32b` | [`zeromodels/qwen3-32b`](https://huggingface.co/zeromodels/qwen3-32b) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3-4B-Instruct-2507")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `1024` | model width |
| `mlp_dim` | `3072` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `Qwen3TextGenerate`

`Qwen3Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3Model`.

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

### `Qwen3Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3Tokenizer(hf_id=None, tokenizer_file=None)
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

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.qwen3 import Qwen3TextGenerate, Qwen3Tokenizer

model = Qwen3TextGenerate.from_weights("zeromodels/qwen3-0.6b")
tokenizer = Qwen3Tokenizer.from_weights("zeromodels/qwen3-0.6b")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
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
from zeromodels.models.qwen3 import Qwen3Model

backbone = Qwen3Model.from_weights("zeromodels/qwen3-0.6b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3TextGenerate.from_weights("hf:Qwen/Qwen3-0.6B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3TextGenerate.from_weights(
    "zeromodels/qwen3-0.6b", quantization="int8", load_dtype="bfloat16"
)
```
