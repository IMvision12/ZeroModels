# Llama 2

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

The second Llama generation, ported to pure Keras 3. Same decoder shape as
Llama (RMSNorm, SwiGLU, rotary embeddings) with grouped-query attention on the
larger variants and a 4K context window. The `-chat` checkpoints are the
RLHF-tuned conversational models.

Links:

- Paper: [Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- HF docs: [transformers/model_doc/llama2](https://huggingface.co/docs/transformers/model_doc/llama2)

See also [llama.md](llama.md), [llama4.md](llama4.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `llama2-7b` | [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| `llama2-7b-chat` | [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| `llama2-13b` | [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| `llama2-13b-chat` | [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| `llama2-70b` | [`meta-llama/Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf) |
| `llama2-70b-chat` | [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
| `codellama-7b` | [`codellama/CodeLlama-7b-hf`](https://huggingface.co/codellama/CodeLlama-7b-hf) |
| `codellama-13b` | [`codellama/CodeLlama-13b-hf`](https://huggingface.co/codellama/CodeLlama-13b-hf) |
| `codellama-34b` | [`codellama/CodeLlama-34b-hf`](https://huggingface.co/codellama/CodeLlama-34b-hf) |
| `codellama-70b` | [`codellama/CodeLlama-70b-hf`](https://huggingface.co/codellama/CodeLlama-70b-hf) |

## API

### `Llama2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `32000` | token vocabulary size |
| `embed_dim` | `4096` | model width |
| `mlp_dim` | `11008` | MLP inner width |
| `num_layers` | `32` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `32` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Llama2TextGenerate`

`Llama2Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Llama2Model`.

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

### `Llama2Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Llama2Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.llama2 import Llama2TextGenerate, Llama2Tokenizer

model = Llama2TextGenerate.from_weights("llama2-7b")
tokenizer = Llama2Tokenizer.from_weights("llama2-7b")

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
from zeromodels.models.llama2 import Llama2Model

backbone = Llama2Model.from_weights("llama2-7b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Llama2TextGenerate.from_weights("hf:meta-llama/Llama-2-7b-hf")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Llama2TextGenerate.from_weights(
    "llama2-7b", quantization="int8", load_dtype="bfloat16"
)
```
