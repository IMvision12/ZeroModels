# Llama

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Meta's first open Llama generation, ported to pure Keras 3. A pre-norm
decoder-only transformer: RMSNorm, SwiGLU MLP, rotary position embeddings and
multi-head attention. This family also carries the Llama 3.x checkpoints, which
keep the same block shape and move to grouped-query attention with a 128K
vocabulary.

Links:

- Paper: [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
- HF docs: [transformers/model_doc/llama](https://huggingface.co/docs/transformers/model_doc/llama)

See also [llama2.md](llama2.md), [llama4.md](llama4.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `llama3-8b` | [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| `llama3-8b-instruct` | [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| `llama3-70b` | [`meta-llama/Meta-Llama-3-70B`](https://huggingface.co/meta-llama/Meta-Llama-3-70B) |
| `llama3-70b-instruct` | [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |
| `llama3.1-8b` | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| `llama3.1-8b-instruct` | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `llama3.1-70b` | [`meta-llama/Llama-3.1-70B`](https://huggingface.co/meta-llama/Llama-3.1-70B) |
| `llama3.1-70b-instruct` | [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| `llama3.2-1b` | [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| `llama3.2-1b-instruct` | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| `llama3.2-3b` | [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| `llama3.2-3b-instruct` | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `llama3.3-70b-instruct` | [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |

## API

### `LlamaModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `128256` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `mlp_dim` | `8192` | MLP inner width |
| `num_layers` | `16` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `500000.0` | rotary base frequency |
| `rope_factor` | `32.0` |  |
| `rope_low_freq_factor` | `1.0` |  |
| `rope_high_freq_factor` | `4.0` |  |
| `rope_original_max_pos` | `8192` |  |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `LlamaTextGenerate`

`LlamaModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `LlamaModel`.

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

### `LlamaTokenizer`

Tokenizer on the `tokenizers` backend.

```python
LlamaTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.llama import LlamaTextGenerate, LlamaTokenizer

model = LlamaTextGenerate.from_weights("llama3-8b")
tokenizer = LlamaTokenizer.from_weights("llama3-8b")

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
from zeromodels.models.llama import LlamaModel

backbone = LlamaModel.from_weights("llama3-8b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = LlamaTextGenerate.from_weights("hf:meta-llama/Meta-Llama-3-8B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = LlamaTextGenerate.from_weights(
    "llama3-8b", quantization="int8", load_dtype="bfloat16"
)
```
