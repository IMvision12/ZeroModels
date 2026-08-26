# Mistral

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Mistral AI's dense decoder-only LLM, ported to pure Keras 3. It keeps the Llama
block shape (RMSNorm, SwiGLU, rotary embeddings) and adds grouped-query
attention plus a sliding-window attention span, which bounds the KV cache on
long sequences.

Links:

- Paper: [Mistral 7B (arXiv:2310.06825)](https://arxiv.org/abs/2310.06825)
- HF docs: [transformers/model_doc/mistral](https://huggingface.co/docs/transformers/model_doc/mistral)

See also [mixtral.md](mixtral.md), [mistral3.md](mistral3.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `mistral-7b-v0.1` | [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
| `mistral-7b-instruct-v0.2` | [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| `mistral-7b-v0.3` | [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3) |
| `mistral-7b-instruct-v0.3` | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| `ministral-8b-instruct-2410` | [`mistralai/Ministral-8B-Instruct-2410`](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| `mistral-nemo-base-2407` | [`mistralai/Mistral-Nemo-Base-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) |
| `mistral-nemo-instruct-2407` | [`mistralai/Mistral-Nemo-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| `mistral-small-24b-base-2501` | [`mistralai/Mistral-Small-24B-Base-2501`](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) |
| `mistral-small-24b-instruct-2501` | [`mistralai/Mistral-Small-24B-Instruct-2501`](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |

## API

### `MistralModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `32000` | token vocabulary size |
| `embed_dim` | `4096` | model width |
| `mlp_dim` | `14336` | MLP inner width |
| `num_layers` | `32` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `sliding_window` | `None` | local attention span |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `MistralTextGenerate`

`MistralModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `MistralModel`.

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

### `MistralTokenizer`

Tokenizer on the `tokenizers` backend.

```python
MistralTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.mistral import MistralTextGenerate, MistralTokenizer

model = MistralTextGenerate.from_weights("mistral-7b-v0.1")
tokenizer = MistralTokenizer.from_weights("mistral-7b-v0.1")

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
from zeromodels.models.mistral import MistralModel

backbone = MistralModel.from_weights("mistral-7b-v0.1")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = MistralTextGenerate.from_weights("hf:mistralai/Mistral-7B-v0.1")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = MistralTextGenerate.from_weights(
    "mistral-7b-v0.1", quantization="int8", load_dtype="bfloat16"
)
```
