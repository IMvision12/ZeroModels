# Mixtral

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Mistral AI's sparse Mixture-of-Experts LLM, ported to pure Keras 3. The
attention stack matches Mistral; each MLP is replaced by a router over experts,
of which only `num_experts_per_tok` run per token.

Memory is governed by **total** parameters, not active ones: every expert stays
resident, so plan capacity for the full weight count even though a fraction runs
per token.

Links:

- Paper: [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088)
- HF docs: [transformers/model_doc/mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)

See also [mistral.md](mistral.md), [mistral3.md](mistral3.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `mixtral-8x7b` | [`mistralai/Mixtral-8x7B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) |
| `mixtral-8x7b-instruct` | [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) |
| `mixtral-8x22b` | [`mistralai/Mixtral-8x22B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| `mixtral-8x22b-instruct` | [`mistralai/Mixtral-8x22B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) |

## API

### `MixtralModel`

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
| `num_experts` | `8` | expert count |
| `num_experts_per_tok` | `2` | experts routed per token |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `sliding_window` | `None` | local attention span |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `MixtralTextGenerate`

`MixtralModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `MixtralModel`.

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

### `MixtralTokenizer`

Tokenizer on the `tokenizers` backend.

```python
MixtralTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.mixtral import MixtralTextGenerate, MixtralTokenizer

model = MixtralTextGenerate.from_weights("mixtral-8x7b")
tokenizer = MixtralTokenizer.from_weights("mixtral-8x7b")

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
from zeromodels.models.mixtral import MixtralModel

backbone = MixtralModel.from_weights("mixtral-8x7b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = MixtralTextGenerate.from_weights("hf:mistralai/Mixtral-8x7B-v0.1")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = MixtralTextGenerate.from_weights(
    "mixtral-8x7b", quantization="int8", load_dtype="bfloat16"
)
```
