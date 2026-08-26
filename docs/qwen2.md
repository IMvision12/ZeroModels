# Qwen2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> plus the Keras weights:
<code>model.weights.h5</code>, or a sharded <code>model.weights.json</code> + shards for the
larger checkpoints). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen2 dense decoder-only LLM, ported to pure Keras 3. A pre-norm
transformer with RMSNorm, SwiGLU MLP, rotary embeddings, grouped-query attention
and biased QKV projections. The smaller variants tie the embedding matrix to the
LM head.

Links:

- HF collection: [Qwen2](https://huggingface.co/collections/zeromodels/qwen2-6a69d274d16370be5d0221c8)
- Paper: [Qwen2 Technical Report (arXiv:2407.10671)](https://arxiv.org/abs/2407.10671)
- HF docs: [transformers/model_doc/qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)

See also [qwen2_moe.md](qwen2_moe.md), [qwen3.md](qwen3.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. The `-instruct` suffix
marks instruction-tuned checkpoints (use the chat template); bare names are base models.

| Variant | Hub |
|---|---|
| `qwen2-0.5b` | [`zeromodels/qwen2-0.5b`](https://huggingface.co/zeromodels/qwen2-0.5b) |
| `qwen2-0.5b-instruct` | [`zeromodels/qwen2-0.5b-instruct`](https://huggingface.co/zeromodels/qwen2-0.5b-instruct) |
| `qwen2-1.5b` | [`zeromodels/qwen2-1.5b`](https://huggingface.co/zeromodels/qwen2-1.5b) |
| `qwen2-1.5b-instruct` | [`zeromodels/qwen2-1.5b-instruct`](https://huggingface.co/zeromodels/qwen2-1.5b-instruct) |
| `qwen2-7b` | [`zeromodels/qwen2-7b`](https://huggingface.co/zeromodels/qwen2-7b) |
| `qwen2-7b-instruct` | [`zeromodels/qwen2-7b-instruct`](https://huggingface.co/zeromodels/qwen2-7b-instruct) |
| `qwen2-72b` | [`zeromodels/qwen2-72b`](https://huggingface.co/zeromodels/qwen2-72b) |
| `qwen2-72b-instruct` | [`zeromodels/qwen2-72b-instruct`](https://huggingface.co/zeromodels/qwen2-72b-instruct) |

## API

### `Qwen2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `896` | model width |
| `mlp_dim` | `4864` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `14` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `Qwen2TextGenerate`

`Qwen2Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen2Model`.

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

### `Qwen2Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen2Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.qwen2 import Qwen2TextGenerate, Qwen2Tokenizer

model = Qwen2TextGenerate.from_weights("zeromodels/qwen2-0.5b-instruct")
tokenizer = Qwen2Tokenizer.from_weights("zeromodels/qwen2-0.5b-instruct")

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
from zeromodels.models.qwen2 import Qwen2Model

backbone = Qwen2Model.from_weights("zeromodels/qwen2-0.5b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen2TextGenerate.from_weights("hf:Qwen/Qwen2-0.5B")
```

### Lower memory

The 72B checkpoint needs quantization to fit comfortably on a single 80GB GPU. See
[quantization.md](quantization.md):

```python
model = Qwen2TextGenerate.from_weights(
    "zeromodels/qwen2-72b-instruct",
    quantization="int8",
)
```
