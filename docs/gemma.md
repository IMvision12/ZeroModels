# Gemma

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Google's first open decoder-only LLM family, built from the Gemini research line and
ported here to pure Keras 3. The decoder is a fairly classic pre-norm transformer:
RMSNorm, a GeGLU MLP, rotary position embeddings, and multi-query attention
(`num_kv_heads=1`) with the embedding matrix tied to the LM head.

- Paper: [Gemma: Open Models Based on Gemini Research and Technology (arXiv:2403.08295)](https://arxiv.org/abs/2403.08295)
- HF docs: [transformers/model_doc/gemma](https://huggingface.co/docs/transformers/model_doc/gemma)

See also [gemma2.md](gemma2.md), [gemma3.md](gemma3.md), [gemma4.md](gemma4.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. The `-it` suffix marks
instruction-tuned checkpoints (use the chat template); bare names are base models.
The `1.1` releases are updated instruction tunes of the same architecture.

| Variant | Hub |
|---|---|
| `gemma-2b` | [`zeromodels/gemma-2b`](https://huggingface.co/zeromodels/gemma-2b) |
| `gemma-2b-it` | [`zeromodels/gemma-2b-it`](https://huggingface.co/zeromodels/gemma-2b-it) |
| `gemma-1.1-2b-it` | [`zeromodels/gemma-1.1-2b-it`](https://huggingface.co/zeromodels/gemma-1.1-2b-it) |
| `gemma-7b` | [`zeromodels/gemma-7b`](https://huggingface.co/zeromodels/gemma-7b) |
| `gemma-7b-it` | [`zeromodels/gemma-7b-it`](https://huggingface.co/zeromodels/gemma-7b-it) |
| `gemma-1.1-7b-it` | [`zeromodels/gemma-1.1-7b-it`](https://huggingface.co/zeromodels/gemma-1.1-7b-it) |

## API

### `GemmaModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.
Use it for feature extraction or as the base for a custom head.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `256000` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `mlp_dim` | `16384` | GeGLU inner width |
| `num_layers` | `18` | decoder blocks |
| `num_heads` | `8` | query heads |
| `num_kv_heads` | `1` | key/value heads (1 = multi-query attention) |
| `head_dim` | `256` | per-head width |
| `norm_eps` | `1e-6` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `GemmaTextGenerate`

`GemmaModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and
adds `.generate()`. Takes the same constructor arguments as `GemmaModel`.

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
| `max_new_tokens` | `None` | tokens to generate (falls back to the model default) |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |

### `GemmaTokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend.

```python
GemmaTokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull `tokenizer.json` from |
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

from zeromodels.models.gemma import GemmaTextGenerate, GemmaTokenizer

model = GemmaTextGenerate.from_weights("zeromodels/gemma-2b-it")
tokenizer = GemmaTokenizer.from_weights("zeromodels/gemma-2b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Batch

Pass a list of strings. The tokenizer pads them and `generate` runs the batch together:

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
from zeromodels.models.gemma import GemmaModel

backbone = GemmaModel.from_weights("zeromodels/gemma-2b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 2048)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including community
fine-tunes:

```python
model = GemmaTextGenerate.from_weights("hf:google/gemma-2b-it")
```

### Lower memory

The 7B checkpoints load in bf16 or quantized weight-only. See [quantization.md](quantization.md):

```python
model = GemmaTextGenerate.from_weights(
    "zeromodels/gemma-7b-it",
    quantization="int8",
    load_dtype="bfloat16",
)
```
