# Gemma 2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The second Gemma generation, ported to pure Keras 3. It keeps the Gemma decoder shape
(RMSNorm, GeGLU MLP, rotary embeddings, tied embeddings) and adds three things that
matter for parity:

- **Alternating attention**: every other layer uses a `sliding_window` local span
  instead of full global attention.
- **Logit softcapping**: attention logits are capped at `attn_logit_softcapping`
  (50.0) and the final LM logits at `final_logit_softcapping` (30.0) via a scaled
  `tanh`.
- **Grouped-query attention** (`num_kv_heads=4`) rather than Gemma 1's multi-query.

Links:

- Paper: [Gemma 2: Improving Open Language Models at a Practical Size (arXiv:2408.00118)](https://arxiv.org/abs/2408.00118)
- HF docs: [transformers/model_doc/gemma2](https://huggingface.co/docs/transformers/model_doc/gemma2)

See also [gemma.md](gemma.md), [gemma3.md](gemma3.md), [gemma4.md](gemma4.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. The `-it` suffix marks
instruction-tuned checkpoints (use the chat template); bare names are base models.

| Variant | Hub |
|---|---|
| `gemma-2-2b` | [`zeromodels/gemma-2-2b`](https://huggingface.co/zeromodels/gemma-2-2b) |
| `gemma-2-2b-it` | [`zeromodels/gemma-2-2b-it`](https://huggingface.co/zeromodels/gemma-2-2b-it) |
| `gemma-2-9b` | [`zeromodels/gemma-2-9b`](https://huggingface.co/zeromodels/gemma-2-9b) |
| `gemma-2-9b-it` | [`zeromodels/gemma-2-9b-it`](https://huggingface.co/zeromodels/gemma-2-9b-it) |
| `gemma-2-27b` | [`zeromodels/gemma-2-27b`](https://huggingface.co/zeromodels/gemma-2-27b) |
| `gemma-2-27b-it` | [`zeromodels/gemma-2-27b-it`](https://huggingface.co/zeromodels/gemma-2-27b-it) |

## API

### `Gemma2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `256000` | token vocabulary size |
| `embed_dim` | `2304` | model width |
| `mlp_dim` | `9216` | GeGLU inner width |
| `num_layers` | `26` | decoder blocks |
| `num_heads` | `8` | query heads |
| `num_kv_heads` | `4` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `query_pre_attn_scalar` | `256.0` | query scaling divisor applied before attention |
| `attn_logit_softcapping` | `50.0` | tanh cap on attention logits |
| `final_logit_softcapping` | `30.0` | tanh cap on output logits |
| `sliding_window` | `4096` | local attention span on alternating layers |
| `norm_eps` | `1e-6` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `Gemma2TextGenerate`

`Gemma2Model` plus a (tied) LM head with final-logit softcapping. Returns
`{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Gemma2Model`.

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

### `Gemma2Tokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend.

```python
Gemma2Tokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull `tokenizer.json` from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string, a list of strings (a batch), or a chat-message list, which is
routed through `apply_chat_template` automatically.

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gemma2 import Gemma2TextGenerate, Gemma2Tokenizer

model = Gemma2TextGenerate.from_weights("zeromodels/gemma-2-2b-it")
tokenizer = Gemma2Tokenizer.from_weights("zeromodels/gemma-2-2b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
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
from zeromodels.models.gemma2 import Gemma2Model

backbone = Gemma2Model.from_weights("zeromodels/gemma-2-2b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 2304)
```

### Loading from the Hub

```python
model = Gemma2TextGenerate.from_weights("hf:google/gemma-2-9b-it")
```

### Lower memory

The 27B checkpoint needs quantization to fit comfortably on a single 80GB GPU. See
[quantization.md](quantization.md):

```python
model = Gemma2TextGenerate.from_weights(
    "zeromodels/gemma-2-27b-it",
    quantization="int8",
    load_dtype="bfloat16",
)
```
