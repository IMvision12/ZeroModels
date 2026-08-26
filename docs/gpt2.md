# GPT-2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

OpenAI's GPT-2 in pure Keras 3: the classic decoder-only language model with learned
token + absolute-position embeddings, pre-LayerNorm causal transformer blocks, a final
LayerNorm (`ln_f`), and a tied LM head. One implementation runs unmodified on
**TensorFlow / Torch / JAX**, bit-close to Hugging Face.

Implementation details that matter for parity:

- **Conv1D `(in, out)` weights**: the attention and MLP projections keep GPT-2's
  `Conv1D` layout and are copied without transposing.
- **`gelu_new`**: the MLP uses the tanh-gelu approximation.
- **Pre-LayerNorm** blocks with a final `ln_f`, and a tied output head (the transposed
  token embedding).

GPT-2 is a **base** language model: it continues a prompt and has no chat template.

Links:

- Paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- HF docs: [transformers/model_doc/gpt2](https://huggingface.co/docs/transformers/model_doc/gpt2)

See also [gpt.md](gpt.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. `gpt2_large` /
`gpt2_xl` are **sharded** on the Hub (a `.weights.json` index plus shards); the two
smaller sizes are a single `.weights.h5`.

| Variant | Hub | Layers | Width | Heads | Params |
|---|---|---|---|---|---|
| `gpt2` | [`zeromodels/gpt2`](https://huggingface.co/zeromodels/gpt2) | 12 | 768 | 12 | 124M |
| `gpt2_medium` | [`zeromodels/gpt2_medium`](https://huggingface.co/zeromodels/gpt2_medium) | 24 | 1024 | 16 | 355M |
| `gpt2_large` | [`zeromodels/gpt2_large`](https://huggingface.co/zeromodels/gpt2_large) | 36 | 1280 | 20 | 774M |
| `gpt2_xl` | [`zeromodels/gpt2_xl`](https://huggingface.co/zeromodels/gpt2_xl) | 48 | 1600 | 25 | 1.5B |

## API

### `GPT2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.
The defaults below are the `gpt2` 124M size; the larger sizes are in the
[Variants](#variants) table.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `50257` | token vocabulary size |
| `embed_dim` | `768` | model width |
| `mlp_dim` | `3072` | MLP inner width |
| `num_layers` | `12` | decoder blocks |
| `num_heads` | `12` | attention heads |
| `max_position_embeddings` | `1024` | learned position table size (context length) |
| `norm_eps` | `1e-5` | LayerNorm epsilon |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `GPT2TextGenerate`

`GPT2Model` plus a tied LM head. Returns
`{"logits": (batch, seq, vocab_size), "last_hidden_state": ...}` and adds `.generate()`.
Same constructor arguments as `GPT2Model`.

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
| `eos_token_id` | `None` | stop token (defaults to GPT-2's `<|endoftext|>`, `50256`) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |

### `GPT2Tokenizer`

Byte-level BPE tokenizer on the `tokenizers` backend. `<|endoftext|>` is the only
special token; there is no chat template.

```python
GPT2Tokenizer(variant="gpt2", hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `"gpt2"` | resolves to the `zeromodels/<variant>` repo's `tokenizer.json` |
| `hf_id` | `None` | explicit Hub repo to pull `tokenizer.json` from (overrides `variant`) |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It accepts
a plain string or a list of strings (a batch).

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gpt2 import GPT2TextGenerate, GPT2Tokenizer

model = GPT2TextGenerate.from_weights("zeromodels/gpt2")
tokenizer = GPT2Tokenizer.from_weights("zeromodels/gpt2")

inputs = tokenizer("The meaning of life is")
outputs = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(outputs[0]))
```

### Batch

```python
prompts = [
    "The capital of France is",
    "Once upon a time,",
    "In a shocking finding, scientists discovered",
]
inputs = tokenizer(prompts)  # {"input_ids": (3, seq), "attention_mask": (3, seq)}
outputs = model.generate(**inputs, max_new_tokens=40)

for text in tokenizer.batch_decode(outputs):
    print(text)
```

### Backbone only

```python
from zeromodels.models.gpt2 import GPT2Model

backbone = GPT2Model.from_weights("zeromodels/gpt2")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 768)
```

### Loading from the Hub

Any upstream GPT-2 checkpoint converts on the fly with the `hf:` prefix:

```python
model = GPT2TextGenerate.from_weights("hf:openai-community/gpt2-medium")
```

### Larger sizes

`gpt2_medium` / `gpt2_large` / `gpt2_xl` load the same way; just change the variant on
both the model and the tokenizer:

```python
model = GPT2TextGenerate.from_weights("zeromodels/gpt2_xl")
tokenizer = GPT2Tokenizer.from_weights("zeromodels/gpt2_xl")
```

## Verified parity

`GPT2TextGenerate` logits vs the real `openai-community/gpt2` (HF, eager attention):
**max |Δ| 4.6e-5**, argmax 100% agree. Build + forward + `.generate()` pass on
TF / Torch / JAX.
