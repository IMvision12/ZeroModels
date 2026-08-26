# GPT

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

OpenAI's original GPT (Radford et al. 2018, "openai-gpt") in pure Keras 3: a
decoder-only language model with learned token + absolute-position embeddings and
post-LayerNorm causal transformer blocks. One implementation runs unmodified on
**TensorFlow / Torch / JAX**, bit-close to Hugging Face.

Same machinery as [GPT-2](gpt2.md) (`Conv1D` `(in, out)` weights copied without
transposing, `gelu_new` activation, tied LM head) with two differences:

- **Post-LayerNorm** blocks: `ln_1(x + attn(x))` then `ln_2(h + mlp(h))`.
- **No final LayerNorm** (GPT-2 adds `ln_f`; GPT does not).

GPT is a **base** language model: it continues a prompt and has no chat template. Its
context length is **512** tokens.

Links:

- Paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- HF docs: [transformers/model_doc/openai-gpt](https://huggingface.co/docs/transformers/model_doc/openai-gpt)

See also [gpt2.md](gpt2.md).

## Variants

Load with `from_weights("zeromodels/<variant>")`. GPT ships a single size.

| Variant | Hub | Layers | Width | Heads | Vocab | Params |
|---|---|---|---|---|---|---|
| `gpt` | [`zeromodels/gpt`](https://huggingface.co/zeromodels/gpt) | 12 | 768 | 12 | 40478 | 117M |

## API

### `GptModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `40478` | token vocabulary size |
| `embed_dim` | `768` | model width |
| `mlp_dim` | `3072` | MLP inner width |
| `num_layers` | `12` | decoder blocks |
| `num_heads` | `12` | attention heads |
| `max_position_embeddings` | `512` | learned position table size (context length) |
| `norm_eps` | `1e-5` | LayerNorm epsilon |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `GptTextGenerate`

`GptModel` plus a tied LM head. Returns
`{"logits": (batch, seq, vocab_size), "last_hidden_state": ...}` and adds `.generate()`.
Same constructor arguments as `GptModel`.

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
| `eos_token_id` | `None` | stop token (GPT has none, so decoding runs the full budget) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |

### `GptTokenizer`

Byte-pair-encoding tokenizer on the `tokenizers` backend. `<unk>` is the unknown token;
there is no chat template.

```python
GptTokenizer(variant="gpt", hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `"gpt"` | resolves to the `zeromodels/<variant>` repo's `tokenizer.json` |
| `hf_id` | `None` | explicit Hub repo to pull `tokenizer.json` from (overrides `variant`) |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It accepts
a plain string or a list of strings (a batch).

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gpt import GptTextGenerate, GptTokenizer

model = GptTextGenerate.from_weights("zeromodels/gpt")
tokenizer = GptTokenizer.from_weights("zeromodels/gpt")

inputs = tokenizer("the meaning of life is")
outputs = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(outputs[0]))
```

### Batch

```python
prompts = [
    "the capital of france is",
    "once upon a time,",
    "in a shocking finding, scientists discovered",
]
inputs = tokenizer(prompts)  # {"input_ids": (3, seq), "attention_mask": (3, seq)}
outputs = model.generate(**inputs, max_new_tokens=40)

for text in tokenizer.batch_decode(outputs):
    print(text)
```

### Backbone only

```python
from zeromodels.models.gpt import GptModel

backbone = GptModel.from_weights("zeromodels/gpt")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 768)
```

### Loading from the Hub

The upstream checkpoint converts on the fly with the `hf:` prefix:

```python
model = GptTextGenerate.from_weights("hf:openai-community/openai-gpt")
```

## Verified parity

`GptTextGenerate` logits vs the real `openai-community/openai-gpt` (HF): **max |Δ| 1.2e-5**,
argmax 100% agree. Build + forward + `.generate()` pass on TF / Torch / JAX.
