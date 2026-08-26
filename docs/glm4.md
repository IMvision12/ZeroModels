# GLM-4

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>, stored
in <b>bfloat16</b>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>,
or convert an original checkpoint on the fly with
<code>from_weights("hf:zai-org/GLM-4-9B-0414")</code>. See
<a href="../loading_weights/">Loading Weights</a>.
</div>

Zhipu's GLM-4 dense decoder-only LLM, ported to pure Keras 3. It uses partial
rotary embeddings (only `partial_rotary_factor` of each head is rotated) with a
NeoX-style rope layout, post-norm residuals, and biased QKV projections.

Note the rope trap: GLM and GLM-4 use *interleaved* partial rope, while GLM-4.5
and later MoE models use the NeoX (half-split) layout. Mixing them silently
destroys parity.

Links:

- Paper: [ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools (arXiv:2406.12793)](https://arxiv.org/abs/2406.12793)
- HF docs: [transformers/model_doc/glm4](https://huggingface.co/docs/transformers/model_doc/glm4)

See also [glm4_moe.md](glm4_moe.md), [glm5_moe.md](glm5_moe.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")` (or convert the
upstream checkpoint on the fly with `from_weights("hf:<upstream>")`).

| Variant | Hosted | Upstream |
|---|---|---|
| `glm-4-9b-0414` | `zeromodels/glm-4-9b-0414` | [`zai-org/GLM-4-9B-0414`](https://huggingface.co/zai-org/GLM-4-9B-0414) |
| `glm-4-32b-0414` | `zeromodels/glm-4-32b-0414` | [`zai-org/GLM-4-32B-0414`](https://huggingface.co/zai-org/GLM-4-32B-0414) |
| `glm-4-32b-base-0414` | `zeromodels/glm-4-32b-base-0414` | [`zai-org/GLM-4-32B-Base-0414`](https://huggingface.co/zai-org/GLM-4-32B-Base-0414) |
| `glm-z1-9b-0414` | `zeromodels/glm-z1-9b-0414` | [`zai-org/GLM-Z1-9B-0414`](https://huggingface.co/zai-org/GLM-Z1-9B-0414) |
| `glm-z1-32b-0414` | `zeromodels/glm-z1-32b-0414` | [`zai-org/GLM-Z1-32B-0414`](https://huggingface.co/zai-org/GLM-Z1-32B-0414) |

## API

### `Glm4Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151552` | token vocabulary size |
| `embed_dim` | `4096` | model width |
| `num_layers` | `40` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mlp_dim` | `13696` | MLP inner width |
| `partial_rotary_factor` | `0.5` | fraction of each head that gets rotated |
| `norm_eps` | `1.5625e-07` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `attention_bias` | `True` | add bias terms to the qkv projections |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Glm4TextGenerate`

`Glm4Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Glm4Model`.

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

### `Glm4Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Glm4Tokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull the tokenizer files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string or a list of strings (a batch). This tokenizer has no chat
template, so pass a prompt you have already rendered rather than a message list.
Decode with `.decode(ids)` for one sequence or `.batch_decode(ids)` for a batch.

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.glm4 import Glm4TextGenerate, Glm4Tokenizer

model = Glm4TextGenerate.from_weights("zeromodels/glm-4-9b-0414")
tokenizer = Glm4Tokenizer.from_weights("zeromodels/glm-4-9b-0414")

inputs = tokenizer("Explain rotary embeddings in one sentence.")
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
from zeromodels.models.glm4 import Glm4Model

backbone = Glm4Model.from_weights("zeromodels/glm-4-9b-0414")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Glm4TextGenerate.from_weights("hf:zai-org/GLM-4-9B-0414")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Glm4TextGenerate.from_weights(
    "zeromodels/glm-4-9b-0414", quantization="int8", load_dtype="bfloat16"
)
```
