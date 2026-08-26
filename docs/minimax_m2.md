# MiniMax-M2

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

MiniMax's M2 Mixture-of-Experts LLM, ported to pure Keras 3. Unlike
MiniMax-Text-01, M2 uses full attention throughout rather than the lightning /
full hybrid, in front of a sparse expert bank.

The hub checkpoints ship in FP8 and are dequantized during conversion. Memory is
governed by **total** parameters, not active ones.


See also [minimax.md](minimax.md), [minimax_m3_vl.md](minimax_m3_vl.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `minimax-m2` | [`MiniMaxAI/MiniMax-M2`](https://huggingface.co/MiniMaxAI/MiniMax-M2) |

## API

### `MiniMaxM2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `200064` | token vocabulary size |
| `embed_dim` | `3072` | model width |
| `mlp_dim` | `1536` | MLP inner width |
| `num_layers` | `62` | decoder blocks |
| `num_heads` | `48` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `num_experts` | `256` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `partial_rotary_factor` | `1.0` | fraction of each head that gets rotated |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `MiniMaxM2TextGenerate`

`MiniMaxM2Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `MiniMaxM2Model`.

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

### `MiniMaxM2Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
MiniMaxM2Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.minimax_m2 import MiniMaxM2TextGenerate, MiniMaxM2Tokenizer

model = MiniMaxM2TextGenerate.from_weights("minimax-m2")
tokenizer = MiniMaxM2Tokenizer.from_weights("minimax-m2")

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
from zeromodels.models.minimax_m2 import MiniMaxM2Model

backbone = MiniMaxM2Model.from_weights("minimax-m2")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = MiniMaxM2TextGenerate.from_weights("hf:MiniMaxAI/MiniMax-M2")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = MiniMaxM2TextGenerate.from_weights(
    "minimax-m2", quantization="int8", load_dtype="bfloat16"
)
```
