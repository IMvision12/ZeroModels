# MiniMax-Text-01

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

MiniMax's hybrid-attention Mixture-of-Experts LLM, ported to pure Keras 3. Most
layers use lightning (linear) attention, with periodic full-softmax layers, in
front of a sparse expert bank.

Port note: the lightning-attention normalization uses eps=1e-6; changing it
breaks parity. Memory is governed by **total** parameters, not active ones.

Links:

- Paper: [MiniMax-01: Scaling Foundation Models with Lightning Attention (arXiv:2501.08313)](https://arxiv.org/abs/2501.08313)
- HF docs: [transformers/model_doc/minimax](https://huggingface.co/docs/transformers/model_doc/minimax)

See also [minimax_m2.md](minimax_m2.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `minimax-text-01` | [`MiniMaxAI/MiniMax-Text-01-hf`](https://huggingface.co/MiniMaxAI/MiniMax-Text-01-hf) |

## API

### `MiniMaxModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `200064` | token vocabulary size |
| `embed_dim` | `6144` | model width |
| `mlp_dim` | `9216` | MLP inner width |
| `num_layers` | `80` | decoder blocks |
| `num_heads` | `64` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `num_experts` | `32` | expert count |
| `num_experts_per_tok` | `2` | experts routed per token |
| `layer_types` | `None` |  |
| `block_size` | `256` |  |
| `full_attn_alpha` | `1.0` |  |
| `full_attn_beta` | `1.0` |  |
| `linear_attn_alpha` | `1.0` |  |
| `linear_attn_beta` | `1.0` |  |
| `mlp_alpha` | `1.0` |  |
| `mlp_beta` | `1.0` |  |
| `partial_rotary_factor` | `1.0` | fraction of each head that gets rotated |
| `rope_theta` | `10000000.0` | rotary base frequency |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `MiniMaxTextGenerate`

`MiniMaxModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `MiniMaxModel`.

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

### `MiniMaxTokenizer`

Tokenizer on the `tokenizers` backend.

```python
MiniMaxTokenizer(hf_id=None, vocab_file=None, merges_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull the tokenizer files from |
| `vocab_file` | `None` | explicit path to the vocab file |
| `merges_file` | `None` | explicit path to the BPE merges file |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string or a list of strings (a batch). This tokenizer has no chat
template, so pass a prompt you have already rendered rather than a message list.
Decode with `.decode(ids)` for one sequence or `.batch_decode(ids)` for a batch.

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.minimax import MiniMaxTextGenerate, MiniMaxTokenizer

model = MiniMaxTextGenerate.from_weights("minimax-text-01")
tokenizer = MiniMaxTokenizer.from_weights("minimax-text-01")

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
from zeromodels.models.minimax import MiniMaxModel

backbone = MiniMaxModel.from_weights("minimax-text-01")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = MiniMaxTextGenerate.from_weights("hf:MiniMaxAI/MiniMax-Text-01-hf")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = MiniMaxTextGenerate.from_weights(
    "minimax-text-01", quantization="int8", load_dtype="bfloat16"
)
```
