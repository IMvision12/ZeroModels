# Llama 4

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Meta's Mixture-of-Experts Llama generation, ported to pure Keras 3. This family
covers the **text decoder** of Scout and Maverick (the `language_model` of the
released multimodal checkpoints); the vision tower is not part of this port.

What is unusual about the block:

- **Sparse MoE**: experts are interleaved every `interleave_moe_layer_step`
  layers, routed with a sigmoid gate on the scaled input, alongside a shared
  dense MLP of width `dense_mlp_dim`. Scout routes 1 of 16 experts per token,
  Maverick 1 of 128.
- **NoPE layers**: every `no_rope_layer_interval`-th layer drops rotary
  embeddings entirely, which is what lets the model generalize to very long
  context.
- **Chunked attention**: non-NoPE layers attend within an
  `attention_chunk_size` window rather than the full sequence.
- **Attention temperature tuning**: logits are rescaled by a `floor_scale` /
  `attn_scale` schedule as position grows.

Memory is governed by **total** parameters, not active ones: all experts stay
resident. Maverick's 128 experts need capacity for the full weight count even
though only 17B are active per token.

Links:

- Paper: [The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes (arXiv:2601.11659)](https://arxiv.org/abs/2601.11659)
- HF docs: [transformers/model_doc/llama4](https://huggingface.co/docs/transformers/model_doc/llama4)

See also [llama.md](llama.md), [llama2.md](llama2.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `llama4-scout-17b-16e` | [`meta-llama/Llama-4-Scout-17B-16E`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) |
| `llama4-scout-17b-16e-instruct` | [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| `llama4-maverick-17b-128e` | [`meta-llama/Llama-4-Maverick-17B-128E`](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) |
| `llama4-maverick-17b-128e-instruct` | [`meta-llama/Llama-4-Maverick-17B-128E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |

## API

### `Llama4Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `202048` | token vocabulary size |
| `embed_dim` | `5120` | model width |
| `mlp_dim` | `8192` | MLP inner width |
| `dense_mlp_dim` | `16384` | width of the shared dense MLP that runs alongside the experts |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `40` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `num_experts` | `16` | expert count |
| `num_experts_per_tok` | `1` | experts routed per token |
| `interleave_moe_layer_step` | `1` | place an MoE block every N layers |
| `no_rope_layer_interval` | `4` | every N-th layer uses no rotary embeddings (NoPE) |
| `attention_chunk_size` | `8192` | chunked local attention span on non-NoPE layers |
| `use_qk_norm` | `True` | RMS-normalize queries and keys before attention |
| `attn_temperature_tuning` | `True` | rescale attention logits as position grows |
| `floor_scale` | `8192.0` | position floor for attention temperature tuning |
| `attn_scale` | `0.1` | slope for attention temperature tuning |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `500000.0` | rotary base frequency |
| `rope_factor` | `16.0` |  |
| `rope_low_freq_factor` | `1.0` |  |
| `rope_high_freq_factor` | `1.0` |  |
| `rope_original_max_pos` | `8192` |  |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Llama4TextGenerate`

`Llama4Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Llama4Model`.

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

### `Llama4Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Llama4Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.llama4 import Llama4TextGenerate, Llama4Tokenizer

model = Llama4TextGenerate.from_weights("llama4-scout-17b-16e")
tokenizer = Llama4Tokenizer.from_weights("llama4-scout-17b-16e")

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
from zeromodels.models.llama4 import Llama4Model

backbone = Llama4Model.from_weights("llama4-scout-17b-16e")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Llama4TextGenerate.from_weights("hf:meta-llama/Llama-4-Scout-17B-16E")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Llama4TextGenerate.from_weights(
    "llama4-scout-17b-16e", quantization="int8", load_dtype="bfloat16"
)
```
