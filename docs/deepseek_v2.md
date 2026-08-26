# DeepSeek-V2

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

DeepSeek's Mixture-of-Experts LLM, ported to pure Keras 3. Two ideas define it:

- **Multi-head Latent Attention (MLA)**: queries and key/values are compressed
  through low-rank bottlenecks (`q_lora_rank`, `kv_lora_rank`) before attention,
  which shrinks the KV cache dramatically. Each head splits into a rotary part
  (`qk_rope_head_dim`) and a non-rotary part (`qk_nope_head_dim`).
- **DeepSeekMoE**: fine-grained routed experts plus always-on shared experts,
  with the first `first_k_dense_replace` layers left dense.

Memory is governed by **total** parameters, not active ones: every expert stays
resident.

Links:

- Paper: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
- HF docs: [transformers/model_doc/deepseek_v2](https://huggingface.co/docs/transformers/model_doc/deepseek_v2)

See also [deepseek_v3.md](deepseek_v3.md), [deepseek_v4.md](deepseek_v4.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `deepseek-v2-lite` | [`deepseek-ai/DeepSeek-V2-Lite`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) |
| `deepseek-v2-lite-chat` | [`deepseek-ai/DeepSeek-V2-Lite-Chat`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) |
| `deepseek-v2` | [`deepseek-ai/DeepSeek-V2`](https://huggingface.co/deepseek-ai/DeepSeek-V2) |
| `deepseek-v2-chat` | [`deepseek-ai/DeepSeek-V2-Chat`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) |
| `deepseek-v2.5` | [`deepseek-ai/DeepSeek-V2.5`](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) |

## API

### `DeepseekV2Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `102400` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `num_layers` | `27` | decoder blocks |
| `num_heads` | `16` | query heads |
| `mlp_dim` | `10944` | MLP inner width |
| `moe_mlp_dim` | `1408` | per-expert inner width |
| `num_experts` | `64` | expert count |
| `num_experts_per_tok` | `6` | experts routed per token |
| `n_shared_experts` | `2` |  |
| `topk_method` | `'greedy'` |  |
| `n_group` | `1` |  |
| `topk_group` | `1` |  |
| `routed_scaling_factor` | `1.0` |  |
| `first_k_dense` | `1` |  |
| `q_lora_rank` | `None` |  |
| `kv_lora_rank` | `512` |  |
| `qk_nope_head_dim` | `128` |  |
| `qk_rope_head_dim` | `64` |  |
| `v_head_dim` | `128` |  |
| `rope_theta` | `10000.0` | rotary base frequency |
| `rope_scaling` | `None` |  |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `max_position_embeddings` | `163840` | maximum context length |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `DeepseekV2TextGenerate`

`DeepseekV2Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `DeepseekV2Model`.

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

### `DeepseekV2Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
DeepseekV2Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.deepseek_v2 import DeepseekV2TextGenerate, DeepseekV2Tokenizer

model = DeepseekV2TextGenerate.from_weights("deepseek-v2-lite")
tokenizer = DeepseekV2Tokenizer.from_weights("deepseek-v2-lite")

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
from zeromodels.models.deepseek_v2 import DeepseekV2Model

backbone = DeepseekV2Model.from_weights("deepseek-v2-lite")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = DeepseekV2TextGenerate.from_weights("hf:deepseek-ai/DeepSeek-V2-Lite")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = DeepseekV2TextGenerate.from_weights(
    "deepseek-v2-lite", quantization="int8", load_dtype="bfloat16"
)
```
