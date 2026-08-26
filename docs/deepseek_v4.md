# DeepSeek-V4

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

DeepSeek's fourth-generation Mixture-of-Experts LLM, ported to pure Keras 3. It
continues the V2/V3 line: Multi-head Latent Attention (MLA) with low-rank query
and key/value bottlenecks, plus DeepSeekMoE routed and shared experts.

Memory is governed by **total** parameters, not active ones.


See also [deepseek_v2.md](deepseek_v2.md), [deepseek_v3.md](deepseek_v3.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `deepseek-v4-flash` | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| `deepseek-v4-flash-base` | [`deepseek-ai/DeepSeek-V4-Flash-Base`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) |
| `deepseek-v4-pro` | [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| `deepseek-v4-pro-base` | [`deepseek-ai/DeepSeek-V4-Pro-Base`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) |

## API

### `DeepseekV4Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `129280` | token vocabulary size |
| `embed_dim` | `4096` | model width |
| `num_layers` | `43` | decoder blocks |
| `num_heads` | `64` | query heads |
| `head_dim` | `512` | per-head width |
| `q_lora_rank` | `1024` |  |
| `qk_rope_head_dim` | `64` |  |
| `o_groups` | `8` |  |
| `o_lora_rank` | `1024` |  |
| `layer_types` | `None` |  |
| `mlp_layer_types` | `None` |  |
| `num_experts` | `256` | expert count |
| `num_experts_per_tok` | `6` | experts routed per token |
| `moe_mlp_dim` | `2048` | per-expert inner width |
| `routed_scaling_factor` | `1.5` |  |
| `swiglu_limit` | `10.0` |  |
| `sliding_window` | `128` | local attention span |
| `compress_rate_csa` | `4` |  |
| `compress_rate_hca` | `128` |  |
| `index_n_heads` | `64` |  |
| `index_head_dim` | `128` |  |
| `index_topk` | `512` |  |
| `hc_mult` | `4` |  |
| `hc_sinkhorn_iters` | `20` |  |
| `hc_eps` | `1e-06` |  |
| `rope_theta` | `10000.0` | rotary base frequency |
| `compress_rope_theta` | `160000.0` |  |
| `rope_scaling` | `{'type': 'yarn', 'factor': 16, 'beta_fast': 32, 'beta_slow': 1, 'original_max_position_embeddings': 65536}` |  |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `DeepseekV4TextGenerate`

`DeepseekV4Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `DeepseekV4Model`.

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

### `DeepseekV4Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
DeepseekV4Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.deepseek_v4 import DeepseekV4TextGenerate, DeepseekV4Tokenizer

model = DeepseekV4TextGenerate.from_weights("deepseek-v4-flash")
tokenizer = DeepseekV4Tokenizer.from_weights("deepseek-v4-flash")

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
from zeromodels.models.deepseek_v4 import DeepseekV4Model

backbone = DeepseekV4Model.from_weights("deepseek-v4-flash")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = DeepseekV4TextGenerate.from_weights("hf:deepseek-ai/DeepSeek-V4-Flash")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = DeepseekV4TextGenerate.from_weights(
    "deepseek-v4-flash", quantization="int8", load_dtype="bfloat16"
)
```
