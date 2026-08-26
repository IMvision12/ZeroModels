# DeepSeek-V3

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

DeepSeek's third-generation Mixture-of-Experts LLM, ported to pure Keras 3. It
keeps V2's Multi-head Latent Attention (MLA) and DeepSeekMoE, and adds
node-limited routing (`n_group` / `topk_group`) with an auxiliary-loss-free
load-balancing bias plus `routed_scaling_factor` on the gate.

The hub checkpoints ship in block-FP8 and are dequantized during conversion.
Memory is governed by **total** parameters, not active ones.

Links:

- Paper: [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- HF docs: [transformers/model_doc/deepseek_v3](https://huggingface.co/docs/transformers/model_doc/deepseek_v3)

See also [deepseek_v2.md](deepseek_v2.md), [deepseek_v4.md](deepseek_v4.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `deepseek-v3` | [`deepseek-ai/DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| `deepseek-v3-0324` | [`deepseek-ai/DeepSeek-V3-0324`](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| `deepseek-v3.1` | [`deepseek-ai/DeepSeek-V3.1`](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) |
| `deepseek-r1` | [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1) |

## API

### `DeepseekV3Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `129280` | token vocabulary size |
| `embed_dim` | `7168` | model width |
| `num_layers` | `61` | decoder blocks |
| `num_heads` | `128` | query heads |
| `mlp_dim` | `18432` | MLP inner width |
| `moe_mlp_dim` | `2048` | per-expert inner width |
| `num_experts` | `256` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `n_shared_experts` | `1` |  |
| `n_group` | `8` |  |
| `topk_group` | `4` |  |
| `norm_topk_prob` | `True` |  |
| `routed_scaling_factor` | `2.5` |  |
| `first_k_dense` | `3` |  |
| `q_lora_rank` | `1536` |  |
| `kv_lora_rank` | `512` |  |
| `qk_nope_head_dim` | `128` |  |
| `qk_rope_head_dim` | `64` |  |
| `v_head_dim` | `128` |  |
| `rope_theta` | `10000.0` | rotary base frequency |
| `rope_scaling` | `None` |  |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `max_position_embeddings` | `163840` | maximum context length |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `DeepseekV3TextGenerate`

`DeepseekV3Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `DeepseekV3Model`.

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

### `DeepseekV3Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
DeepseekV3Tokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.deepseek_v3 import DeepseekV3TextGenerate, DeepseekV3Tokenizer

model = DeepseekV3TextGenerate.from_weights("deepseek-v3")
tokenizer = DeepseekV3Tokenizer.from_weights("deepseek-v3")

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
from zeromodels.models.deepseek_v3 import DeepseekV3Model

backbone = DeepseekV3Model.from_weights("deepseek-v3")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = DeepseekV3TextGenerate.from_weights("hf:deepseek-ai/DeepSeek-V3")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = DeepseekV3TextGenerate.from_weights(
    "deepseek-v3", quantization="int8", load_dtype="bfloat16"
)
```
