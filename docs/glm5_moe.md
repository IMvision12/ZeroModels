# GLM-5 (GLM-5 MoE)

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Zhipu's GLM-5 / GLM-5.1 / GLM-5.2 Mixture-of-Experts LLM, ported to pure Keras 3.
It combines Multi-head Latent Attention (MLA) with DeepSeek-style sparse experts
and adds DSA (DeepSeek Sparse Attention) on top.

Two parity notes from the port: the rope layout is **interleaved** (not NeoX),
and matching the reference needs a 4D causal mask, since the DSA path otherwise
runs bidirectional. The MLA bottleneck norms (`q_a_layernorm` / `kv_a_layernorm`)
always use eps=1e-6 regardless of `norm_eps`.

Memory is governed by **total** parameters, not active ones.

Links:

- Paper: [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763)

See also [glm4.md](glm4.md), [glm4_moe.md](glm4_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `glm5` | [`zai-org/GLM-5`](https://huggingface.co/zai-org/GLM-5) |
| `glm5_1` | [`zai-org/GLM-5.1`](https://huggingface.co/zai-org/GLM-5.1) |
| `glm5_2` | [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) |

## API

### `Glm5MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `154880` | token vocabulary size |
| `embed_dim` | `6144` | model width |
| `num_layers` | `78` | decoder blocks |
| `num_heads` | `64` | query heads |
| `mlp_dim` | `12288` | MLP inner width |
| `moe_mlp_dim` | `2048` | per-expert inner width |
| `num_experts` | `256` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `n_shared_experts` | `1` |  |
| `n_group` | `1` |  |
| `topk_group` | `1` |  |
| `norm_topk_prob` | `True` |  |
| `routed_scaling_factor` | `2.5` |  |
| `first_k_dense` | `3` |  |
| `q_lora_rank` | `2048` |  |
| `kv_lora_rank` | `512` |  |
| `qk_nope_head_dim` | `192` |  |
| `qk_rope_head_dim` | `64` |  |
| `v_head_dim` | `256` |  |
| `index_n_heads` | `32` |  |
| `index_head_dim` | `128` |  |
| `index_topk` | `2048` |  |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `attention_bias` | `False` | add bias terms to the qkv projections |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Glm5MoeTextGenerate`

`Glm5MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Glm5MoeModel`.

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

### `Glm5MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Glm5MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.glm5_moe import Glm5MoeTextGenerate, Glm5MoeTokenizer

model = Glm5MoeTextGenerate.from_weights("glm5")
tokenizer = Glm5MoeTokenizer.from_weights("glm5")

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
from zeromodels.models.glm5_moe import Glm5MoeModel

backbone = Glm5MoeModel.from_weights("glm5")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Glm5MoeTextGenerate.from_weights("hf:zai-org/GLM-5")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Glm5MoeTextGenerate.from_weights(
    "glm5", quantization="int8", load_dtype="bfloat16"
)
```
