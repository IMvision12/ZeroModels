# GLM-4.7-Flash (GLM-4 MoE Lite)

<div class="kf-note kf-note--weights">
<b>Weights:</b> preconverted Keras weights are hosted at
<a href="https://huggingface.co/zeromodels/glm-4.7-flash">zeromodels/glm-4.7-flash</a>.
Load the model and tokenizer with
<code>from_weights("zeromodels/glm-4.7-flash")</code>. See
<a href="../loading_weights/">Loading Weights</a>.
</div>

GLM-4.7-Flash is a compact Mixture-of-Experts text model implemented in pure
Keras 3. Its ZeroModels architecture name is `glm4_moe_lite`. The decoder uses
DeepSeek-V3-style Multi-head Latent Attention (MLA) and aux-loss-free
DeepSeekMoE routing. Unlike GLM-5, it has no DSA sparse-attention indexer.

All experts remain resident in memory even though only four routed experts are
active for each token.

Links:

- Model card: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- Keras weights: [zeromodels/glm-4.7-flash](https://huggingface.co/zeromodels/glm-4.7-flash)
- Related architecture: [GLM-4.5 (GLM-4 MoE)](glm4_moe.md)
- Related architecture: [GLM-5 MoE](glm5_moe.md)

## Variant

GLM-4 MoE Lite currently has one hosted variant.

| Variant | Hosted | Upstream |
|---|---|---|
| `glm-4.7-flash` | `zeromodels/glm-4.7-flash` | [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash) |

## API

The family exports these public classes:

- `Glm4MoeLiteConfig`
- `Glm4MoeLiteModel`
- `Glm4MoeLiteTextGenerate`
- `Glm4MoeLiteTokenizer`

### `Glm4MoeLiteModel`

The decoder backbone returns
`{"last_hidden_state": (batch, sequence, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `154880` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `num_layers` | `47` | decoder blocks |
| `num_heads` | `20` | query heads |
| `mlp_dim` | `10240` | dense-layer SwiGLU width |
| `moe_mlp_dim` | `1536` | per-expert SwiGLU width |
| `num_experts` | `64` | routed expert count |
| `num_experts_per_tok` | `4` | routed experts selected per token |
| `n_shared_experts` | `1` | shared expert count |
| `n_group` | `1` | expert routing group count |
| `topk_group` | `1` | routing groups retained per token |
| `norm_topk_prob` | `True` | renormalize selected routing weights |
| `routed_scaling_factor` | `1.8` | selected-expert output scale |
| `first_k_dense` | `1` | leading dense decoder layers |
| `q_lora_rank` | `768` | query projection bottleneck |
| `kv_lora_rank` | `512` | key/value projection bottleneck |
| `qk_nope_head_dim` | `192` | non-rotary query/key width per head |
| `qk_rope_head_dim` | `64` | rotary query/key width per head |
| `v_head_dim` | `256` | value width per head |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `rope_scaling` | `None` | optional Hugging Face YaRN scaling dictionary |
| `norm_eps` | `1e-5` | RMSNorm epsilon |
| `max_position_embeddings` | `202752` | maximum configured context length |
| `tie_embeddings` | `False` | reuse token embeddings for the LM head |

### `Glm4MoeLiteTextGenerate`

`Glm4MoeLiteModel` plus the language-model head. It returns
`{"logits": (batch, sequence, vocab_size)}` and adds `.generate()`.

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

When `eos_token_id` is omitted, the generation class recognizes the model's
three end markers: `154820`, `154827`, and `154829`.

### `Glm4MoeLiteTokenizer`

The BPE tokenizer uses the `tokenizers` backend and loads `tokenizer.json` from
the Hub repository unless a local file is provided.

```python
Glm4MoeLiteTokenizer(hf_id=None, tokenizer_file=None)
```

Calling it with a string or list of strings returns token IDs. Decode one
sequence with `.decode(ids)` or a batch with `.batch_decode(ids)`.

## End-to-end example

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.glm4_moe_lite import (
    Glm4MoeLiteTextGenerate,
    Glm4MoeLiteTokenizer,
)

weights = "zeromodels/glm-4.7-flash"
model = Glm4MoeLiteTextGenerate.from_weights(
    weights,
    load_dtype="bfloat16",
)
tokenizer = Glm4MoeLiteTokenizer.from_weights(weights)

inputs = tokenizer("Explain mixture-of-experts routing in one sentence.")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
```

The upstream model is large despite its sparse activation. Make sure the
selected backend has enough host and accelerator memory before converting it.

## Backbone example

```python
from zeromodels.models.glm4_moe_lite import Glm4MoeLiteModel

backbone = Glm4MoeLiteModel.from_weights(
    "zeromodels/glm-4.7-flash",
    load_dtype="bfloat16",
)
hidden = backbone(inputs)["last_hidden_state"]
```
