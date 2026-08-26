# Qwen3-MoE

<div class="kf-note kf-note--weights">
<b>Weights:</b> the 30B-A3B checkpoints are hosted as pretrained Keras weights on Hugging
Face under <a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> and <code>tokenizer.json</code> plus a sharded
<code>model.weights.json</code> + shards). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>. The 235B-A22B is not re-hosted;
load it on the fly with the <code>hf:</code> prefix.
</div>

The Mixture-of-Experts variant of Qwen3, ported to pure Keras 3. It keeps Qwen3's
QK-RMSNorm attention and bias-free projections, replacing each MLP with a
softmax-routed expert bank.

Memory is governed by **total** parameters, not active ones.

Links:

- HF collection: [Qwen3-MoE](https://huggingface.co/collections/zeromodels/qwen3-moe-6a7f9b1eacaba9aba25a1d63)
- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3_moe](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)

See also [qwen3.md](qwen3.md), [qwen2_moe.md](qwen2_moe.md).

## Variants

The 30B-A3B sizes are hosted, preconverted, under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-base` suffix marks the base checkpoint and
`-2507` the July 2025 refresh. Qwen3-MoE is Apache 2.0.

| Variant | Hub |
|---|---|
| `qwen3-30b-a3b` | [`zeromodels/qwen3-30b-a3b`](https://huggingface.co/zeromodels/qwen3-30b-a3b) |
| `qwen3-30b-a3b-base` | [`zeromodels/qwen3-30b-a3b-base`](https://huggingface.co/zeromodels/qwen3-30b-a3b-base) |
| `qwen3-30b-a3b-instruct-2507` | [`zeromodels/qwen3-30b-a3b-instruct-2507`](https://huggingface.co/zeromodels/qwen3-30b-a3b-instruct-2507) |
| `qwen3-30b-a3b-thinking-2507` | [`zeromodels/qwen3-30b-a3b-thinking-2507`](https://huggingface.co/zeromodels/qwen3-30b-a3b-thinking-2507) |

The **235B-A22B** flagship is not re-hosted (too large); load it on the fly via the `hf:`
prefix, e.g. `from_weights("hf:Qwen/Qwen3-235B-A22B")` (pass `cache_converted=True` to keep
the converted result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `4` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_experts` | `128` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `moe_mlp_dim` | `768` | per-expert inner width |
| `norm_topk_prob` | `True` |  |
| `decoder_sparse_step` | `1` |  |
| `mlp_only_layers` | `()` |  |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Qwen3MoeTextGenerate`

`Qwen3MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3MoeModel`.

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

### `Qwen3MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.qwen3_moe import Qwen3MoeTextGenerate, Qwen3MoeTokenizer

model = Qwen3MoeTextGenerate.from_weights("zeromodels/qwen3-30b-a3b")
tokenizer = Qwen3MoeTokenizer.from_weights("zeromodels/qwen3-30b-a3b")

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
from zeromodels.models.qwen3_moe import Qwen3MoeModel

backbone = Qwen3MoeModel.from_weights("zeromodels/qwen3-30b-a3b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3MoeTextGenerate.from_weights("hf:Qwen/Qwen3-30B-A3B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3MoeTextGenerate.from_weights(
    "zeromodels/qwen3-30b-a3b", quantization="int8", load_dtype="bfloat16"
)
```
