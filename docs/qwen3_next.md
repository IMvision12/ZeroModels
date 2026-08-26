# Qwen3-Next

<div class="kf-note kf-note--weights">
<b>Weights:</b> the 80B-A3B checkpoints are hosted as pretrained Keras weights on Hugging
Face under <a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> and <code>tokenizer.json</code> plus a sharded
<code>model.weights.json</code> + shards). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen3-Next (`Qwen3-Next-80B-A3B`), ported to pure Keras 3. A hybrid decoder:
most blocks are Gated-DeltaNet linear-attention layers, with a full-attention block
every fourth layer; both feed a sparse Mixture-of-Experts MLP (a softmax router over the
routed experts plus a sigmoid-gated shared expert). This is the actual released MoE
checkpoint; [qwen3_5.md](qwen3_5.md) documents the dense-MLP form of the same hybrid.

Memory is governed by **total** parameters, not active ones.

Links:

- HF collection: [Qwen3-Next](https://huggingface.co/collections/zeromodels/qwen3-next-6a7e551ff86ebf2cca455ef1)
- HF docs: [transformers/model_doc/qwen3_next](https://huggingface.co/docs/transformers/model_doc/qwen3_next)

See also [qwen3_5.md](qwen3_5.md), [qwen3_moe.md](qwen3_moe.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-instruct` checkpoint is
instruction-tuned and `-thinking` the reasoning checkpoint. Qwen3-Next is Apache 2.0.

| Variant | Hub |
|---|---|
| `qwen3-next-80b-a3b-instruct` | [`zeromodels/qwen3-next-80b-a3b-instruct`](https://huggingface.co/zeromodels/qwen3-next-80b-a3b-instruct) |
| `qwen3-next-80b-a3b-thinking` | [`zeromodels/qwen3-next-80b-a3b-thinking`](https://huggingface.co/zeromodels/qwen3-next-80b-a3b-thinking) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3-Next-80B-A3B-Instruct")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3NextModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `mlp_dim` | `5120` | MLP inner width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `10000000.0` | rotary base frequency |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |
| `full_attention_interval` | `4` |  |
| `linear_conv_kernel_dim` | `4` |  |
| `linear_key_head_dim` | `128` |  |
| `linear_value_head_dim` | `128` |  |
| `linear_num_key_heads` | `16` |  |
| `linear_num_value_heads` | `32` |  |
| `num_experts` | `512` | expert count |
| `num_experts_per_tok` | `10` | experts routed per token |
| `moe_mlp_dim` | `512` | per-expert inner width |
| `shared_mlp_dim` | `512` |  |
| `norm_topk_prob` | `True` |  |
| `decoder_sparse_step` | `1` |  |
| `mlp_only_layers` | `()` |  |

### `Qwen3NextTextGenerate`

`Qwen3NextModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3NextModel`.

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

### `Qwen3NextTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3NextTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.qwen3_next import Qwen3NextTextGenerate, Qwen3NextTokenizer

model = Qwen3NextTextGenerate.from_weights("zeromodels/qwen3-next-80b-a3b-instruct")
tokenizer = Qwen3NextTokenizer.from_weights("zeromodels/qwen3-next-80b-a3b-instruct")

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
from zeromodels.models.qwen3_next import Qwen3NextModel

backbone = Qwen3NextModel.from_weights("zeromodels/qwen3-next-80b-a3b-instruct")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3NextTextGenerate.from_weights("hf:Qwen/Qwen3-Next-80B-A3B-Instruct")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3NextTextGenerate.from_weights(
    "zeromodels/qwen3-next-80b-a3b-instruct",
    quantization="int8",
    load_dtype="bfloat16",
)
```
