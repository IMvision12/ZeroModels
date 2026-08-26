# Qwen2-MoE

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> plus the Keras weights, sharded as
<code>model.weights.json</code> + <code>model_*.weights.h5</code>). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The Mixture-of-Experts variant of Qwen2, ported to pure Keras 3. The attention
stack matches Qwen2 (biased QKV, grouped-query attention); each MLP becomes a
softmax-routed expert bank with an always-on shared expert.

Memory is governed by **total** parameters, not active ones.

Links:

- HF collection: [Qwen2-MoE](https://huggingface.co/collections/zeromodels/qwen2-moe-6a7f9afdca48ae23da66c04e)
- Paper: [Qwen2 Technical Report (arXiv:2407.10671)](https://arxiv.org/abs/2407.10671)
- HF docs: [transformers/model_doc/qwen2_moe](https://huggingface.co/docs/transformers/model_doc/qwen2_moe)

See also [qwen2.md](qwen2.md), [qwen3_moe.md](qwen3_moe.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. The `-instruct` / `-chat`
suffix marks instruction-tuned checkpoints (use the chat template); bare names are base
models.

| Variant | Hub |
|---|---|
| `qwen1.5-moe-a2.7b` | [`zeromodels/qwen1.5-moe-a2.7b`](https://huggingface.co/zeromodels/qwen1.5-moe-a2.7b) |
| `qwen1.5-moe-a2.7b-chat` | [`zeromodels/qwen1.5-moe-a2.7b-chat`](https://huggingface.co/zeromodels/qwen1.5-moe-a2.7b-chat) |
| `qwen2-57b-a14b` | [`zeromodels/qwen2-57b-a14b`](https://huggingface.co/zeromodels/qwen2-57b-a14b) |
| `qwen2-57b-a14b-instruct` | [`zeromodels/qwen2-57b-a14b-instruct`](https://huggingface.co/zeromodels/qwen2-57b-a14b-instruct) |

## API

### `Qwen2MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `16` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `mlp_dim` | `5632` | MLP inner width |
| `num_experts` | `60` | expert count |
| `num_experts_per_tok` | `4` | experts routed per token |
| `moe_mlp_dim` | `1408` | per-expert inner width |
| `shared_mlp_dim` | `5632` |  |
| `norm_topk_prob` | `False` |  |
| `decoder_sparse_step` | `1` |  |
| `mlp_only_layers` | `()` |  |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Qwen2MoeTextGenerate`

`Qwen2MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen2MoeModel`.

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

### `Qwen2MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen2MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from zeromodels.models.qwen2_moe import Qwen2MoeTextGenerate, Qwen2MoeTokenizer

model = Qwen2MoeTextGenerate.from_weights("zeromodels/qwen1.5-moe-a2.7b-chat")
tokenizer = Qwen2MoeTokenizer.from_weights("zeromodels/qwen1.5-moe-a2.7b-chat")

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
from zeromodels.models.qwen2_moe import Qwen2MoeModel

backbone = Qwen2MoeModel.from_weights("zeromodels/qwen1.5-moe-a2.7b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen2MoeTextGenerate.from_weights("hf:Qwen/Qwen1.5-MoE-A2.7B")
```

### Lower memory

MoE memory is governed by total (not active) parameters, so the 57B-A14B checkpoint
needs quantization to fit comfortably on a single 80GB GPU. See
[quantization.md](quantization.md):

```python
model = Qwen2MoeTextGenerate.from_weights(
    "zeromodels/qwen2-57b-a14b-instruct",
    quantization="int8",
)
```
