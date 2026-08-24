# GPT-OSS

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/kerasformers">kerasformers/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.json</code>, with the
MoE experts kept in MXFP4). Load with <code>from_weights("kerasformers/&lt;variant&gt;")</code>.
</div>

OpenAI's GPT-OSS, an open-weight mixture-of-experts decoder-only LLM, ported to pure
Keras 3. One implementation runs unmodified on **TensorFlow / Torch / JAX**. Beyond a
standard decoder it adds:

- **Mixture-of-experts feed-forward**: a top-`num_experts_per_tok` router selects
  experts whose softmax weights combine per-expert outputs, with GPT-OSS's clamped
  gated-SiLU on the interleaved gate/up halves (`(up+1) * gate*sigmoid(1.702*gate)`,
  clamp 7). The expert bank evaluates only the routed experts on single-token decode
  and the full bank on longer prefills (see MXFP4 experts below); both are exact and
  backend-agnostic.
- **Attention sinks**: a learned per-head logit is appended to the attention scores
  before softmax and dropped afterward, letting a head attend to "nothing".
- **Alternating attention**: even layers use a `sliding_window` (128) local span;
  odd layers use full causal attention.
- **YaRN rotary** scaling (factor 32, beta_fast 32, beta_slow 1, original context
  4096) with the mscale cos/sin factor.
- **MXFP4 experts**: the experts stay packed in MXFP4 (uint8 nibble blocks + e8m0
  scales) exactly as OpenAI ships them and are dequantized on the fly in the expert
  layer's `call`, keeping the official ~13 GB / ~65 GB footprint instead of a ~4x
  larger bf16 expansion. On single-token **decode** only the top-`num_experts_per_tok`
  routed experts are dequantized (~8x less than the full 32/128-expert bank, and
  faster); longer **prefills** dequantize the whole bank once (cheaper when many
  tokens share experts). Both paths give identical results. The dequant is a
  backend-agnostic `keras.ops` port of HF's `convert_moe_packed_tensors`, so it runs
  on every backend including CPU. The hosted repos declare this in a
  `quantization_config` block, and a `Mxfp4KfQuantizer` swaps the packed experts in on
  load, so the model itself carries no mxfp4 flag. See [mxfp4](quantization_mxfp4.md)
  for the format and using it as a general `quantize_model` scheme.

Links:

- Model card: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

## Variants

Load either with `from_weights("kerasformers/<variant>")`. Upstream safetensors also
convert on the fly via `from_weights("hf:openai/<variant>")`.

| Variant | Hub | layers | experts |
|---|---|---|---|
| `gpt-oss-20b` | [`kerasformers/gpt-oss-20b`](https://huggingface.co/kerasformers/gpt-oss-20b) | 24 | 32 |
| `gpt-oss-120b` | [`kerasformers/gpt-oss-120b`](https://huggingface.co/kerasformers/gpt-oss-120b) | 36 | 128 |

## API

### `GptOssModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.
A functional model built over `{input_ids, attention_mask}`; the imperative KV-cache decode
autoregressive generation needs lives on the generation head.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `201088` | token vocabulary size |
| `embed_dim` | `2880` | model width |
| `mlp_dim` | `2880` | per-expert hidden width |
| `num_layers` | `24` | decoder blocks (36 for 120b) |
| `num_heads` | `64` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `64` | per-head width |
| `num_experts` | `32` | MoE experts (128 for 120b) |
| `num_experts_per_tok` | `4` | experts routed per token (top-k) |
| `sliding_window` | `128` | local span on the alternating (even) layers |
| `norm_eps` | `1e-5` | RMSNorm epsilon |
| `rope_theta` | `150000.0` | rotary base frequency |
| `rope_factor` | `32.0` | YaRN scaling factor |
| `rope_beta_fast` | `32.0` | YaRN beta_fast |
| `rope_beta_slow` | `1.0` | YaRN beta_slow |
| `rope_truncate` | `False` | YaRN correction-range truncation |
| `rope_original_max_pos` | `4096` | YaRN original context length |
| `attention_bias` | `True` | whether q/k/v/o projections carry a bias |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `GptOssTextGenerate`

`GptOssModel` plus an (untied) LM head. Returns `{"logits": (batch, seq, vocab_size),
"last_hidden_state": ...}` and adds `.generate()` (greedy, with a KV cache that
respects each layer's sliding window). Same constructor arguments as `GptOssModel`.

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

### `GptOssTokenizer`

`o200k_harmony` BPE tokenizer on the `tokenizers` backend.

```python
GptOssTokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull `tokenizer.json` from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string, a list of strings (a batch), or a chat-message list, which is
routed through `apply_chat_template` automatically.

`decode_message(ids)` turns a generated turn into a Harmony-aware chat dict,
`{"role": "assistant", "content": <final answer>, "thinking": <reasoning>}`: it pulls
the `final` channel into `content` and the `analysis` channel into `thinking` (added
only when present), so the reasoning is separated out instead of munged into the
answer. `parse_harmony(ids)` returns the raw `(final, reasoning)` pair.

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from kerasformers.models.gpt_oss import GptOssTextGenerate, GptOssTokenizer

model = GptOssTextGenerate.from_weights("kerasformers/gpt-oss-20b")
tokenizer = GptOssTokenizer.from_weights("kerasformers/gpt-oss-20b")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=256)  # reasoning model: leave room

# GPT-OSS emits a Harmony turn (analysis + final channels); decode_message splits them.
message = tokenizer.decode_message(outputs[0])
print(message["content"])  # the answer
print(message.get("thinking"))  # the chain-of-thought, when present
```

### Batch

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
from kerasformers.models.gpt_oss import GptOssModel

backbone = GptOssModel.from_weights("kerasformers/gpt-oss-20b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 2880)
```

### Loading from the Hub

```python
model = GptOssTextGenerate.from_weights("hf:openai/gpt-oss-20b")
```

### Precision

GPT-OSS loads in **bfloat16 by default** (the dense weights are bf16, the experts stay
4-bit MXFP4), matching how OpenAI ships it, so `from_weights` keeps the native
~2 bytes/param footprint instead of upcasting to fp32. To force fp32 (double the dense
memory), pass `load_dtype="float32"`:

```python
model = GptOssTextGenerate.from_weights("kerasformers/gpt-oss-20b")  # bf16 (default)
model = GptOssTextGenerate.from_weights(
    "kerasformers/gpt-oss-20b", load_dtype="float32"
)  # fp32
```

The 120B checkpoint still needs a big machine (loading builds the full model), but at bf16
that is ~66 GB rather than ~130 GB: an 80 GB GPU fits it under the default policy.
