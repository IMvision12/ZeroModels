# BART

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> + <code>tokenizer.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

BART (Lewis et al. 2019) in pure Keras 3: a denoising sequence-to-sequence transformer with a
bidirectional encoder (like BERT) and an autoregressive decoder (like GPT), pretrained to
reconstruct corrupted text. It is strong at summarization, translation, and other text-to-text
tasks. The blocks are post-norm (`x = LayerNorm(x + sublayer(x))`), use learned absolute position
embeddings (offset by 2) plus a `layernorm_embedding` after the token and position embeddings, GELU
feed-forwards, and a token embedding shared across the encoder, the decoder, and the LM head. One
implementation runs unmodified on TensorFlow / Torch / JAX, bit-exact with Hugging Face on real
checkpoints.

- Paper: [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (arXiv:1910.13461)](https://arxiv.org/abs/1910.13461)
- HF docs: [transformers/model_doc/bart](https://huggingface.co/docs/transformers/model_doc/bart)

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub | layers / dim | Task |
|---|---|---|---|
| `bart_base` | [`zeromodels/bart_base`](https://huggingface.co/zeromodels/bart_base) | 6 / 768 | base seq2seq |
| `bart_large` | [`zeromodels/bart_large`](https://huggingface.co/zeromodels/bart_large) | 12 / 1024 | base seq2seq |
| `bart_large_cnn` | [`zeromodels/bart_large_cnn`](https://huggingface.co/zeromodels/bart_large_cnn) | 12 / 1024 | summarization (CNN / DailyMail) |
| `bart_large_xsum` | [`zeromodels/bart_large_xsum`](https://huggingface.co/zeromodels/bart_large_xsum) | 12 / 1024 | summarization (XSum) |

## API

All classes are functional `BaseModel`s; the heads extend the backbone. The one hosted
`model.weights.h5` (declaring `BartConditionalGenerate`) carries the full encoder-decoder plus the
tied LM head; every other class loads its own subset by path suffix (`BartConditionalGenerate` ties
the LM head, so it adds no projection weight; the classification / QA heads are randomly initialized,
ready for fine-tuning). Fine-tuned task checkpoints (e.g. `facebook/bart-large-mnli`) load via the
`hf:` prefix.

### `BartModel`

The encoder-decoder backbone. Takes `input_ids` / `attention_mask` / `decoder_input_ids` and returns
`{"last_hidden_state": (B, T, d), "encoder_last_hidden_state": (B, S, d)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `50265` | token vocabulary size (byte-level BPE, shared with RoBERTa) |
| `hidden_dim` | `1024` | model width (`d_model`) |
| `encoder_num_layers` / `decoder_num_layers` | `12` / `12` | transformer blocks per stack |
| `encoder_attention_heads` / `decoder_attention_heads` | `16` / `16` | attention heads |
| `encoder_ffn_dim` / `decoder_ffn_dim` | `4096` / `4096` | feed-forward inner width |
| `max_position_embeddings` | `1024` | learned position table size (plus the +2 offset) |
| `activation_function` | `"gelu"` | feed-forward activation |
| `scale_embedding` | `False` | scale the token embedding by `sqrt(hidden_dim)` |
| `layer_norm_eps` | `1e-5` | LayerNorm epsilon |
| `num_labels` | `3` | class count for `BartSequenceClassify` |
| `pad_token_id` / `bos_token_id` / `eos_token_id` / `decoder_start_token_id` | `1` / `0` / `2` / `2` | special tokens (BART starts decoding from `</s>`) |

### `BartConditionalGenerate`

`BartModel` plus the tied LM head and a learned `final_logits_bias`. Returns
`{"logits": (B, T, vocab_size), ...}` and adds `.generate()` (runs the encoder once, then decodes with
a fixed KV cache, cross-attending to the frozen encoder output at each step). Takes the same
constructor arguments as `BartModel`.

```python
generate(
    encoder_inputs,  # {"input_ids", "attention_mask"} dict, or an input_ids tensor
    decoder_input_ids,  # start tokens, e.g. [[model.decoder_start_token_id]] per row
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `encoder_inputs` | required | the source: a tokenizer dict (`input_ids` + `attention_mask`) or an `input_ids` tensor |
| `decoder_input_ids` | required | `(B, 1)` start tokens (`decoder_start_token_id = </s>`) |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token; pass `tokenizer.eos_token_id` to stop at `</s>` |
| `sampler` | `None` | sampling strategy; greedy when unset |

A padded (variable-length) source batch is handled correctly: the encoder padding mask is applied to
the decoder's cross-attention at every step.

### Other classes

| Class | HF equivalent | Output |
|---|---|---|
| `BartSequenceClassify` | `BartForSequenceClassification` | `(B, num_labels)` (pooled at the last `</s>`) |
| `BartQnA` | `BartForQuestionAnswering` | `{"start_logits": (B, T), "end_logits": (B, T)}` |

### `BartTokenizer`

Byte-level BPE on the `tokenizers` (Rust) backend, the same family as RoBERTa: the `<s> A </s>`
(single) and `<s> A </s></s> B </s>` (pair) post-processing is baked into `tokenizer.json`. Returns
`input_ids` / `attention_mask` (BART has no token-type ids); the decoder ids are produced by the
model, not the tokenizer.

```python
BartTokenizer(variant="bart_large", tokenizer_file=None, max_seq_len=1024)
```

## End-to-end example

### Summarization

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.bart import BartConditionalGenerate, BartTokenizer

model = BartConditionalGenerate.from_weights("zeromodels/bart_large_cnn")
tokenizer = BartTokenizer.from_weights("zeromodels/bart_large_cnn")

article = (
    "The tower is 324 metres tall, about the same height as an 81-storey building..."
)
inputs = tokenizer(article)
ids = model.generate(
    inputs,
    [[model.decoder_start_token_id]],
    max_new_tokens=142,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

### Encoder features

```python
from zeromodels.models.bart import BartModel, BartTokenizer

model = BartModel.from_weights("zeromodels/bart_large")
tokenizer = BartTokenizer.from_weights("zeromodels/bart_large")

inputs = tokenizer("The quick brown fox.")
out = model(
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": model.shift_right(inputs["input_ids"]),
    }
)
out["encoder_last_hidden_state"]  # (1, S, 1024)
```

### Zero-shot / NLI classification (fine-tune via `hf:`)

```python
from zeromodels.models.bart import BartSequenceClassify

# facebook/bart-large-mnli is a fine-tuned classifier; load it on the fly with hf:
model = BartSequenceClassify.from_weights("hf:facebook/bart-large-mnli")
```

### Loading from the Hub

```python
model = BartConditionalGenerate.from_weights("hf:facebook/bart-large-cnn")
```

## Architecture notes

- **Post-norm** blocks (`x = LayerNorm(x + sublayer(x))`), unlike T5's pre-norm. Each of the encoder
  and decoder embeds tokens, adds a **learned absolute position** (a table offset by 2), then applies
  a `layernorm_embedding`.
- **Standard attention**: `1/sqrt(head_dim)` scaling with biases on the q / k / v / output
  projections; GELU (exact) feed-forwards.
- The token embedding is **shared** across the encoder, the decoder, and the LM head; the LM head adds
  only a learned `final_logits_bias` (`logits = hidden @ shared.T + bias`). The decoder starts from
  `</s>` (`decoder_start_token_id = eos = 2`); `BartModel.shift_right` builds decoder inputs by
  right-shifting the labels.
- **One shared checkpoint**: the hosted `model.weights.h5` declares `BartConditionalGenerate`; the
  other heads copy the backbone out of it by path suffix (task heads start random). Fine-tuned task
  checkpoints load via the `hf:` prefix.

## Parity

Bit-exact with Hugging Face `transformers` (eager, float32): all four heads match to `< 4e-5` max-abs
difference (`BartModel` encoder / decoder hidden states, `BartConditionalGenerate` logits, and the
`BartSequenceClassify` / `BartQnA` heads), and the cached decode path reproduces the teacher-forced
forward logits. See `convert_bart_hf_to_keras.py`.
