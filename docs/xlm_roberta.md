# XLM-RoBERTa

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Facebook AI's XLM-RoBERTa in pure Keras 3: the multilingual variant of RoBERTa, pretrained on
2.5 TB of filtered CommonCrawl across 100 languages, with a masked-LM head plus sequence / token
classification, question-answering, and multiple-choice heads. It is **architecturally identical
to RoBERTa** (same encoder, padding-offset position ids, single token-type, `1e-5` LayerNorm, no
next-sentence head) and differs only in scale: a 250k multilingual SentencePiece vocabulary
instead of RoBERTa's 50k byte-level BPE. One implementation runs unmodified on TensorFlow / Torch
/ JAX.

- Paper: [Unsupervised Cross-lingual Representation Learning at Scale (arXiv:1911.02116)](https://arxiv.org/abs/1911.02116)
- HF docs: [transformers/model_doc/xlm-roberta](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)

See also [roberta.md](roberta.md), [bert.md](bert.md), [deberta.md](deberta.md), [electra.md](electra.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub | layers / dim |
|---|---|---|
| `xlm_roberta_base` | [`zeromodels/xlm_roberta_base`](https://huggingface.co/zeromodels/xlm_roberta_base) | 12 / 768 |
| `xlm_roberta_large` | [`zeromodels/xlm_roberta_large`](https://huggingface.co/zeromodels/xlm_roberta_large) | 24 / 1024 |

## API

### `XLMRobertaModel`

The encoder backbone plus a `tanh` pooler over the `<s>` token. Takes a dict of `input_ids` /
`attention_mask` / `token_type_ids` (all `(B, L)` int; segment ids are always `0`) and returns
`{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `250002` | multilingual token vocabulary size |
| `embed_dim` | `768` | model / hidden width |
| `num_layers` | `12` | transformer blocks |
| `num_heads` | `12` | attention heads |
| `mlp_dim` | `3072` | feed-forward inner width |
| `max_position_embeddings` | `514` | position-table size (padding-offset) |
| `type_vocab_size` | `1` | token-type embeddings |
| `hidden_act` | `"gelu"` | feed-forward activation |
| `layer_norm_eps` | `1e-5` | LayerNorm epsilon |
| `pad_token_id` | `1` | padding token id |

### Task heads

Each composes an `XLMRobertaModel` backbone and adds a head; all take the same backbone
constructor args, plus the extras below. The pretrained encoder + masked-LM head load real
weights; the classification / QA heads start randomly initialized (ready for fine-tuning) and load
trained weights from a `hf:` fine-tune.

| Class | Extra args | Output |
|---|---|---|
| `XLMRobertaMaskedLM` | | MLM logits `(B, L, vocab_size)` |
| `XLMRobertaSequenceClassify` | `num_classes` | `(B, num_classes)` |
| `XLMRobertaTokenClassify` | `num_classes` | `(B, L, num_classes)` |
| `XLMRobertaQnA` | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `XLMRobertaMultipleChoice` | `num_choices` | `(B, num_choices)` |

### `XLMRobertaTokenizer`

SentencePiece tokenizer on the `tokenizers` (Rust) backend: reads `sentencepiece.bpe.model`,
applies the fairseq id offset (`<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, every piece shifted by
`+1`, `<mask>` last), and reuses the model's `Precompiled` normalizer + `▁` metaspace with
`<s> A </s>` / `<s> A </s> </s> B </s>` post-processing.

```python
XLMRobertaTokenizer(variant="xlm_roberta_base", tokenizer_file=None, max_seq_len=512)
```

Calling it accepts a string, a list of strings, or a sentence pair (`text_pair=`), and returns the
`input_ids` / `attention_mask` / `token_type_ids` dict the models consume.

## End-to-end example

### Fill-mask (multilingual)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.xlm_roberta import XLMRobertaMaskedLM, XLMRobertaTokenizer

mlm = XLMRobertaMaskedLM.from_weights("zeromodels/xlm_roberta_base")
tokenizer = XLMRobertaTokenizer.from_weights("zeromodels/xlm_roberta_base")

inputs = tokenizer("La capitale de la France est <mask>.")
logits = mlm(inputs)  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))  # -> "Paris"
```

### Backbone features

```python
from zeromodels.models.xlm_roberta import XLMRobertaModel, XLMRobertaTokenizer

model = XLMRobertaModel.from_weights("zeromodels/xlm_roberta_base")
tokenizer = XLMRobertaTokenizer.from_weights("zeromodels/xlm_roberta_base")
out = model(tokenizer(["Hello, world.", "Bonjour le monde."]))
out["last_hidden_state"]  # (2, L, 768)
```

### Classification (community fine-tunes)

```python
from zeromodels.models.xlm_roberta import XLMRobertaSequenceClassify, XLMRobertaQnA

clf = XLMRobertaSequenceClassify.from_weights(
    "hf:cardiffnlp/twitter-xlm-roberta-base-sentiment"
)  # multilingual sentiment
qa = XLMRobertaQnA.from_weights("hf:deepset/xlm-roberta-base-squad2")  # multilingual QA
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
`XLMRobertaMultipleChoice` takes a static `num_choices` at build; its `classifier` head is
shape-independent of it, so the same weights load for any value.

### Loading from the Hub

```python
model = XLMRobertaMaskedLM.from_weights("hf:FacebookAI/xlm-roberta-base")
```

## Architecture notes

XLM-RoBERTa reuses RoBERTa's encoder verbatim (`roberta_backbone`): see the
[RoBERTa notes](roberta.md) for the embeddings (masked-`cumsum` position offset), post-LayerNorm
encoder blocks (`1e-5` epsilon, exact `gelu`), `<s>` pooler, and head structures. The only
differences are scale (`vocab_size = 250002`, `embed_dim` 768 / 1024) and the SentencePiece
tokenizer. `convert_xlm_roberta_hf_to_keras.py` reuses the RoBERTa weight transfer (the HF backbone
is exposed as `roberta.*` in both).

## Parity

Bit-close to Hugging Face `transformers` (eager, float32) on real checkpoints, including a padded
sequence so the padding-offset position ids are exercised: `XLMRobertaModel` matches to `5.7e-6`
and `XLMRobertaMaskedLM` to `6.1e-5`. The SentencePiece tokenizer reproduces HF's ids exactly
across languages (Latin, German, French, and Japanese inputs checked). See
`convert_xlm_roberta_hf_to_keras.py`.
