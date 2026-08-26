# BERT

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Google's BERT in pure Keras 3: the bidirectional transformer text encoder, with a masked-LM
head plus sequence / token classification, next-sentence, question-answering, and
multiple-choice heads. The encoder is a stack of post-LayerNorm blocks (multi-head
self-attention + a GELU feed-forward) over summed word / absolute-position / token-type
embeddings, with a `tanh` pooler on the `[CLS]` token. One implementation runs unmodified on
TensorFlow / Torch / JAX.

- Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arXiv:1810.04805)](https://arxiv.org/abs/1810.04805)
- HF docs: [transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert)

See also [modernbert.md](modernbert.md), [roberta.md](roberta.md), [electra.md](electra.md), [deberta.md](deberta.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. The architecture is identical
across variants; only the vocabulary and tokenizer casing differ.

| Variant | Hub | vocab | casing |
|---|---|---|---|
| `bert_base_uncased` | [`zeromodels/bert_base_uncased`](https://huggingface.co/zeromodels/bert_base_uncased) | 30522 | lowercased |
| `bert_large_uncased` | [`zeromodels/bert_large_uncased`](https://huggingface.co/zeromodels/bert_large_uncased) | 30522 | lowercased |
| `bert_base_cased` | [`zeromodels/bert_base_cased`](https://huggingface.co/zeromodels/bert_base_cased) | 28996 | case-preserving |
| `bert_large_cased` | [`zeromodels/bert_large_cased`](https://huggingface.co/zeromodels/bert_large_cased) | 28996 | case-preserving |

## API

### `BertModel`

The encoder backbone plus a `tanh` pooler. Takes a dict of `input_ids` / `attention_mask` /
`token_type_ids` (all `(B, L)` int) and returns `{"last_hidden_state": (B, L, embed_dim),
"pooler_output": (B, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `30522` | token vocabulary size |
| `embed_dim` | `768` | model / hidden width |
| `num_layers` | `12` | transformer blocks |
| `num_heads` | `12` | attention heads |
| `mlp_dim` | `3072` | feed-forward inner width |
| `max_position_embeddings` | `512` | position-table size |
| `type_vocab_size` | `2` | token-type (segment) embeddings |
| `hidden_act` | `"gelu"` | feed-forward activation |
| `norm_eps` | `1e-12` | LayerNorm epsilon (alias `layer_norm_eps`) |
| `pad_token_id` | `0` | padding token id |
| `add_pooler` | `True` | attach the `[CLS]` pooler |

### Task heads

Each composes a `BertModel` backbone and adds a head; all take the same backbone constructor
args, plus the extras below. The pretrained encoder (and the NSP head) load real weights; the
classification / QA heads start randomly initialized (ready for fine-tuning) and load trained
weights from a `hf:` fine-tune.

| Class | Extra args | Output |
|---|---|---|
| `BertMaskedLM` | | MLM logits `(B, L, vocab_size)` |
| `BertSequenceClassify` | `num_classes` | `(B, num_classes)` |
| `BertTokenClassify` | `num_classes` | `(B, L, num_classes)` |
| `BertNextSentencePredict` | | `(B, 2)` |
| `BertQnA` | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `BertMultipleChoice` | `num_choices` | `(B, num_choices)` |

### `BertTokenizer`

WordPiece tokenizer on the `tokenizers` (Rust) backend: the BERT normalizer, greedy WordPiece,
and `[CLS] A [SEP] B [SEP]` post-processing with segment ids.

```python
BertTokenizer(variant="bert_base_uncased", tokenizer_file=None, max_seq_len=512)
```

Calling it accepts a string, a list of strings (a batch), or a sentence pair (`text_pair=`), and
returns the `input_ids` / `attention_mask` / `token_type_ids` dict the models consume. Decode
with `.decode(ids)`.

## End-to-end example

### Fill-mask

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.bert import BertMaskedLM, BertTokenizer

mlm = BertMaskedLM.from_weights("zeromodels/bert_base_uncased")
tokenizer = BertTokenizer.from_weights("zeromodels/bert_base_uncased")

inputs = tokenizer("the capital of france is [MASK].")
logits = mlm(inputs)  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.ids_to_tokens[int(logits[0, mask].argmax())])  # -> "paris"
```

### Backbone features

```python
from zeromodels.models.bert import BertModel, BertTokenizer

model = BertModel.from_weights("zeromodels/bert_base_uncased")
tokenizer = BertTokenizer.from_weights("zeromodels/bert_base_uncased")
out = model(tokenizer("Hello, world."))
out["last_hidden_state"]  # (1, L, 768)
out["pooler_output"]  # (1, 768)
```

### Classification (community fine-tunes)

```python
from zeromodels.models.bert import BertSequenceClassify, BertTokenClassify, BertQnA

clf = BertSequenceClassify.from_weights(
    "hf:textattack/bert-base-uncased-SST-2"
)  # sentiment
ner = BertTokenClassify.from_weights("hf:dslim/bert-base-NER")  # NER, cased
qa = BertQnA.from_weights("hf:deepset/bert-base-cased-squad2")  # extractive QA
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
`BertMultipleChoice` takes a static `num_choices` at build (the choice axis is folded into the
batch through the shared backbone), and its `classifier` head is shape-independent of it, so the
same weights load for any value.

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including community fine-tunes;
it reads `config.json` (architecture + `num_labels`) and loads the checkpoint:

```python
model = BertMaskedLM.from_weights("hf:google-bert/bert-base-uncased")
```

## Architecture notes

- **Embeddings**: summed word + absolute-position + token-type embeddings, then LayerNorm +
  dropout. Position ids come from `cumsum(ones_like(input_ids)) - 1` (not `arange`) so the model
  stays shape-polymorphic across backends.
- **Encoder**: `num_layers` post-LayerNorm blocks (`LayerNorm(x + Sublayer(x))`): multi-head
  self-attention with an additive padding mask, then a `mlp_dim` GELU feed-forward.
- **Pooler / heads**: a `tanh` dense over `[CLS]`; `BertMaskedLM` adds a dense + GELU + LayerNorm
  transform and a vocabulary projection; the classify heads add dropout + a dense classifier.

## Parity

Bit-close to Hugging Face `transformers` (eager, float32) on real checkpoints:
`BertModel` / `BertMaskedLM` match to `~1e-5`; the classification / QA fine-tunes
(`textattack/bert-base-uncased-SST-2`, `dslim/bert-base-NER`, `deepset/bert-base-cased-squad2`,
loaded via `hf:`) match to `< 5e-6`. The WordPiece tokenizer reproduces HF's `input_ids` /
`token_type_ids` / `attention_mask` exactly. See `convert_bert_hf_to_keras.py`.
