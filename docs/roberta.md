# RoBERTa

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Facebook AI's RoBERTa in pure Keras 3: a robustly-optimized retraining of BERT's bidirectional
encoder, with a masked-LM head plus sequence / token classification, question-answering, and
multiple-choice heads. It shares BERT's encoder but differs in three ways: position ids are
offset by the padding id (so the table has two extra slots), there is a single token-type, and
the sentence classifier reads the `<s>` token through a `tanh` head. There is **no
next-sentence head**. One implementation runs unmodified on TensorFlow / Torch / JAX.

- Paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach (arXiv:1907.11692)](https://arxiv.org/abs/1907.11692)
- HF docs: [transformers/model_doc/roberta](https://huggingface.co/docs/transformers/model_doc/roberta)

See also [bert.md](bert.md), [xlm_roberta.md](xlm_roberta.md), [deberta.md](deberta.md), [electra.md](electra.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub | layers / dim |
|---|---|---|
| `roberta_base` | [`zeromodels/roberta_base`](https://huggingface.co/zeromodels/roberta_base) | 12 / 768 |
| `roberta_large` | [`zeromodels/roberta_large`](https://huggingface.co/zeromodels/roberta_large) | 24 / 1024 |

## API

### `RobertaModel`

The encoder backbone plus a `tanh` pooler over the `<s>` token. Takes a dict of `input_ids` /
`attention_mask` / `token_type_ids` (all `(B, L)` int; segment ids are always `0`) and returns
`{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `50265` | token vocabulary size |
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

Each composes a `RobertaModel` backbone and adds a head; all take the same backbone constructor
args, plus the extras below. The pretrained encoder + masked-LM head load real weights; the
classification / QA heads start randomly initialized (ready for fine-tuning) and load trained
weights from a `hf:` fine-tune.

| Class | Extra args | Output |
|---|---|---|
| `RobertaMaskedLM` | | MLM logits `(B, L, vocab_size)` |
| `RobertaSequenceClassify` | `num_classes` | `(B, num_classes)` |
| `RobertaTokenClassify` | `num_classes` | `(B, L, num_classes)` |
| `RobertaQnA` | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `RobertaMultipleChoice` | `num_choices` | `(B, num_choices)` |

### `RobertaTokenizer`

Byte-level BPE tokenizer on the `tokenizers` (Rust) backend: byte-level pre-tokenization, BPE over
`vocab.json` + `merges.txt`, and `<s> A </s>` / `<s> A </s> </s> B </s>` post-processing (the
left-stripping `<mask>` token is protected).

```python
RobertaTokenizer(variant="roberta_base", tokenizer_file=None, max_seq_len=512)
```

Calling it accepts a string, a list of strings, or a sentence pair (`text_pair=`), and returns
the `input_ids` / `attention_mask` / `token_type_ids` dict the models consume.

## End-to-end example

### Fill-mask

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.roberta import RobertaMaskedLM, RobertaTokenizer

mlm = RobertaMaskedLM.from_weights("zeromodels/roberta_base")
tokenizer = RobertaTokenizer.from_weights("zeromodels/roberta_base")

inputs = tokenizer("The capital of France is <mask>.")
logits = mlm(inputs)  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))  # -> " Paris"
```

### Backbone features

```python
from zeromodels.models.roberta import RobertaModel, RobertaTokenizer

model = RobertaModel.from_weights("zeromodels/roberta_base")
tokenizer = RobertaTokenizer.from_weights("zeromodels/roberta_base")
out = model(tokenizer("Hello, world."))
out["last_hidden_state"]  # (1, L, 768)
```

### Classification (community fine-tunes)

```python
from zeromodels.models.roberta import RobertaSequenceClassify, RobertaQnA

# 3-class NLI (0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT), common for zero-shot
nli = RobertaSequenceClassify.from_weights("hf:FacebookAI/roberta-large-mnli")
qa = RobertaQnA.from_weights("hf:deepset/roberta-base-squad2")  # extractive QA
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
`RobertaMultipleChoice` takes a static `num_choices` at build; its `classifier` head is
shape-independent of it, so the same weights load for any value.

### Loading from the Hub

```python
model = RobertaMaskedLM.from_weights("hf:FacebookAI/roberta-base")
```

## Architecture notes

- **Embeddings**: summed word + absolute-position + token-type embeddings, then LayerNorm +
  dropout. Position ids are derived from the non-padding mask
  (`cumsum(input_ids != pad) * mask + pad`) so padding maps to the padding slot and the first
  real token starts at `pad + 1` (masked `cumsum`, not `arange`, keeps it shape-polymorphic).
- **Encoder**: `num_layers` post-LayerNorm blocks (multi-head self-attention + `mlp_dim` GELU
  feed-forward), `layer_norm_eps = 1e-5`.
- **Pooler / heads**: a `tanh` dense over `<s>`; `RobertaSequenceClassify` uses RoBERTa's
  classification head (dropout + `tanh` dense + dropout + projection) on the `<s>` token.

## Parity

Bit-close to Hugging Face `transformers` (eager, float32) on real checkpoints, including a padded
sequence so the padding-offset position ids are exercised: `RobertaModel` / `RobertaMaskedLM`
match to `~1e-5`; the SST-2 / SQuAD fine-tunes (via `hf:`) match to `< 1.1e-5`. The byte-level BPE
tokenizer reproduces HF's ids exactly. See `convert_roberta_hf_to_keras.py`.
