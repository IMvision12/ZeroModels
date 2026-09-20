# MPNet

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Microsoft's MPNet in pure Keras 3: a bidirectional encoder pre-trained with **masked and
permuted** language modelling, which unifies BERT's masked-LM objective with XLNet's
permuted one. It carries a masked-LM head plus sequence / token classification,
question-answering, and multiple-choice heads. One implementation runs unmodified on
TensorFlow / Torch / JAX.

Three things distinguish it from BERT:

- **No token-type embeddings.** Inputs are `input_ids` + `attention_mask` only, so
  `MPNetTokenizer` emits no `token_type_ids`.
- **A shared relative position bias.** Every attention layer adds the same
  `(1, num_heads, L, L)` bias, gathered from `relative_attention_num_buckets`
  log-spaced buckets and computed **once** per forward pass.
- **The attention output projection lives inside self-attention** (`attention.attn.o`
  upstream), so the block's LayerNorm sits directly on `attention`, not on
  `attention.output` as in BERT.

Like RoBERTa, position ids are offset past the padding id.

- Paper: [MPNet: Masked and Permuted Pre-training for Language Understanding (arXiv:2004.09297)](https://arxiv.org/abs/2004.09297)
- HF docs: [transformers/model_doc/mpnet](https://huggingface.co/docs/transformers/model_doc/mpnet)

See also [bert.md](bert.md), [roberta.md](roberta.md), [modernbert.md](modernbert.md),
[electra.md](electra.md).

MPNet is also the backbone of `sentence-transformers/all-mpnet-base-v2`, one of the most
widely used sentence-embedding models.

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub | layers / dim |
|---|---|---|
| `mpnet_base` | [`zeromodels/mpnet_base`](https://huggingface.co/zeromodels/mpnet_base) | 12 / 768 |

## API

### `MPNetModel`

The encoder backbone plus a `tanh` pooler over the `<s>` token. Takes a dict of
`input_ids` / `attention_mask` (both `(B, L)` int) and returns
`{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}`. Pass
`add_pooler=False` to drop the pooler.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `30527` | token vocabulary size |
| `embed_dim` | `768` | model / hidden width |
| `num_layers` | `12` | transformer blocks |
| `num_heads` | `12` | attention heads |
| `mlp_dim` | `3072` | feed-forward inner width |
| `max_position_embeddings` | `512` | position-table size (padding-offset) |
| `relative_attention_num_buckets` | `32` | relative position bias buckets |
| `hidden_act` | `"gelu"` | feed-forward activation |
| `layer_norm_eps` | `1e-12` | LayerNorm epsilon |
| `pad_token_id` | `1` | padding token id |

### Task heads

Each composes an `MPNetModel` backbone and adds a head; all take the same backbone
constructor args, plus the extras below. The pretrained encoder + masked-LM head load
real weights; the classification / QA heads start randomly initialized (ready for
fine-tuning) and load trained weights from a `hf:` fine-tune.

| Class | Extra args | Output |
|---|---|---|
| `MPNetMaskedLM` | `add_pooler=False` | `(B, L, vocab_size)` token logits |
| `MPNetSequenceClassify` | `num_classes=2`, `classifier_dropout=0.0`, `classifier_activation="linear"` | `(B, num_classes)` |
| `MPNetTokenClassify` | `num_classes=2`, `classifier_dropout=0.0`, `classifier_activation="linear"` | `(B, L, num_classes)` |
| `MPNetQnA` | — | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `MPNetMultipleChoice` | `num_choices=2`, `classifier_dropout=0.0` | `(B, num_choices)` |

`MPNetSequenceClassify` pools the `<s>` token inside its own head (dropout → `tanh`
dense → dropout → projection) rather than reading the encoder pooler, matching the
reference implementation.

`MPNetMultipleChoice` takes `(B, num_choices, L)` inputs, folds the choice axis into the
batch, and folds the scores back out.

### `MPNetTokenizer`

WordPiece tokenizer (`tokenizers` Rust backend) loading a `tokenizer.json`. MPNet pairs
RoBERTa-style special tokens (`<s>`, `</s>`, `<pad>`, `<mask>`) with a BERT-style
WordPiece vocabulary and `[UNK]`. `call` returns `input_ids` + `attention_mask` only.

## Usage

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.mpnet import MPNetModel, MPNetTokenizer

model = MPNetModel.from_weights("zeromodels/mpnet_base")
tokenizer = MPNetTokenizer.from_weights("zeromodels/mpnet_base")

out = model(tokenizer(["the quick brown fox", "jumped over the lazy dog"]))
print(out["last_hidden_state"].shape, out["pooler_output"].shape)
```

Fill-mask with the pretrained head:

```python
from zeromodels.models.mpnet import MPNetMaskedLM

mlm = MPNetMaskedLM.from_weights("zeromodels/mpnet_base")
logits = mlm(tokenizer(["the capital of France is <mask>."]))
```

Or convert any MPNet checkpoint straight from the Hub:

```python
model = MPNetModel.from_weights("hf:microsoft/mpnet-base")
```

## Conversion accuracy

Converted from `microsoft/mpnet-base` and checked against the reference
implementation on a padded batch (`max|Δ|`, pad positions excluded):

| Class | last_hidden_state | pooler_output | logits |
|---|---|---|---|
| `MPNetModel` | 2.6e-06 | 2.7e-07 | — |
| `MPNetMaskedLM` | — | — | 1.3e-05 |

Masked-LM argmax agreement with the reference is 100%.

Each variant repo hosts **one** file: the superset `MPNetMaskedLM(add_pooler=True)`
(encoder + pooler + masked-LM head). `MPNetModel.CHECKPOINT_SOURCE` points every class at
it, so the encoder and the task heads copy their own subset out of that single checkpoint —
the numbers above are re-measured after that round trip.

> **Converter note.** MPNet ships both `lm_head.bias` and `lm_head.decoder.bias` with
> *different* values and does not tie them; the forward pass reads `decoder.bias`, while
> `lm_head.bias` is vestigial. The converter maps the decoder bias to
> `lm_head.decoder.bias` accordingly — using `lm_head.bias` (as is correct for BERT and
> RoBERTa, where HF does tie them) shifts every logit by up to ~7.8.
