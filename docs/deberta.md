# DeBERTa (v1 / v2 / v3)

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Microsoft's DeBERTa in pure Keras 3: the disentangled-attention bidirectional text encoder, in
its three generations (v1, v2, v3), each with a masked-LM head plus sequence / token
classification, question-answering, and multiple-choice heads. DeBERTa replaces BERT's
absolute-position embeddings with **disentangled attention** (separate content and relative-
position vectors), has **no pooler and no next-sentence head**, and attaches a *context pooler*
(dense + GELU over `[CLS]`) in the sequence / multiple-choice heads. One implementation per
version runs unmodified on TensorFlow / Torch / JAX.

- Papers: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention (arXiv:2006.03654)](https://arxiv.org/abs/2006.03654) (v1 / v2) · [DeBERTaV3 (arXiv:2111.09543)](https://arxiv.org/abs/2111.09543) (v3)
- HF docs: [deberta](https://huggingface.co/docs/transformers/model_doc/deberta) · [deberta-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)

See also [bert.md](bert.md), [roberta.md](roberta.md), [electra.md](electra.md), [modernbert.md](modernbert.md).

## Variants

Each generation lives in its own package (`zeromodels.models.deberta` / `deberta_v2` /
`deberta_v3`) with the same class set and a `V2` / `V3` class prefix. Load with
`from_weights("zeromodels/<variant>")`.

| Version | Variant | Hub | layers / dim |
|---|---|---|---|
| v1 | `deberta_base` | [`zeromodels/deberta_base`](https://huggingface.co/zeromodels/deberta_base) | 12 / 768 |
| v1 | `deberta_large` | [`zeromodels/deberta_large`](https://huggingface.co/zeromodels/deberta_large) | 24 / 1024 |
| v2 | `deberta_v2_xlarge` | [`zeromodels/deberta_v2_xlarge`](https://huggingface.co/zeromodels/deberta_v2_xlarge) | 24 / 1536 |
| v2 | `deberta_v2_xxlarge` | [`zeromodels/deberta_v2_xxlarge`](https://huggingface.co/zeromodels/deberta_v2_xxlarge) | 48 / 1536 |
| v3 | `deberta_v3_xsmall` | [`zeromodels/deberta_v3_xsmall`](https://huggingface.co/zeromodels/deberta_v3_xsmall) | 12 / 384 |
| v3 | `deberta_v3_small` | [`zeromodels/deberta_v3_small`](https://huggingface.co/zeromodels/deberta_v3_small) | 6 / 768 |
| v3 | `deberta_v3_base` | [`zeromodels/deberta_v3_base`](https://huggingface.co/zeromodels/deberta_v3_base) | 12 / 768 |
| v3 | `deberta_v3_large` | [`zeromodels/deberta_v3_large`](https://huggingface.co/zeromodels/deberta_v3_large) | 24 / 1024 |

## API

### `DebertaModel` / `DebertaV2Model` / `DebertaV3Model`

The encoder backbone (no pooler). Takes a dict of `input_ids` / `attention_mask` /
`token_type_ids` (segment ids are accepted for API parity but unused) and returns
`{"last_hidden_state": (B, L, embed_dim)}`. Common backbone args (v1 defaults):

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `50265` (v1) / `128100` (v2/v3) | token vocabulary size |
| `embed_dim` | `768` | hidden width |
| `num_layers` | `12` | transformer blocks |
| `num_heads` | `12` | attention heads |
| `mlp_dim` | `3072` | feed-forward inner width |
| `max_position_embeddings` | `512` | max sequence length |
| `max_relative_positions` | `512` | relative-position span |
| `pos_att_type` | `("c2p", "p2c")` | disentangled score terms |
| `hidden_act` | `"gelu"` | feed-forward activation |
| `layer_norm_eps` | `1e-7` | LayerNorm epsilon |
| `pad_token_id` | `0` | padding token id |

**v2 / v3** add `position_buckets` (`256`, log-bucketed relative positions), `norm_rel_ebd`
(`True`, LayerNorm on relative embeddings), and `conv_kernel_size` / `conv_act` (a depthwise
convolution after the first layer: `3` for v2, `0`/disabled for v3).

### Task heads

Each composes the matching backbone; all take the same backbone args, plus the extras below. The
masked-LM head is pretrained; the classification / QA heads start randomly initialized (ready for
fine-tuning) and load trained weights from a `hf:` fine-tune. (v1 has no multiple-choice head.)

| Class (per version) | Extra args | Output |
|---|---|---|
| `*MaskedLM` | | MLM logits `(B, L, vocab_size)` |
| `*SequenceClassify` | `num_classes` | `(B, num_classes)` |
| `*TokenClassify` | `num_classes` | `(B, L, num_classes)` |
| `*QnA` | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `*MultipleChoice` (v2 / v3) | `num_choices` | `(B, num_choices)` |

### Tokenizers

- **v1** `DebertaTokenizer`: byte-level BPE (`vocab.json` + `merges.txt`) with BERT-style specials.
- **v2 / v3** `DebertaV2Tokenizer` / `DebertaV3Tokenizer`: SentencePiece Unigram (`spm.model`,
  128 100 pieces) with `[CLS] A [SEP] B [SEP]` post-processing. v3 differs from v2 only in the
  underlying `spm.model`. All return `input_ids` / `attention_mask` / `token_type_ids`.

## End-to-end example

### Fill-mask

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.deberta_v3 import DebertaV3MaskedLM, DebertaV3Tokenizer

mlm = DebertaV3MaskedLM.from_weights("zeromodels/deberta_v3_base")
tokenizer = DebertaV3Tokenizer.from_weights("zeromodels/deberta_v3_base")

inputs = tokenizer("The capital of France is [MASK].")
logits = mlm(inputs)  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))
```

### Backbone features

```python
from zeromodels.models.deberta_v3 import DebertaV3Model, DebertaV3Tokenizer

model = DebertaV3Model.from_weights("zeromodels/deberta_v3_base")
tokenizer = DebertaV3Tokenizer.from_weights("zeromodels/deberta_v3_base")
out = model(tokenizer("Hello, world."))["last_hidden_state"]  # (1, L, 768)
```

### Classification (community fine-tunes)

```python
from zeromodels.models.deberta_v3 import DebertaV3SequenceClassify, DebertaV3QnA

nli = DebertaV3SequenceClassify.from_weights(
    "hf:org/deberta-v3-base-mnli"
)  # 3-class NLI
qa = DebertaV3QnA.from_weights("hf:org/deberta-v3-squad")  # extractive QA
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
`*MultipleChoice` takes a static `num_choices` at build; its `classifier` head is
shape-independent of it.

### Loading from the Hub

```python
model = DebertaV3Model.from_weights("hf:microsoft/deberta-v3-base")
```

> **DeBERTa-v3 checkpoints ship as float16.** Load the HF reference with
> `from_pretrained(..., dtype=torch.float32)` to compare like-for-like; Keras runs in fp32.

## Architecture notes

Disentangled attention: each token has separate content and relative-position vectors, and the
attention score sums content→content, content→position (c2p), and position→content (p2c) terms.
Position information enters only through attention (the input embeddings are word embeddings +
LayerNorm, no absolute-position or token-type embeddings).

- **v1**: fused `in_proj` q/k/v, dedicated `pos_proj` (c2p) / `pos_q_proj` (p2c) over a raw
  relative-position table. `layer_norm_eps = 1e-7`.
- **v2**: separate `query_proj` / `key_proj` / `value_proj`; log-bucketed relative positions
  (`position_buckets = 256`); `share_att_key`; a LayerNorm on relative embeddings; and a depthwise
  convolution (`conv_kernel_size = 3`) after the first layer. SentencePiece vocabulary (128 100).
- **v3**: the v2 backbone (HF `model_type = "deberta-v2"`) without the convolution, pretrained
  ELECTRA-style. Only the backbone is ported; the discriminator-only pretraining heads are unused.

## Parity

Bit-close to Hugging Face `transformers` (eager, float32) on real checkpoints:
`DebertaModel` / v2 / v3 match to `~1e-5` or better (v2-xlarge to `< 1e-6`). See each
`convert_deberta*_hf_to_keras.py`.
