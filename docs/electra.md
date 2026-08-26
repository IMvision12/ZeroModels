# ELECTRA

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Google's ELECTRA in pure Keras 3: a BERT-style bidirectional text encoder pre-trained as a
replaced-token **discriminator** (with a smaller **generator** producing the corrupted tokens),
with a masked-LM head plus sequence / token classification, question-answering, and
multiple-choice heads. Unlike BERT it embeds tokens at a separate `embedding_size` and projects
up to the hidden size when they differ, and it has **no pooler**. One implementation runs
unmodified on TensorFlow / Torch / JAX, bit-exact with Hugging Face on real checkpoints.

- Paper: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (arXiv:2003.10555)](https://arxiv.org/abs/2003.10555)
- HF docs: [transformers/model_doc/electra](https://huggingface.co/docs/transformers/model_doc/electra)

See also [bert.md](bert.md), [modernbert.md](modernbert.md), [roberta.md](roberta.md), [deberta.md](deberta.md).

## Variants

ELECTRA ships two checkpoints per size, hosted as one repo each. The **discriminator** repo
(zm_config declares `ElectraModel`) serves the encoder + the classify / QA / token /
multiple-choice heads; the **generator** repo (zm_config declares `ElectraMaskedLM`) serves the
masked-LM. Load with `from_weights("zeromodels/<variant>")`.

| Size | Discriminator (encoder / downstream) | Generator (masked-LM) |
|---|---|---|
| small | [`zeromodels/electra_small_discriminator`](https://huggingface.co/zeromodels/electra_small_discriminator) | [`zeromodels/electra_small_generator`](https://huggingface.co/zeromodels/electra_small_generator) |
| base | [`zeromodels/electra_base_discriminator`](https://huggingface.co/zeromodels/electra_base_discriminator) | [`zeromodels/electra_base_generator`](https://huggingface.co/zeromodels/electra_base_generator) |
| large | [`zeromodels/electra_large_discriminator`](https://huggingface.co/zeromodels/electra_large_discriminator) | [`zeromodels/electra_large_generator`](https://huggingface.co/zeromodels/electra_large_generator) |

## API

### `ElectraModel`

The encoder backbone (no pooler). Takes a dict of `input_ids` / `attention_mask` /
`token_type_ids` (all `(B, L)` int) and returns `{"last_hidden_state": (B, L, embed_dim)}`.
Defaults below are for the small variant; base / large keep `embedding_size == embed_dim` (no
projection).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `30522` | token vocabulary size |
| `embedding_size` | `128` | token-embedding width (projected up when `!= embed_dim`) |
| `embed_dim` | `256` | hidden width |
| `num_layers` | `12` | transformer blocks |
| `num_heads` | `4` | attention heads |
| `mlp_dim` | `1024` | feed-forward inner width |
| `max_position_embeddings` | `512` | position-table size |
| `type_vocab_size` | `2` | token-type embeddings |
| `hidden_act` | `"gelu"` | feed-forward activation |
| `layer_norm_eps` | `1e-12` | LayerNorm epsilon |
| `pad_token_id` | `0` | padding token id |

### Task heads

Each composes an `ElectraModel` backbone; all take the same backbone constructor args, plus the
extras below. The discriminator heads take the pretrained encoder and a randomly-initialized task
layer (ready for fine-tuning, or a `hf:` fine-tune); `ElectraMaskedLM` loads from the generator
repo with real head weights.

| Class | Repo | Extra args | Output |
|---|---|---|---|
| `ElectraMaskedLM` | generator | | MLM logits `(B, L, vocab_size)` |
| `ElectraSequenceClassify` | discriminator | `num_classes` | `(B, num_classes)` |
| `ElectraTokenClassify` | discriminator | `num_classes` | `(B, L, num_classes)` |
| `ElectraQnA` | discriminator | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `ElectraMultipleChoice` | discriminator | `num_choices` | `(B, num_choices)` |

### `ElectraTokenizer`

WordPiece tokenizer on the `tokenizers` (Rust) backend (the discriminator and generator of a size
share one vocabulary), with `[CLS] A [SEP] B [SEP]` post-processing and segment ids.

```python
ElectraTokenizer(
    variant="electra_base_discriminator", tokenizer_file=None, max_seq_len=512
)
```

## End-to-end example

### Backbone features (discriminator)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.electra import ElectraModel, ElectraTokenizer

model = ElectraModel.from_weights("zeromodels/electra_base_discriminator")
tokenizer = ElectraTokenizer.from_weights("zeromodels/electra_base_discriminator")
out = model(tokenizer("Hello, world."))["last_hidden_state"]  # (1, L, 768)
```

### Fill-mask (generator)

```python
from zeromodels.models.electra import ElectraMaskedLM, ElectraTokenizer

mlm = ElectraMaskedLM.from_weights("zeromodels/electra_base_generator")
tokenizer = ElectraTokenizer.from_weights("zeromodels/electra_base_generator")
logits = mlm(tokenizer("The capital of France is [MASK]."))  # (1, L, vocab_size)
```

### Classification (fine-tune the discriminator)

```python
from zeromodels.models.electra import ElectraSequenceClassify, ElectraQnA

clf = ElectraSequenceClassify.from_weights(
    "zeromodels/electra_base_discriminator", num_classes=2
)  # encoder pretrained, classifier random -> fine-tune
qa = ElectraQnA.from_weights("hf:org/electra-base-squad2")  # or a community fine-tune
```

`num_classes` is read from a `hf:` fine-tune's config. `ElectraMultipleChoice` takes a static
`num_choices` at build; its `classifier` head is shape-independent of it.

### Loading from the Hub

```python
model = ElectraModel.from_weights("hf:google/electra-base-discriminator")
mlm = ElectraMaskedLM.from_weights("hf:google/electra-base-generator")
```

## Architecture notes

- **Separate `embedding_size`**: embeds at `embedding_size` and projects to `embed_dim` with an
  `embed_project` linear only when they differ (small: 128 -> 256; base / large keep them equal).
- **BERT-style encoder**: post-LayerNorm blocks (multi-head self-attention + GELU feed-forward),
  no pooler.
- **Heads**: classification reads the first (`[CLS]`) token through a dense + GELU + linear head;
  the masked-LM head is a dense -> GELU -> LayerNorm at `embedding_size`, then a decoder tied to
  the word embeddings.

## Parity

Bit-exact with Hugging Face `transformers` (eager, float32): every class matches the reference
forward to `< 1e-6` max-abs difference (`ElectraModel`, `ElectraMaskedLM`, and each task head),
including the `embed_project` projection on the small variant. See
`convert_electra_hf_to_keras.py`.
