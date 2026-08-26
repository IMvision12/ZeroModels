# ModernBERT

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code> + <code>tokenizer.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Answer.AI / LightOn's ModernBERT in pure Keras 3: a modernized bidirectional text encoder with a
masked-LM head plus sequence / token classification, question-answering, and multiple-choice
heads. It updates BERT with rotary position embeddings, attention that alternates between a global
(full) layer and local sliding-window layers, GeGLU feed-forwards, pre-LayerNorm residuals, and an
8192-token context. It has **no pooler and no token-type embeddings** (position is injected by
rotary embeddings), so the tokenizer emits only `input_ids` / `attention_mask`. One implementation
runs unmodified on TensorFlow / Torch / JAX.

- Paper: [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (arXiv:2412.13663)](https://arxiv.org/abs/2412.13663)
- HF docs: [transformers/model_doc/modernbert](https://huggingface.co/docs/transformers/model_doc/modernbert)

See also [bert.md](bert.md), [roberta.md](roberta.md), [electra.md](electra.md), [deberta.md](deberta.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`. Both share one tokenizer.

| Variant | Hub | layers / dim |
|---|---|---|
| `modernbert_base` | [`zeromodels/modernbert_base`](https://huggingface.co/zeromodels/modernbert_base) | 22 / 768 |
| `modernbert_large` | [`zeromodels/modernbert_large`](https://huggingface.co/zeromodels/modernbert_large) | 28 / 1024 |

## API

### `ModernBertModel`

The encoder backbone (no pooler). Takes a dict of `input_ids` / `attention_mask` (both `(B, L)`
int) and returns `{"last_hidden_state": (B, L, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `50368` | token vocabulary size |
| `embed_dim` | `768` | model / hidden width |
| `num_layers` | `22` | transformer blocks |
| `num_heads` | `12` | attention heads |
| `mlp_dim` | `1152` | GeGLU inner width |
| `max_position_embeddings` | `8192` | maximum sequence length |
| `hidden_act` | `"gelu"` | GeGLU activation |
| `norm_eps` | `1e-5` | LayerNorm epsilon |
| `local_attention` | `128` | sliding-window size of local layers |
| `global_attn_every_n_layers` | `3` | period of global (full-attention) layers |
| `global_rope_theta` | `160000.0` | RoPE base for global layers |
| `local_rope_theta` | `10000.0` | RoPE base for local layers |
| `pad_token_id` | `50283` | padding token id |

### Task heads

Each composes a `ModernBertModel` backbone and adds a head; all take the same backbone constructor
args, plus the extras below. The pretrained encoder + masked-LM head (and the shared prediction
head) load real weights; the final classifier / span layer starts randomly initialized (ready for
fine-tuning) and loads trained weights from a `hf:` fine-tune.

| Class | Extra args | Output |
|---|---|---|
| `ModernBertMaskedLM` | | MLM logits `(B, L, vocab_size)` |
| `ModernBertSequenceClassify` | `num_classes`, `classifier_pooling` (`"mean"` / `"cls"`) | `(B, num_classes)` |
| `ModernBertTokenClassify` | `num_classes` | `(B, L, num_classes)` |
| `ModernBertQnA` | | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `ModernBertMultipleChoice` | `num_choices`, `classifier_pooling` | `(B, num_choices)` |

### `ModernBertTokenizer`

Byte-level BPE tokenizer on the `tokenizers` (Rust) backend, with `[CLS] A [SEP]` post-processing.
ModernBERT has no token-type ids, so `call` returns only `input_ids` / `attention_mask`.

```python
ModernBertTokenizer(variant="modernbert_base", tokenizer_file=None, max_seq_len=8192)
```

## End-to-end example

### Fill-mask

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.modernbert import ModernBertMaskedLM, ModernBertTokenizer

mlm = ModernBertMaskedLM.from_weights("zeromodels/modernbert_base")
tokenizer = ModernBertTokenizer.from_weights("zeromodels/modernbert_base")

inputs = tokenizer("The capital of France is [MASK].")
logits = mlm(inputs)  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))
```

### Backbone features

```python
from zeromodels.models.modernbert import ModernBertModel, ModernBertTokenizer

model = ModernBertModel.from_weights("zeromodels/modernbert_base")
tokenizer = ModernBertTokenizer.from_weights("zeromodels/modernbert_base")
out = model(tokenizer("Hello, world."))["last_hidden_state"]  # (1, L, 768)
```

### Classification (community fine-tunes)

```python
from zeromodels.models.modernbert import (
    ModernBertSequenceClassify,
    ModernBertTokenClassify,
)

clf = ModernBertSequenceClassify.from_weights("hf:org/modernbert-base-sentiment")
ner = ModernBertTokenClassify.from_weights("hf:org/modernbert-base-ner")
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
`ModernBertMultipleChoice` takes a static `num_choices` at build; its `classifier` head is
shape-independent of it.

### Loading from the Hub

```python
model = ModernBertMaskedLM.from_weights("hf:answerdotai/ModernBERT-base")
```

## Architecture notes

- **Rotary position embeddings** with two bases: global layers use `global_rope_theta = 160000`,
  local layers use `local_rope_theta = 10000`.
- **Alternating attention**: every `global_attn_every_n_layers` (3rd) layer uses full attention;
  the rest use a sliding window of `local_attention` (128) tokens.
- **GeGLU** feed-forward (`Wi` projects to `2 * mlp_dim`, gated, then `Wo`); pre-LayerNorm
  residuals; bias-free linears and LayerNorms; the first layer's attention LayerNorm is the
  identity (the embeddings are already normalized); a final LayerNorm ends the stack.
- The MLM decoder is **tied** to the token embeddings.

## Parity

Bit-close to Hugging Face `transformers` (eager, float32) on real checkpoints: `ModernBertModel`
and `ModernBertMaskedLM` match to a very high cosine (the larger max residual on `large` is
deep/wide fp32 op-order accumulation, mean residual `~6e-6`, cosine `0.9999999`, not an
architectural difference), so the converter gates on cosine `>= 0.9999`. The tokenizer reproduces
HF's `input_ids` / `attention_mask` exactly. See `convert_modernbert_hf_to_keras.py`.
