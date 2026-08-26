# Gemma 4 Unified

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The **12B** Gemma 4 checkpoints (`google/gemma-4-12B`) are a distinct,
**encoder-free** any-to-any architecture (`model_type` `gemma4_unified`), ported to
pure Keras 3. Unlike the [`gemma4`](gemma4.md) family (NaViT vision tower + USM audio
conformer), the unified model has **no vision or audio transformer towers**:

- **Vision** arrives as raw 48px merged pixel patches, projected by a
  `LayerNorm → Dense → LayerNorm → + factorized 2-D position embedding → LayerNorm`
  pipeline and a shared soft-token projector (`Gemma4UnifiedVisionEmbedder`).
- **Audio** arrives as raw 640-sample (40 ms) waveform frames, projected straight to
  text space through an RMSNorm + Dense (no mel, no downsampling).

The text decoder is the plain dense Gemma 4 decoder (no Per-Layer Embeddings, no
MoE) with global `K = V` attention and a learned per-layer scalar, so it reuses
[`Gemma4Model`](gemma4.md). A single `Gemma4UnifiedConditionalGenerate` drives text-only,
image+text, and image+audio+text generation; `Gemma4UnifiedTextGenerate` is its
text-only counterpart (built with no towers).

Links:

- Paper: [Gemma 4 Technical Report (arXiv:2607.02770)](https://arxiv.org/abs/2607.02770)
- HF docs: [transformers/model_doc/gemma4_unified](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)

See also [gemma4.md](gemma4.md), [gemma3.md](gemma3.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; `-it` is instruction-tuned. Gemma 4 is
Apache 2.0.

| Variant | Hub | Modalities |
|---|---|---|
| `gemma-4-12b` | [`zeromodels/gemma-4-12b`](https://huggingface.co/zeromodels/gemma-4-12b) | text + image + audio |
| `gemma-4-12b-it` | [`zeromodels/gemma-4-12b-it`](https://huggingface.co/zeromodels/gemma-4-12b-it) | text + image + audio |

Upstream Google safetensors also load via the `hf:` prefix, e.g.
`from_weights("hf:google/gemma-4-12B-it")` (converts in process; pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

Configs are typed: `Gemma4UnifiedConfig` (composite) over `Gemma4TextConfig` (reused
from `gemma4`), `Gemma4UnifiedVisionConfig`, and `Gemma4UnifiedAudioConfig`.

### `Gemma4UnifiedModel`

The backbone (no LM head): the reused `Gemma4Model` text decoder plus the
encoder-free vision and audio embedders. Image / audio soft tokens are scattered
onto their placeholder positions in `input_ids` before the decoder runs. Returns
`{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `text_config` | `None` | dict of `Gemma4Model` constructor args |
| `vision_config` | `None` | encoder-free vision-embedder settings; `None` skips it |
| `audio_config` | `None` | encoder-free audio-embedder settings; `None` skips it |
| `image_token_id` | `258880` | placeholder id filled with image soft tokens |
| `video_token_id` | `258884` | placeholder id filled with video soft tokens |
| `audio_token_id` | `258881` | placeholder id filled with audio soft tokens |
| `pad_token_id` | `0` | id used to embed placeholder slots before the scatter |
| `use_bidirectional_vision` | `True` | blockwise bidirectional attention within each image block |

### `Gemma4UnifiedConditionalGenerate`

`Gemma4UnifiedModel` plus a (tied) LM head with final-logit softcapping. Returns
`{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. The multimodal prefill
fuses the soft tokens and applies the blockwise vision mask; decoding is text-only
over the per-layer KV cache.

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    pixel_values=None,
    pixel_position_ids=None,
    input_features=None,
    input_features_mask=None,
)
```

### `Gemma4UnifiedTextGenerate`

The text-only counterpart of `Gemma4UnifiedConditionalGenerate`: the unified decoder plus
a (tied) LM head, built with no vision / audio embedder. `.generate()` takes just token
ids. It shares the decoder weights with `Gemma4UnifiedConditionalGenerate`. Set
`config_class = Gemma4TextConfig`.

```python
from zeromodels.models.gemma4_unified import (
    Gemma4UnifiedTextGenerate,
    Gemma4UnifiedTokenizer,
)

model = Gemma4UnifiedTextGenerate.from_weights("zeromodels/gemma-4-12b-it")
tokenizer = Gemma4UnifiedTokenizer.from_weights("zeromodels/gemma-4-12b-it")
outputs = model.generate(
    **tokenizer([{"role": "user", "content": "Summarize attention in one line."}]),
    max_new_tokens=64,
)
print(tokenizer.decode(outputs[0]))
```

### `Gemma4UnifiedVisionEmbedder`

The encoder-free vision embedder used by the backbone: raw merged pixel patches ->
`LayerNorm -> Dense -> LayerNorm -> + factorized 2-D position embedding -> LayerNorm`
-> shared soft-token projector into text space. Exposed from
`zeromodels.models.gemma4_unified` for custom pipelines.

### `Gemma4UnifiedImageProcessor`

Image processor in pure Keras ops: aspect-ratio-preserving resize into a patch
budget, then merge each `pooling_kernel_size x pooling_kernel_size` block of teacher
patches into one 48px model patch. Returns `pixel_values`, `image_position_ids`, and
`num_soft_tokens_per_image`.

```python
Gemma4UnifiedImageProcessor(patch_size=16, max_soft_tokens=280, pooling_kernel_size=3)
```

### `Gemma4UnifiedAudioFeatureExtractor`

Audio feature extractor in pure Keras ops: chunk raw 16 kHz audio into fixed
`audio_samples_per_token` (640, 40 ms) frames. No mel, no downsampling. Returns
`input_features` and `input_features_mask`.

```python
Gemma4UnifiedAudioFeatureExtractor(feature_size=640, audio_samples_per_token=640)
```

### `Gemma4UnifiedTokenizer` and `Gemma4UnifiedProcessor`

`Gemma4UnifiedTokenizer` is the Gemma 4 SentencePiece-BPE tokenizer (same markers as
`Gemma4Tokenizer`). `Gemma4UnifiedProcessor` composes it with the two encoder-free
preprocessors: it renders the chat template and expands each `<|image|>` / `<|audio|>`
marker into its soft-token run (audio has no downsampling, so one token per valid
640-sample frame).

```python
Gemma4UnifiedProcessor(
    hf_id=None, tokenizer=None, image_processor=None, feature_extractor=None
)
```

## End-to-end example

### Single input (text only)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gemma4_unified import (
    Gemma4UnifiedConditionalGenerate,
    Gemma4UnifiedTokenizer,
)

model = Gemma4UnifiedConditionalGenerate.from_weights("zeromodels/gemma-4-12b-it")
tokenizer = Gemma4UnifiedTokenizer.from_weights("zeromodels/gemma-4-12b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Image + audio + text

```python
from PIL import Image
from zeromodels.models.gemma4_unified import (
    Gemma4UnifiedConditionalGenerate,
    Gemma4UnifiedProcessor,
)

model = Gemma4UnifiedConditionalGenerate.from_weights("zeromodels/gemma-4-12b-it")
processor = Gemma4UnifiedProcessor.from_weights("zeromodels/gemma-4-12b-it")

inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("photo.jpg")},
                {"type": "audio", "path": "clip.wav"},
                {"type": "text", "text": "Describe the image and what you hear."},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Loading from the Hub (upstream)

```python
model = Gemma4UnifiedConditionalGenerate.from_weights("hf:google/gemma-4-12B-it")
```

### Lower memory

The 12B fits comfortably in bf16; weight-only quantization shrinks it further. See
[quantization.md](quantization.md):

```python
model = Gemma4UnifiedConditionalGenerate.from_weights(
    "zeromodels/gemma-4-12b-it",
    quantization="int8",
    load_dtype="bfloat16",
)
```
