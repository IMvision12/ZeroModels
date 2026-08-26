# Gemma 3n

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Google's on-device Gemma 3n, ported to pure Keras 3: a novel text decoder paired with
a **MobileNet-V5** vision tower and a **USM** audio tower, for image + audio + text.
A single `Gemma3nConditionalGenerate` drives generation for text, image+text, and
image+audio+text; `Gemma3nTextGenerate` is its text-only counterpart. The text-only
backbone is `Gemma3nTextModel`, and the multimodal backbone is `Gemma3nModel`.

What is new in the Gemma 3n decoder:

- **AltUp (Alternating Updates)**: the hidden state is carried as
  `altup_num_inputs` (4) parallel, magnitude-matched streams; each block runs a learned
  *predict* step before attention and a *correct* step after the MLP, updating the
  streams sparsely.
- **LAuReL (Learned Augmented Residual Layer)**: a low-rank (`laurel_rank`) residual
  branch, normed and added alongside the attention residual.
- **MatFormer**: a nested per-layer MLP width (`mlp_dim` may be a per-layer list), so a
  smaller sub-model (E2B) is a slice of the larger one (E4B).
- **Per-Layer Embeddings**: an auxiliary embedding table
  (`vocab_size_per_layer_input`) projected and folded into every block through a
  gate + projection.
- **Activation sparsity**: a Gaussian top-k gate zeroes most of the GeGLU activation on
  the early layers (`activation_sparsity_pattern`).
- **KV-sharing**: a tail of `num_kv_shared_layers` (15) layers reuse an earlier layer's
  K/V per attention type, with a 5:1 sliding/global schedule and dual rotary bases
  (global `1e6`, local `1e4`).
- **Plain-weight RMSNorm**: the norm weight is used directly (no `1 + w` offset); the
  attention value norm is scaleless.
- **MobileNet-V5 vision tower**: a timm `mobilenetv5_300m_enc` encoder (universal
  inverted bottlenecks, multi-query mobile attention, a multi-scale fusion adapter)
  turns an image into 256 soft tokens.
- **USM audio tower**: a chunked-local-attention Conformer with cumulative group norm
  turns audio frames into up to 188 soft tokens.

Links:

- HF docs: [transformers/model_doc/gemma3n](https://huggingface.co/docs/transformers/model_doc/gemma3n)
- Google: [Gemma 3n model card](https://huggingface.co/google/gemma-3n-E4B)

See also [gemma.md](gemma.md), [gemma2.md](gemma2.md), [gemma3.md](gemma3.md),
[gemma4.md](gemma4.md), [gemma4_unified.md](gemma4_unified.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-it` suffix marks instruction-tuned
checkpoints (use the chat template). Gemma 3n is under the **Gemma license** (gated):
accept the terms on the upstream Google card before downloading.

| Variant | Hub | Architecture | Modalities |
|---|---|---|---|
| `gemma-3n-e2b` | [`zeromodels/gemma-3n-e2b`](https://huggingface.co/zeromodels/gemma-3n-e2b) | MatFormer slice (30 layers) | text + image + audio |
| `gemma-3n-e2b-it` | [`zeromodels/gemma-3n-e2b-it`](https://huggingface.co/zeromodels/gemma-3n-e2b-it) | MatFormer slice (30 layers) | text + image + audio |
| `gemma-3n-e4b` | [`zeromodels/gemma-3n-e4b`](https://huggingface.co/zeromodels/gemma-3n-e4b) | full (35 layers) | text + image + audio |
| `gemma-3n-e4b-it` | [`zeromodels/gemma-3n-e4b-it`](https://huggingface.co/zeromodels/gemma-3n-e4b-it) | full (35 layers) | text + image + audio |

`Gemma3nConditionalGenerate` builds the towers automatically from the checkpoint:
the MobileNet-V5 vision tower and the USM audio tower for every variant. E2B is a
MatFormer sub-model of E4B (fewer layers, narrower per-layer MLPs), loaded the same way.

Upstream Google safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:google/gemma-3n-E2B-it")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

Configs are typed: `Gemma3nConfig` (composite) over `Gemma3nTextConfig`,
`Gemma3nVisionConfig`, and `Gemma3nAudioConfig`; each repo's `zm_config.json` parses
through them.

### `Gemma3nTextModel`

The text decoder backbone, no LM head. Returns
`{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `262400` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `mlp_dim` | `16384` | GeGLU inner width (a per-layer list selects MatFormer widths) |
| `num_layers` | `35` | decoder blocks |
| `num_heads` | `8` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `sliding_window` | `512` | local attention span |
| `sliding_window_pattern` | `5` | one global layer every N (when `layer_types` is unset) |
| `layer_types` | `None` | explicit per-layer sliding/global schedule |
| `final_logit_softcapping` | `30.0` | tanh cap on output logits |
| `rope_theta` / `rope_local_theta` | `1e6` / `1e4` | global / local rotary base |
| `vocab_size_per_layer_input` | `262144` | Per-Layer Embedding vocabulary |
| `hidden_size_per_layer_input` | `256` | Per-Layer Embedding width |
| `altup_num_inputs` | `4` | parallel AltUp hidden streams |
| `num_kv_shared_layers` | `15` | tail layers that reuse an earlier layer's K/V |
| `laurel_rank` | `64` | inner rank of the LAuReL residual |
| `activation_sparsity_pattern` | `None` | per-layer Gaussian top-k sparsity (default: 0.95 on the first 10 layers) |

### `Gemma3nConditionalGenerate`

The single generation class, over the `Gemma3nModel` backbone plus a (tied) LM head with
final-logit softcapping. Returns `{"logits": (batch, seq, vocab_size)}` and adds
`.generate()`. Text-only when no image/audio tensors are passed, multimodal when they
are; the multimodal prefill fuses the soft tokens, and decoding is text-only over the
per-layer sliding/global KV cache.

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    pixel_values=None,
    input_features=None,
    input_features_mask=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` / `seed` | `None` | sampling strategy (greedy when unset) / seed |
| `pixel_values` | `None` | `(num_images, H, W, 3)` normalized image pixels |
| `input_features` / `input_features_mask` | `None` | audio mel frames + valid mask |

### `Gemma3nTextGenerate`

The text-only counterpart of `Gemma3nConditionalGenerate`: the Gemma 3n text decoder plus
a (tied) LM head, built with no vision / audio tower. `.generate()` takes just token ids.
The Gemma 3n checkpoints are all multimodal, so this head extracts just their text
backbone from the checkpoint (its `FULL_CHECKPOINT_SOURCES` builds the full model and
copies the decoder weights out, dropping the towers). Set `config_class = Gemma3nTextConfig`.

```python
from zeromodels.models.gemma3n import Gemma3nTextGenerate, Gemma3nTokenizer

model = Gemma3nTextGenerate.from_weights("zeromodels/gemma-3n-e2b-it")
tokenizer = Gemma3nTokenizer.from_weights("zeromodels/gemma-3n-e2b-it")
outputs = model.generate(
    **tokenizer([{"role": "user", "content": "What is on-device inference?"}]),
    max_new_tokens=64,
)
print(tokenizer.decode(outputs[0]))
```

### `Gemma3nModel`

The multimodal backbone: the `Gemma3nTextModel` decoder plus the MobileNet-V5 vision
tower (`MobileNetV5Encoder`) and the USM audio tower (`Gemma3nAudioEncoder`), with the
soft-token projectors (`Gemma3nMultimodalEmbedder`). Image / audio soft tokens are
scattered onto their placeholder positions in `input_ids` before the decoder runs.

| Arg | Default | Meaning |
|---|---|---|
| `text_config` | `None` | dict of `Gemma3nTextModel` constructor args |
| `vision_config` | `None` | dict of vision-tower args; `None` skips the vision tower |
| `audio_config` | `None` | dict of `Gemma3nAudioEncoder` args; `None` skips the audio tower |
| `image_token_id` | `262145` | placeholder id filled with image soft tokens |
| `audio_token_id` | `262273` | placeholder id filled with audio soft tokens |
| `vision_soft_tokens_per_image` | `256` | image soft tokens per image |
| `audio_soft_tokens_per_image` | `188` | audio soft tokens per clip |
| `tie_word_embeddings` | `True` | tie the LM head to the token embedding |

### `MobileNetV5Encoder` and `Gemma3nAudioEncoder`

The towers can be used on their own. `MobileNetV5Encoder` takes
`pixel_values` (channels-last `(batch, H, W, 3)`, or `channels_first` `(batch, 3, H, W)`)
and returns a `(batch, 16, 16, 2048)` feature map (256
soft tokens after flatten); `Gemma3nAudioEncoder` takes `(input_features, mask)` and
returns `(soft_tokens, valid_mask)`. Both are exposed from `zeromodels.models.gemma3n`.

### `Gemma3nImageProcessor`

SigLIP-style image processor in pure Keras ops: fixed-size square resize (768, bilinear),
rescale to `[0, 1]`, and normalize with `image_mean` / `image_std` (0.5). Returns
channels-last `pixel_values`.

```python
Gemma3nImageProcessor(size=768, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
```

### `Gemma3nAudioFeatureExtractor`

USM log-mel feature extractor in pure Keras ops (HTK pre-emphasis, periodic-Hann STFT,
HTK mel filterbank, `log(max(mel, 1e-5))`). Returns `input_features` and
`input_features_mask` (True = valid frames).

### `Gemma3nTokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend; batching (the `<bos>` prefix and
padding) runs through the backend's own post-processor.

```python
Gemma3nTokenizer(hf_id=None, tokenizer_file=None)
```

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It accepts
a string, a list of strings, or a chat-message list (routed through `apply_chat_template`).

### `Gemma3nProcessor`

Text + image + audio to model inputs. Renders the chat template, preprocesses images and
audio, and expands each `<image_soft_token>` / `<audio_soft_token>` marker into its fixed
soft-token run (256 image, 188 audio).

```python
Gemma3nProcessor(
    hf_id=None, tokenizer=None, image_processor=None, feature_extractor=None
)
```

## Data Format

The model reads `keras.config.image_data_format()` when it is constructed and accepts both
`channels_last` and `channels_first` `pixel_values`. For `channels_first` input, the
MobileNet-V5 vision tower transposes the image to `channels_last` at its entry (the "door")
and runs all convolutions in `channels_last` internally. The soft image tokens are identical
under either layout.

## End-to-end example

### Single input (text only)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gemma3n import (
    Gemma3nConditionalGenerate,
    Gemma3nTokenizer,
)

model = Gemma3nConditionalGenerate.from_weights("zeromodels/gemma-3n-e2b-it")
tokenizer = Gemma3nTokenizer.from_weights("zeromodels/gemma-3n-e2b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Single input (image + text)

```python
from PIL import Image
from zeromodels.models.gemma3n import (
    Gemma3nConditionalGenerate,
    Gemma3nProcessor,
)

model = Gemma3nConditionalGenerate.from_weights("zeromodels/gemma-3n-e4b-it")
processor = Gemma3nProcessor.from_weights("zeromodels/gemma-3n-e4b-it")

inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("photo.jpg")},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Image + audio + text

Mix image, audio, and text in one prompt:

```python
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
from zeromodels.models.gemma3n import Gemma3nTextModel

backbone = Gemma3nTextModel.from_weights("zeromodels/gemma-3n-e2b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub (upstream)

```python
model = Gemma3nConditionalGenerate.from_weights("hf:google/gemma-3n-E2B-it")
model = Gemma3nConditionalGenerate.from_weights("hf:google/gemma-3n-E4B-it")
```

### Lower memory

Load in int8 to shrink the resident weights (the per-layer embedding table is large on
E4B). See [quantization.md](quantization.md):

```python
model = Gemma3nConditionalGenerate.from_weights(
    "zeromodels/gemma-3n-e4b-it",
    quantization="int8",
    load_dtype="bfloat16",
)
```
