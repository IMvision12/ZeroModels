# Gemma 4

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The fourth Gemma generation, ported to pure Keras 3: dense and Mixture-of-Experts
text models paired with a NaViT vision tower and a USM audio tower, with a 256K
context window and the "Elastic" **E-variants** (per-layer embeddings + KV-sharing).
A single `Gemma4ConditionalGenerate` drives generation for text-only, image+text, and
image+audio+text checkpoints; `Gemma4TextGenerate` is its text-only counterpart (built
with no towers). The backbones are `Gemma4Model` (text decoder) and
`Gemma4MultimodalModel` (vision + audio + text).

The **12B** checkpoints are a separate, encoder-free architecture
(`gemma4_unified`) and live on their own page: see
[gemma4_unified.md](gemma4_unified.md).

What changed from Gemma 3:

- **Plain-weight RMSNorm**: the norm weight is used directly, without Gemma's usual
  `1 + w` offset.
- **Zero-pad partial rope**: only `partial_rotary_factor` (0.25) of each head is
  rotated; the remainder is passed through.
- **K=V global MQA**: global layers use a wider `global_head_dim` (512) with few K/V
  heads; on the dense/MoE sizes key and value projections are tied (`k_eq_v=True`).
- **Parallel MoE**: the MoE variant runs experts alongside the dense MLP rather than
  replacing it.
- **NaViT vision tower**: patch soft tokens from a 2-D-rotary ViT are scattered onto
  the image placeholders and attend bidirectionally within each image block on the
  sliding-window layers (global layers stay causal).
- **USM audio tower**: a chunked-local-attention Conformer turns audio frames into
  soft tokens, scattered onto the audio placeholders (E2B / E4B).
- **Elastic E-variants**: E2B / E4B add Per-Layer Embeddings
  (`hidden_size_per_layer_input`), a tail of layers that reuse an earlier layer's K/V
  (`num_kv_shared_layers`), and a double-wide MLP on those shared layers. Their
  sliding/global schedule is an explicit `layer_types` list (globals on a 5:1
  pattern, unlike the 6-layer pattern of 26B / 31B).

Links:

- Paper: [Gemma 4 Technical Report (arXiv:2607.02770)](https://arxiv.org/abs/2607.02770)
- HF docs: [transformers/model_doc/gemma4](https://huggingface.co/docs/transformers/model_doc/gemma4)

See also [gemma.md](gemma.md), [gemma2.md](gemma2.md), [gemma3.md](gemma3.md),
[gemma4_unified.md](gemma4_unified.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-it` suffix marks instruction-tuned
checkpoints (use the chat template). Gemma 4 is Apache 2.0.

| Variant | Hub | Architecture | Modalities |
|---|---|---|---|
| `gemma-4-e2b` | [`zeromodels/gemma-4-e2b`](https://huggingface.co/zeromodels/gemma-4-e2b) | Elastic (PLE + KV-share) | text + image + audio |
| `gemma-4-e2b-it` | [`zeromodels/gemma-4-e2b-it`](https://huggingface.co/zeromodels/gemma-4-e2b-it) | Elastic (PLE + KV-share) | text + image + audio |
| `gemma-4-e4b` | [`zeromodels/gemma-4-e4b`](https://huggingface.co/zeromodels/gemma-4-e4b) | Elastic (PLE + KV-share) | text + image + audio |
| `gemma-4-e4b-it` | [`zeromodels/gemma-4-e4b-it`](https://huggingface.co/zeromodels/gemma-4-e4b-it) | Elastic (PLE + KV-share) | text + image + audio |
| `gemma-4-26b-a4b` | [`zeromodels/gemma-4-26b-a4b`](https://huggingface.co/zeromodels/gemma-4-26b-a4b) | MoE, 26B total / 4B active | text + image |
| `gemma-4-26b-a4b-it` | [`zeromodels/gemma-4-26b-a4b-it`](https://huggingface.co/zeromodels/gemma-4-26b-a4b-it) | MoE, 26B total / 4B active | text + image |
| `gemma-4-31b` | [`zeromodels/gemma-4-31b`](https://huggingface.co/zeromodels/gemma-4-31b) | dense | text + image |
| `gemma-4-31b-it` | [`zeromodels/gemma-4-31b-it`](https://huggingface.co/zeromodels/gemma-4-31b-it) | dense | text + image |

`Gemma4ConditionalGenerate` builds the towers automatically from the checkpoint: the NaViT
vision tower for every variant, plus the USM audio tower for E2B / E4B.

Note that MoE memory is governed by **total** parameters, not active ones: every
expert is resident, so `gemma-4-26b-a4b-it` needs room for 26B weights even though
only 4B are used per token.

Upstream Google safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:google/gemma-4-E2B-it")`, which converts them in process (pass
`cache_converted=True` to keep the result). See
[Loading Weights](loading_weights.md).

## API

Configs are typed: `Gemma4Config` (composite) over `Gemma4TextConfig`,
`Gemma4VisionConfig`, and `Gemma4AudioConfig`; each repo's `kf_config.json` parses
through them.

### `Gemma4Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `262144` | token vocabulary size |
| `embed_dim` | `3840` | model width |
| `mlp_dim` | `15360` | GeGLU inner width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads on local layers (GQA) |
| `num_global_kv_heads` | `1` | key/value heads on global layers |
| `head_dim` | `256` | per-head width on local layers |
| `global_head_dim` | `512` | per-head width on global layers |
| `k_eq_v` | `True` | tie the key and value projections on global layers |
| `sliding_window` | `1024` | local attention span |
| `sliding_window_pattern` | `6` | one global layer every N (when `layer_types` is unset) |
| `layer_types` | `None` | explicit per-layer sliding/global schedule (E2B/E4B) |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `final_logit_softcapping` | `30.0` | tanh cap on output logits |
| `enable_moe` | `False` | turn on the parallel MoE block |
| `num_experts` / `num_experts_per_tok` / `moe_mlp_dim` | `0` | MoE parameters |
| `hidden_size_per_layer_input` | `0` | Per-Layer Embedding width (256 on E2B/E4B; 0 disables) |
| `num_kv_shared_layers` | `0` | tail layers that reuse an earlier layer's K/V |
| `use_double_wide_mlp` | `False` | double-width MLP on the KV-shared layers |

### `Gemma4ConditionalGenerate`

The single generation class, over the `Gemma4MultimodalModel` backbone plus a
(tied) LM head with final-logit softcapping. Returns
`{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Text-only when no
image/audio tensors are passed, multimodal when they are; the multimodal prefill
fuses the soft tokens and applies the blockwise vision mask, and decoding is always
text-only over the per-layer KV cache.

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

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` / `seed` | `None` | sampling strategy (greedy when unset) / seed |
| `pixel_values` / `pixel_position_ids` | `None` | image patches + 2-D coords |
| `input_features` / `input_features_mask` | `None` | audio frames + mask |

### `Gemma4TextGenerate`

The text-only counterpart of `Gemma4ConditionalGenerate`: the `Gemma4Model` decoder plus
a (tied) LM head, built with no vision / audio tower. `.generate()` takes just token ids
(no `pixel_values` / `input_features`). It shares the decoder weights with
`Gemma4ConditionalGenerate`, so a text-only Gemma 4 repo loads under either head. Set
`config_class = Gemma4TextConfig`.

```python
from zeromodels.models.gemma4 import Gemma4TextGenerate, Gemma4Tokenizer

model = Gemma4TextGenerate.from_weights("zeromodels/gemma-4-e4b-it")
tokenizer = Gemma4Tokenizer.from_weights("zeromodels/gemma-4-e4b-it")
outputs = model.generate(
    **tokenizer([{"role": "user", "content": "Explain MoE routing in one sentence."}]),
    max_new_tokens=64,
)
print(tokenizer.decode(outputs[0]))
```

### `Gemma4MultimodalModel`

The multimodal backbone: the `Gemma4Model` text decoder plus the NaViT vision tower
(`Gemma4VisionModel`) and, on E2B / E4B, the USM audio tower (`Gemma4AudioModel`),
with the soft-token projectors. Image / audio soft tokens are scattered onto their
placeholder positions in `input_ids` before the decoder runs.

| Arg | Default | Meaning |
|---|---|---|
| `text_config` | `None` | dict of `Gemma4Model` constructor args |
| `vision_config` | `None` | dict of `Gemma4VisionModel` args; `None` skips the vision tower |
| `audio_config` | `None` | dict of `Gemma4AudioModel` args; `None` skips the audio tower |
| `image_token_id` | `258880` | placeholder id filled with image soft tokens |
| `video_token_id` | `258884` | placeholder id filled with video soft tokens |
| `audio_token_id` | `258881` | placeholder id filled with audio soft tokens |
| `pad_token_id` | `0` | id used to embed placeholder slots before the scatter |
| `use_bidirectional_vision` | `True` | blockwise bidirectional attention within each image block |

### `Gemma4VisionModel` and `Gemma4AudioModel`

The towers can be used on their own. `Gemma4VisionModel` takes
`(pixel_values, pixel_position_ids)` and returns pooled soft tokens;
`Gemma4AudioModel` takes `(input_features, input_features_mask)` and returns
`(soft_tokens, valid_mask)`. Both are exposed from `zeromodels.models.gemma4`.

### `Gemma4ImageProcessor`

NaViT image processor in pure Keras ops: aspect-ratio-preserving resize into a patch
budget, rescale to `[0, 1]`, patchify, and pad. Returns `pixel_values`,
`image_position_ids`, and `num_soft_tokens_per_image`.

```python
Gemma4ImageProcessor(patch_size=16, max_soft_tokens=280, pooling_kernel_size=3)
```

### `Gemma4AudioFeatureExtractor`

USM log-mel feature extractor in pure Keras ops (periodic-Hann STFT, HTK mel
filterbank, `log(mel + 1e-3)`). Returns `input_features` and `input_features_mask`.

### `Gemma4Tokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend; batching (the `<bos>`
prefix and padding) runs through the backend's own post-processor.

```python
Gemma4Tokenizer(hf_id=None, tokenizer_file=None)
```

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a string, a list of strings, or a chat-message list (routed through
`apply_chat_template`).

### `Gemma4Processor`

Text + image + audio to model inputs. Renders the chat template, preprocesses images
and audio, and expands each `<|image|>` / `<|audio|>` marker into its soft-token run.

```python
Gemma4Processor(
    hf_id=None, tokenizer=None, image_processor=None, feature_extractor=None
)
```

## End-to-end example

### Single input (text only)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from zeromodels.models.gemma4 import Gemma4ConditionalGenerate, Gemma4Tokenizer

model = Gemma4ConditionalGenerate.from_weights("zeromodels/gemma-4-e2b-it")
tokenizer = Gemma4Tokenizer.from_weights("zeromodels/gemma-4-e2b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Single input (image + text)

Use `Gemma4Processor` with any variant (all carry the vision tower):

```python
from PIL import Image
from zeromodels.models.gemma4 import Gemma4ConditionalGenerate, Gemma4Processor

model = Gemma4ConditionalGenerate.from_weights("zeromodels/gemma-4-31b-it")
processor = Gemma4Processor.from_weights("zeromodels/gemma-4-31b-it")

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

### Image + audio + text (E-variants)

E2B / E4B are any-to-any: mix image, audio, and text in one prompt.

```python
model = Gemma4ConditionalGenerate.from_weights("zeromodels/gemma-4-e4b-it")
processor = Gemma4Processor.from_weights("zeromodels/gemma-4-e4b-it")

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
from zeromodels.models.gemma4 import Gemma4Model

backbone = Gemma4Model.from_weights("zeromodels/gemma-4-e2b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, embed_dim)
```

### Loading from the Hub (upstream)

```python
model = Gemma4ConditionalGenerate.from_weights("hf:google/gemma-4-E2B-it")
model = Gemma4ConditionalGenerate.from_weights("hf:google/gemma-4-31B-it")
```

### Lower memory

The 31B dense and 26B MoE checkpoints need quantization to fit on a single 80GB GPU
at full precision. See [quantization.md](quantization.md):

```python
model = Gemma4ConditionalGenerate.from_weights(
    "zeromodels/gemma-4-31b-it",
    quantization="int8",
    load_dtype="bfloat16",
)
```
