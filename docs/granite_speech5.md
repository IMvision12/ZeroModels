# Granite Speech 5

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.json</code> shards
plus <code>tokenizer.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Granite Speech 5.0 **turboctc** is a fast, **non-autoregressive CTC** ASR model, not a
speech-aware LLM like [Granite Speech](granite_speech.md). It is a conformer encoder,
block-wise self-attention with Shaw relative positional embeddings and two early
time-subsampling blocks, capped with a self-conditioned CTC head. The whole utterance is
transcribed in a single forward pass (no decoder loop); greedy CTC decoding, collapsing
repeats and dropping the blank token, turns the per-frame argmax into text.

**Paper**: [Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities](https://arxiv.org/abs/2505.08699)

## API

### GraniteSpeech5CTC

```python
GraniteSpeech5CTC(
    vocab_size=16384,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=16,
    num_attention_heads=8,
    head_dim=128,
    num_mel_bins=80,
    hidden_act="silu",
    max_position_embeddings=512,
    context_size=128,
    conv_kernel_size=7,
    conv_expansion_factor=2,
    subsample_layers=(0, 1),
    attention_bias=True,
    pad_token_id=0,
    name="GraniteSpeech5CTC",
)
```

The conformer encoder plus the CTC head (tied to the encoder's mid-layer self-conditioning
projection). **This is the class for speech-to-text.** Takes a dict of `input_features`
`(B, frames, num_mel_bins * 4)` and `attention_mask` `(B, frames)` (from
`GraniteSpeech5FeatureExtractor`) and returns `logits` `(B, frames // 4, vocab_size)`.

**Parameters**

- **vocab_size** (`int`, *optional*, defaults to `16384`): CTC output vocabulary size.
- **hidden_size** / **intermediate_size** / **num_hidden_layers** / **num_attention_heads** / **head_dim** (`int`, *optional*): conformer shape. Filled in by `from_weights` from the variant config.
- **num_mel_bins** (`int`, *optional*, defaults to `80`): mel bins; the stacked input feature width is `num_mel_bins * 4`.
- **max_position_embeddings** (`int`, *optional*, defaults to `512`): span of the Shaw relative-position table.
- **context_size** (`int`, *optional*, defaults to `128`): block-wise attention window, in frames.
- **conv_kernel_size** / **conv_expansion_factor** (`int`, *optional*): depthwise-conv kernel and channel expansion.
- **subsample_layers** (`tuple`, *optional*, defaults to `(0, 1)`): block indices that subsample time by 2 (total 4x).
- **pad_token_id** (`int`, *optional*, defaults to `0`): CTC blank id.
- **name** (`str`, *optional*, defaults to `"GraniteSpeech5CTC"`): model name.

**generate**

```python
model.generate(inputs)
```

Greedy CTC decoding: per-frame `argmax` of the logits, with padded output frames set to the
blank id. Returns integer ids `(B, frames // 4)`; pass them to
`GraniteSpeech5Tokenizer.batch_decode` (or the processor) to collapse repeats, drop the
blank, and render text. The same ids carry word timing, so
`processor.batch_decode(ids, return_timestamps=True)` gives word-level timestamps (see
[Word Timestamps](#word-timestamps)).

### GraniteSpeech5Model

```python
GraniteSpeech5Model(...)  # same encoder arguments as GraniteSpeech5CTC
```

The conformer encoder backbone alone. Returns `last_hidden_state`
`(B, frames // 4, hidden_size)` (plus the subsampled `output_attention_mask`), without the
CTC projection, for feature extraction or a custom head.

## Preprocessing

### GraniteSpeech5FeatureExtractor

```python
GraniteSpeech5FeatureExtractor(
    sampling_rate=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    num_mel_bins=80,
    delta_win_length=3,
    logmel_floor_db=8.0,
    frame_stacking=2,
)
```

Pure-Keras log-mel(+delta) feature extractor. Computes a `torchaudio`-style mel
spectrogram, floors the log-mel at `max - logmel_floor_db` dB, concatenates each frame with
its time-delta, and stacks consecutive frames in pairs. `call(audio, sampling_rate=16000)`
returns `{"input_features", "attention_mask"}` for 16 kHz mono audio (a single clip or a
batch of clips).

### GraniteSpeech5Tokenizer

```python
GraniteSpeech5Tokenizer.from_weights("zeromodels/granite-speech-5.0-470m-turboctc")
```

The CTC tokenizer (Parakeet-style). `batch_decode(ids)` performs CTC decoding: group
consecutive duplicate ids, drop the blank (`<|blank|>`, id 0), then map the survivors to
text. `tokenize(text)` runs the plain encoder for building CTC training labels.

### GraniteSpeech5Processor

```python
GraniteSpeech5Processor.from_weights("zeromodels/granite-speech-5.0-470m-turboctc")
```

Composes the feature extractor and the tokenizer. `call(audio, text=None, sampling_rate=16000)`
returns `input_features` + `attention_mask` (and padded CTC `labels` when `text` is given);
`batch_decode(ids)` CTC-decodes the model's greedy output, and
`batch_decode(ids, return_timestamps=True)` adds word-level timestamps (see
[Word Timestamps](#word-timestamps)).

## Model Variants

| Variant id | Params | Notes |
| --- | --- | --- |
| `granite-speech-5.0-470m-turboctc` | ~470 M | CTC ASR (English) |

## Basic Usage: Transcription

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import soundfile as sf
from zeromodels.models.granite_speech5 import (
    GraniteSpeech5CTC,
    GraniteSpeech5FeatureExtractor,
    GraniteSpeech5Tokenizer,
)

model = GraniteSpeech5CTC.from_weights("zeromodels/granite-speech-5.0-470m-turboctc")
features = GraniteSpeech5FeatureExtractor()
tokenizer = GraniteSpeech5Tokenizer.from_weights(
    "zeromodels/granite-speech-5.0-470m-turboctc"
)

audio, sr = sf.read("speech.wav", dtype="float32")  # 16 kHz mono
inputs = features(audio, sampling_rate=sr)
predicted_ids = model.generate(inputs)
print(tokenizer.batch_decode(predicted_ids))
```

Or drive it end to end through the processor:

```python
from zeromodels.models.granite_speech5 import GraniteSpeech5Processor

processor = GraniteSpeech5Processor.from_weights(
    "zeromodels/granite-speech-5.0-470m-turboctc"
)
inputs = processor(audio=audio, sampling_rate=sr)
predicted_ids = model.generate(inputs)
print(processor.batch_decode(predicted_ids))
```

## Word Timestamps

Because CTC emits one token per frame, each word can be placed in time from the frames it is
decoded at, no extra model output needed. Pass `return_timestamps=True` to `batch_decode`:
it returns one dict per clip, `{"text", "chunks": [{"text", "timestamp": (start, end)}, ...]}`,
the same shape Whisper uses.

```python
inputs = processor(audio=audio, sampling_rate=sr)
result = processor.batch_decode(model.generate(inputs), return_timestamps=True)[0]

print(result["text"])
for chunk in result["chunks"]:
    print(chunk["timestamp"], chunk["text"])
```

```
as for etchings they are of 2 kinds british and foreign
(0.0, 0.08) as
(0.64, 0.72) for
(0.88, 1.2) etchings
(1.52, 1.6) they
...
```

Each CTC output frame covers `processor.frame_seconds` of audio (0.08 s for the released
model: a 10 ms mel hop, stacked in pairs, then reduced 4x by the encoder's two subsampling
blocks). CTC alignment is spiky, so a timestamp marks roughly where a word lands rather than
a precise start and end. If you build the encoder with a different subsampling depth, set
`GraniteSpeech5Processor(encoder_downsample=...)` so the seconds stay correct.

## Audio Format

|  | What it expects |
| --- | --- |
| Feature extractor / Processor | The 16 kHz mono waveform in `audio=`. `sampling_rate` must be 16000; it does not resample. |
| Models | `input_features` `(B, frames, num_mel_bins * 4)` and `attention_mask` `(B, frames)` from the extractor. |

## Loading Fine-tuned and Community Weights

Upstream and community safetensors load on the fly with the `hf:` prefix (no re-hosting):

```python
model = GraniteSpeech5CTC.from_weights(
    "hf:ibm-granite/granite-speech-5.0-470m-turboctc"
)
tokenizer = GraniteSpeech5Tokenizer.from_hf(
    "ibm-granite/granite-speech-5.0-470m-turboctc"
)
```
