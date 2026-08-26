# Moonshine

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Moonshine is built for live transcription and voice commands, where latency matters more
than covering 99 languages. Two choices follow from that.

It eats the **raw 16 kHz waveform**: a three-layer Conv1D stem replaces the log-mel front
end entirely, so there is no spectrogram step. And it has **no fixed input window**, unlike
[Whisper](whisper.md)'s mandatory 30-second pad. A one-second command costs one second of
compute, which is the whole point: Whisper pads that same clip to 30 s and pays for the
silence.

Positions come from GLM-style **partial** rotary embeddings, the decoder MLP is gated
SiLU, and every LayerNorm is scale-only.

**Paper**: [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)

## API

### MoonshineConditionalGenerate

```python
MoonshineConditionalGenerate(
    hidden_dim=288,
    encoder_num_layers=6,
    decoder_num_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    encoder_num_kv_heads=None,
    decoder_num_kv_heads=None,
    encoder_ffn_dim=1152,
    decoder_ffn_dim=1152,
    vocab_size=32768,
    max_position_embeddings=194,
    partial_rotary_factor=0.9,
    rope_theta=10000.0,
    encoder_activation="gelu",
    decoder_activation="silu",
    layer_norm_eps=1e-05,
    name="MoonshineConditionalGenerate",
)
```

The conv stem, encoder, decoder, and tied LM head, plus a `generate` that runs the whole
pipeline. **This is the class for speech to text.**

**Parameters**

- **hidden_dim** (`int`, *optional*, defaults to `288`): model width. Filled in by `from_weights` from the variant config.
- **encoder_num_layers** / **decoder_num_layers** (`int`, *optional*): blocks per stack, the size lever between tiny and base.
- **encoder_attention_heads** / **decoder_attention_heads** (`int`, *optional*): attention heads per stack.
- **encoder_num_kv_heads** / **decoder_num_kv_heads** (`int`, *optional*): grouped-query KV heads. `None` means full multi-head.
- **encoder_ffn_dim** / **decoder_ffn_dim** (`int`, *optional*): MLP inner width. The decoder MLP is gated SiLU, so its `fc1` projects to twice this.
- **vocab_size** (`int`, *optional*, defaults to `32768`): BPE vocabulary.
- **partial_rotary_factor** (`float`, *optional*, defaults to `0.9`): fraction of each head rotated by RoPE; the rest passes through.
- **rope_theta** (`float`, *optional*, defaults to `10000.0`): RoPE base frequency.
- **max_position_embeddings** (`int`, *optional*, defaults to `194`): size of the stored rotary tables. It does **not** cap the audio you can transcribe, see [Audio Format](#audio-format).
- **encoder_activation** / **decoder_activation** / **layer_norm_eps**: block-level knobs, set from the variant config.
- **name** (`str`, *optional*, defaults to `"MoonshineConditionalGenerate"`): model name.

**Call** `model({"input_values": ..., "decoder_input_ids": ...})` for a teacher-forced
forward pass. **Returns** a `dict` with **logits** `(B, T, vocab_size)` and
**encoder_hidden_states** `(B, T', hidden_dim)`. For transcription use `generate`.

**generate**

```python
model.generate(
    audio, processor, max_new_tokens=200, sampling_rate=16000, return_ids=False
)
```

- **audio**: a 1-D float32 waveform in `[-1, 1]`, or a list of them for a batch.
- **processor** (`MoonshineProcessor`): supplies the feature extractor and tokenizer.
- **max_new_tokens** (`int`, *optional*, defaults to `200`): decode budget.
- **sampling_rate** (`int`, *optional*, defaults to `16000`): sample rate of `audio`.
- **return_ids** (`bool`, *optional*, defaults to `False`): return token ids instead of strings.

**Returns** a list of strings, one per clip.

### MoonshineModel

```python
MoonshineModel(hidden_dim=288, encoder_num_layers=6, decoder_num_layers=6, ...,
               name="MoonshineModel")
```

The encoder-decoder without the LM head, for features or a custom head. Same arguments as
`MoonshineConditionalGenerate`.

## Preprocessing

### MoonshineFeatureExtractor

```python
MoonshineFeatureExtractor(sampling_rate=16000, padding_value=0.0)
```

There is no spectrogram here. The extractor only batches and pads waveforms; the conv stem
does the rest.

**Parameters**

- **sampling_rate** (`int`, *optional*, defaults to `16000`): rate the model was trained at. Resample your audio to match; this does not resample for you.
- **padding_value** (`float`, *optional*, defaults to `0.0`): value used to pad a batch to its longest clip.

`feat(raw_audio)` **returns** `(B, num_samples)` float32.

### MoonshineProcessor

```python
MoonshineProcessor(
    variant=None,
    tokenizer_file=None,
    sampling_rate=16000,
    decoder_start_token_id=1,
    bos_token_id=1,
    eos_token_id=2,
    unk_token_id=0,
    tokenizer=None,
    feature_extractor=None,
)
```

Bundles the feature extractor and tokenizer.

**Call** `processor(audio=..., text=...)`. **Returns** `input_values` and, when text is
given, `labels`.

> **Prefer `MoonshineProcessor.from_weights(variant)`.** It fetches the vocabulary that
> matches the checkpoint, which is what `generate` expects.

## Model Variants

| Variant id | Params |
|---|---:|
| `moonshine_tiny` | 27 M |
| `moonshine_base` | 61 M |

Both are **English only**. Even `moonshine_base` is smaller than `whisper_base`, and
because there is no 30-second pad the gap in wall-clock time on short clips is far larger
than the parameter counts suggest.

## Basic Usage: Transcription

The sample below is a LibriSpeech clip, 5.12 s of 16 kHz mono, kept in the repo at
`assets/speech_etchings.wav`:

<audio controls src="../assets/speech_etchings.wav"></audio>

Its reference transcript is *"AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN"*.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import soundfile as sf
from zeromodels.models.moonshine import (
    MoonshineProcessor,
    MoonshineConditionalGenerate,
)

model = MoonshineConditionalGenerate.from_weights("zeromodels/moonshine_tiny")
processor = MoonshineProcessor.from_weights("zeromodels/moonshine_tiny")

audio, sr = sf.read("assets/speech_etchings.wav", dtype="float32")  # 16 kHz mono
text = model.generate(audio, processor)
print(repr(text[0]))
```

```
'As for etchings, there are of two kinds, British and foreign.'
```

Cased and punctuated, like Whisper and unlike [Speech2Text](speech2text.md). The reference
reads "they are of two kinds"; tiny hears "there are of", leaving an ungrammatical "are
of". `moonshine_base` cleans that to `'As for etchings, there are two kinds, British and
foreign.'`, which reads properly but still swaps "they" for "there".

### Short clips are the point

Because nothing is padded to a fixed window, cost scales with the audio you actually have:

```python
command = audio[: 1 * sr]  # one second
print(repr(model.generate(command, processor)[0]))
```

```
'As for.'
```

One second of audio is one second of encoder work; Whisper would pad the same clip to 30 s
before the encoder ever ran. The transcript is short because one second of this sentence
really is just "As for", which is the honest answer to what was said.

### Batching

Pass a list of waveforms; the extractor pads them to the longest in the batch:

```python
clips = [audio, audio[: 3 * sr]]
for line in model.generate(clips, processor):
    print(repr(line))
```

Padding is per batch, so grouping clips of similar length wastes less compute.

## Audio Format

**A 1-D float32 waveform in `[-1, 1]` at 16 kHz, fed to the model as-is.**

| | What it expects |
|---|---|
| `generate` / processor | A 1-D `float32` array (or a list of them). `sampling_rate` tells it what rate you are handing over; it does not resample. |
| Models | `input_values`, the raw `(B, num_samples)` waveform. No spectrogram step. |

```python
import librosa
import soundfile as sf

audio, sr = sf.read("assets/speech_etchings.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo to mono
if sr != 16000:
    audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
```

There is **no duration cap**: the rotary tables are derived for whatever length the encoder
produces, so `max_position_embeddings` sizes the stored tables but does not limit the
audio. Attention cost still grows with duration, and Moonshine is aimed at utterances and
commands, so cut long recordings into segments rather than feeding a whole meeting.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"moonshine"` loads with the `hf:` prefix.

```python
from zeromodels.models.moonshine import (
    MoonshineProcessor,
    MoonshineConditionalGenerate,
)

model = MoonshineConditionalGenerate.from_weights("hf:UsefulSensors/moonshine-tiny")
processor = MoonshineProcessor.from_weights("hf:UsefulSensors/moonshine-tiny")

# Architecture only, randomly initialized
model = MoonshineConditionalGenerate.from_weights(
    "zeromodels/moonshine_tiny", load_weights=False
)
```

See also [Whisper](whisper.md) for multilingual coverage, and [Speech2Text](speech2text.md)
for the fairseq lineage.
