# Speech2Text

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Speech2Text (S2T) is fairseq's convolution-plus-transformer encoder-decoder for
end-to-end speech recognition and speech translation. It predates the huge weakly
supervised models: these checkpoints are trained on LibriSpeech alone, which makes them
small, fast, and narrowly specialised.

Two details set it apart from [Whisper](whisper.md). The input is **80-channel log-mel
filterbank features with per-utterance mean/variance normalization**, not a fixed 30-second
mel window, so the encoder length tracks the audio. And a 1-D convolutional subsampler
(kernel 5, stride 2, GLU) downsamples time by **4x** before the transformer stack, which is
what keeps a full utterance affordable.

**Paper**: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171)

## API

### Speech2TextConditionalGenerate

```python
Speech2TextConditionalGenerate(
    hidden_dim=256,
    encoder_num_layers=12,
    decoder_num_layers=6,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048,
    vocab_size=10000,
    num_mel_bins=80,
    max_source_positions=6000,
    max_target_positions=1024,
    conv_channels=1024,
    conv_kernel_sizes=(5, 5),
    num_conv_layers=2,
    scale_embedding=True,
    activation_function="relu",
    layer_norm_eps=1e-05,
    name="Speech2TextConditionalGenerate",
)
```

The conv subsampler, encoder, decoder, and LM head, plus a `generate` that runs the whole
pipeline. **This is the class for speech to text.**

**Parameters**

- **hidden_dim** (`int`, *optional*, defaults to `256`): model width. Filled in by `from_weights` from the variant config.
- **encoder_num_layers** / **decoder_num_layers** (`int`, *optional*): blocks per stack, the main size lever from small to large.
- **encoder_attention_heads** / **decoder_attention_heads** (`int`, *optional*): attention heads per stack.
- **encoder_ffn_dim** / **decoder_ffn_dim** (`int`, *optional*): MLP inner width.
- **vocab_size** (`int`, *optional*, defaults to `10000`): SentencePiece vocabulary.
- **num_mel_bins** (`int`, *optional*, defaults to `80`): filterbank channels the encoder expects.
- **max_source_positions** / **max_target_positions** (`int`, *optional*): encoder and decoder position limits.
- **conv_channels** / **conv_kernel_sizes** / **num_conv_layers**: the 1-D subsampler that downsamples time by 4x before the transformer stack.
- **scale_embedding** (`bool`, *optional*, defaults to `True`): scale embeddings by `sqrt(hidden_dim)`.
- **activation_function** / **layer_norm_eps**: block-level knobs, set from the variant config.
- **name** (`str`, *optional*, defaults to `"Speech2TextConditionalGenerate"`): model name.

**Call** `model({"input_features": ..., "decoder_input_ids": ...})` for a teacher-forced
forward pass. **Returns** a `dict` with **logits** `(B, T, vocab_size)` and
**encoder_hidden_states** `(B, T', hidden_dim)`. For transcription use `generate`.

**generate**

```python
model.generate(
    audio, processor, max_new_tokens=200, sampling_rate=16000, return_ids=False
)
```

- **audio**: a 1-D float32 waveform in `[-1, 1]`, or a list of them for a batch.
- **processor** (`Speech2TextProcessor`): supplies the feature extractor and tokenizer.
- **max_new_tokens** (`int`, *optional*, defaults to `200`): decode budget.
- **sampling_rate** (`int`, *optional*, defaults to `16000`): sample rate of `audio`.
- **return_ids** (`bool`, *optional*, defaults to `False`): return token ids instead of strings.

**Returns** a list of strings, one per clip.

There is no `language` or `task` argument: unlike Whisper, an S2T checkpoint does exactly
the one job it was trained for.

### Speech2TextModel

```python
Speech2TextModel(hidden_dim=256, encoder_num_layers=12, decoder_num_layers=6,
                 ..., name="Speech2TextModel")
```

The encoder-decoder without the LM head, for features or a custom head. Same arguments as
`Speech2TextConditionalGenerate`.

## Preprocessing

### Speech2TextFeatureExtractor

```python
Speech2TextFeatureExtractor(
    sampling_rate=16000,
    num_mel_bins=80,
    frame_length_ms=25.0,
    frame_shift_ms=10.0,
    preemphasis=0.97,
    normalize_means=True,
    normalize_vars=True,
)
```

Computes log-mel filterbank features on 25 ms frames every 10 ms, then normalizes each
utterance.

**Parameters**

- **sampling_rate** (`int`, *optional*, defaults to `16000`): rate the model was trained at. Resample your audio to match; this does not resample for you.
- **num_mel_bins** (`int`, *optional*, defaults to `80`): filterbank channels.
- **frame_length_ms** / **frame_shift_ms** (`float`, *optional*, defaults to `25.0` / `10.0`): window and hop.
- **preemphasis** (`float`, *optional*, defaults to `0.97`): high-pass applied before framing.
- **normalize_means** / **normalize_vars** (`bool`, *optional*, defaults to `True`): per-utterance mean and variance normalization.

`feat(raw_audio)` **returns** `(B, frames, num_mel_bins)`.

> **Normalization is per utterance, not global.** Each clip is standardized against its own
> statistics, so a quiet recording and a loud one arrive at the encoder on the same scale
> without any gain matching from you.

### Speech2TextProcessor

```python
Speech2TextProcessor(
    vocab_file=None,
    spm_file=None,
    sampling_rate=16000,
    num_mel_bins=80,
    do_upper_case=False,
    do_lower_case=False,
    decoder_start_token_id=2,
    tokenizer=None,
    feature_extractor=None,
)
```

Bundles the feature extractor and the SentencePiece tokenizer.

**Call** `processor(audio=..., text=...)`. **Returns** `input_features` and, when text is
given, `labels`.

> **Prefer `Speech2TextProcessor.from_weights(variant)`.** The SentencePiece vocabulary
> differs between checkpoints, and `generate` needs the one that matches the model.

## Model Variants

| Variant id | Params | Trained on |
|---|---:|---|
| `s2t-small-librispeech-asr` | 30 M | LibriSpeech 960 h |
| `s2t-medium-librispeech-asr` | 71 M | LibriSpeech 960 h |
| `s2t-large-librispeech-asr` | 268 M | LibriSpeech 960 h |

All three are **English ASR only**. They emit lowercase, unpunctuated text, because that is
how LibriSpeech is transcribed.

## Basic Usage: Transcription

The sample below is a LibriSpeech clip, 4.82 s of 16 kHz mono, kept in the repo at
`assets/speech_quilter_manner.wav`:

<audio controls src="../assets/speech_quilter_manner.wav"></audio>

Its reference transcript is *"NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS
MATTER"*.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import soundfile as sf
from zeromodels.models.speech2text import (
    Speech2TextProcessor,
    Speech2TextConditionalGenerate,
)

model = Speech2TextConditionalGenerate.from_weights(
    "zeromodels/s2t-small-librispeech-asr"
)
processor = Speech2TextProcessor.from_weights("zeromodels/s2t-small-librispeech-asr")

audio, sr = sf.read("assets/speech_quilter_manner.wav", dtype="float32")  # 16 kHz mono
text = model.generate(audio, processor)
print(repr(text[0]))
```

```
"nor is mister coleter's manner less interesting than his matter"
```

Lowercase and unpunctuated, the style the LibriSpeech training transcripts use, and note it
keeps the apostrophe in the possessive. The one miss is the proper noun: "coleter" for
"Quilter". Names are exactly where a 30 M model trained on 960 hours struggles, and
[Whisper](whisper.md), trained on 680,000 hours, gets this one right. Compare its output
style too: cased and punctuated on the same kind of audio, which is a convention
difference rather than an accuracy one.

### Batching

Pass a list of waveforms; the extractor pads them to a common length:

```python
clips = [audio, audio[: 3 * sr]]
for line in model.generate(clips, processor):
    print(repr(line))
```

## Audio Format

**A 1-D float32 waveform in `[-1, 1]` at 16 kHz.**

| | What it expects |
|---|---|
| `generate` / processor | A 1-D `float32` array (or a list of them). `sampling_rate` tells it what rate you are handing over; it does not resample. |
| Models | `input_features`, the `(B, frames, 80)` filterbank tensor from the extractor. |

```python
import librosa
import soundfile as sf

audio, sr = sf.read("assets/speech_quilter_manner.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo to mono
if sr != 16000:
    audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
```

Because the encoder length follows the audio rather than a fixed window, there is no
30-second ceiling to work around, but attention still costs quadratic time in the
subsampled length, so cut very long recordings into utterances.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"speech_to_text"` loads with the `hf:` prefix,
including the multilingual speech-translation checkpoints.

```python
from zeromodels.models.speech2text import (
    Speech2TextProcessor,
    Speech2TextConditionalGenerate,
)

model = Speech2TextConditionalGenerate.from_weights(
    "hf:facebook/s2t-small-librispeech-asr"
)
processor = Speech2TextProcessor.from_weights("hf:facebook/s2t-small-librispeech-asr")

# Architecture only, randomly initialized
model = Speech2TextConditionalGenerate.from_weights(
    "zeromodels/s2t-small-librispeech-asr", load_weights=False
)
```

Load the processor from the same source as the model: the SentencePiece vocabulary differs
between checkpoints.

See also [Whisper](whisper.md) for multilingual transcription and translation, and
[Moonshine](moonshine.md) for a latency-oriented alternative.
