# Whisper

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Whisper is an encoder-decoder transformer trained on 680,000 hours of weakly supervised
audio. It takes a log-mel spectrogram and decodes text, and because the task is expressed
in the prompt (special tokens for language, transcribe-vs-translate, timestamps) one
checkpoint does multilingual transcription, translation into English, and language
identification.

The audio side is fixed: every clip is padded or trimmed to **30 seconds** and turned into
an 80-bin mel spectrogram, so the encoder always sees `(80, 3000)`. Shorter clips are
padded, longer ones need chunking.

**Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)

## API

### WhisperConditionalGenerate

```python
WhisperConditionalGenerate(
    hidden_dim=384,
    encoder_num_layers=4,
    decoder_num_layers=4,
    encoder_attention_heads=6,
    decoder_attention_heads=6,
    encoder_ffn_dim=1536,
    decoder_ffn_dim=1536,
    num_mel_bins=80,
    max_source_positions=1500,
    max_target_positions=448,
    vocab_size=51865,
    activation_function="gelu",
    layer_norm_eps=1e-05,
    scale_embedding=False,
    name="WhisperConditionalGenerate",
)
```

The encoder, decoder, and tied LM head, plus a `generate` that owns the whole
transcription pipeline. **This is the class for speech to text.**

**Parameters**

- **hidden_dim** (`int`, *optional*, defaults to `384`): model width. Filled in by `from_weights` from the variant config.
- **encoder_num_layers** / **decoder_num_layers** (`int`, *optional*): blocks per stack, the main size lever from tiny to large.
- **encoder_attention_heads** / **decoder_attention_heads** (`int`, *optional*): attention heads per stack.
- **encoder_ffn_dim** / **decoder_ffn_dim** (`int`, *optional*): MLP inner width.
- **num_mel_bins** (`int`, *optional*, defaults to `80`): mel channels the encoder expects. `large_v3` uses `128`.
- **max_source_positions** (`int`, *optional*, defaults to `1500`): encoder frames, the 30 s window after the stride-2 conv stem.
- **max_target_positions** (`int`, *optional*, defaults to `448`): decoder context length.
- **vocab_size** (`int`, *optional*, defaults to `51865`): BPE vocabulary.
- **activation_function** / **layer_norm_eps** / **scale_embedding**: block-level knobs, set from the variant config.
- **name** (`str`, *optional*, defaults to `"WhisperConditionalGenerate"`): model name.

**Call** `model({"input_features": ..., "decoder_input_ids": ...})` for a teacher-forced
forward pass. **Returns** a `dict` with **logits** `(B, T, vocab_size)` and
**encoder_hidden_states** `(B, 1500, hidden_dim)`. For transcription use `generate`
instead, which runs the whole loop.

**generate**

```python
model.generate(
    audio,
    processor,
    language="en",
    task="transcribe",
    no_timestamps=True,
    max_new_tokens=224,
    sampling_rate=16000,
    return_ids=False,
    suppress_tokens=None,
    begin_suppress_tokens=None,
)
```

- **audio**: a 1-D float32 waveform in `[-1, 1]`, or a list of them for a batch.
- **processor** (`WhisperProcessor`): supplies the feature extractor and tokenizer.
- **language** (`str`, *optional*, defaults to `"en"`): forced language token. `None` lets the model detect it.
- **task** (`str`, *optional*, defaults to `"transcribe"`): `"transcribe"` keeps the source language, `"translate"` renders English.
- **no_timestamps** (`bool`, *optional*, defaults to `True`): emit plain text rather than timestamp tokens.
- **max_new_tokens** (`int`, *optional*, defaults to `224`): decode budget, half the 448-token context.
- **sampling_rate** (`int`, *optional*, defaults to `16000`): sample rate of `audio`; resample first if yours differs.
- **return_ids** (`bool`, *optional*, defaults to `False`): return token ids instead of strings.
- **suppress_tokens** / **begin_suppress_tokens** (`list`, *optional*): token ids to ban, overriding the defaults.

**Returns** a list of strings, one per clip (or a list of id lists when `return_ids=True`).

### WhisperModel

```python
WhisperModel(hidden_dim=384, encoder_num_layers=4, decoder_num_layers=4, ...,
             name="WhisperModel")
```

The encoder-decoder without the LM head, for features or a custom head. Same arguments as
`WhisperConditionalGenerate`.

### WhisperAudioClassify

```python
WhisperAudioClassify(
    hidden_dim=384,
    encoder_num_layers=4,
    encoder_attention_heads=6,
    encoder_ffn_dim=1536,
    num_mel_bins=80,
    max_source_positions=1500,
    num_classes=2,
    classifier_proj_size=256,
    use_weighted_layer_sum=False,
    activation_function="gelu",
    layer_norm_eps=1e-05,
    name="WhisperAudioClassify",
)
```

The encoder plus a pooling classification head, for tasks like language ID.

- **num_classes** (`int`, *optional*, defaults to `2`): head width.
- **classifier_proj_size** (`int`, *optional*, defaults to `256`): projection before the head.
- **use_weighted_layer_sum** (`bool`, *optional*, defaults to `False`): pool a learned weighting of every encoder layer instead of the last.

**Call** `model(input_features, training=False)`. **Returns** class logits
`(B, num_classes)`.

## Preprocessing

### WhisperFeatureExtractor

```python
WhisperFeatureExtractor(
    sampling_rate=16000, n_fft=400, hop_length=160, n_mels=80, chunk_length=30
)
```

Pads or trims the waveform to `chunk_length` seconds and computes a log-mel spectrogram.

**Parameters**

- **sampling_rate** (`int`, *optional*, defaults to `16000`): rate the model was trained at. Resample your audio to match; this does not resample for you.
- **n_fft** (`int`, *optional*, defaults to `400`): FFT window, 25 ms at 16 kHz.
- **hop_length** (`int`, *optional*, defaults to `160`): hop, 10 ms at 16 kHz.
- **n_mels** (`int`, *optional*, defaults to `80`): mel channels. `large_v3` uses `128`.
- **chunk_length** (`int`, *optional*, defaults to `30`): seconds every clip is padded or trimmed to.

`feat(raw_audio)` **returns** `(B, n_mels, 3000)`.

### WhisperProcessor

```python
WhisperProcessor(
    variant="whisper_tiny",
    n_mels=80,
    sampling_rate=16000,
    n_fft=400,
    hop_length=160,
    chunk_length=30,
    tokenizer_file=None,
    bos_token_id=50257,
    eos_token_id=50257,
    pad_token_id=50257,
    tokenizer=None,
    feature_extractor=None,
)
```

Bundles the feature extractor and tokenizer.

**Call** `processor(audio=..., text=...)`. **Returns** `input_features` and, when text is
given, `labels`.

> **Prefer `WhisperProcessor.from_weights(variant)`.** It picks up the tokenizer and mel
> configuration that match the checkpoint, which is what `generate` expects; the bare
> constructor defaults to `whisper_tiny`'s 80-bin setup and will mismatch `large_v3`.

## Model Variants

| Variant id | Params | Notes |
|---|---:|---|
| `whisper_tiny` | 39 M | multilingual |
| `whisper_base` | 74 M | multilingual |
| `whisper_small` | 244 M | multilingual |
| `whisper_medium` | 769 M | multilingual |
| `whisper_large` | 1.55 B | multilingual |
| `whisper_large_v2` | 1.55 B | multilingual |
| `whisper_large_v3` | 1.55 B | 128 mel bins |
| `whisper_large_v3_turbo` | 809 M | 4 decoder layers |

`large_v3` uses 128 mel bins rather than 80; `from_weights` configures the matching
feature extractor for you. `large_v3_turbo` keeps the v3 encoder but prunes the decoder to
4 layers, trading a little accuracy for much faster decoding.

## Basic Usage: Transcription

The sample below is a LibriSpeech clip, 12.48 s of 16 kHz mono, kept in the repo at
`assets/speech_festive_season.wav`:

<audio controls src="../assets/speech_festive_season.wav"></audio>

Its reference transcript is *"HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH
CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR
MOST READILY TO THE MIND"*.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import soundfile as sf
from zeromodels.models.whisper import WhisperProcessor, WhisperConditionalGenerate

model = WhisperConditionalGenerate.from_weights("zeromodels/whisper_base")
processor = WhisperProcessor.from_weights("zeromodels/whisper_base")

audio, sr = sf.read("assets/speech_festive_season.wav", dtype="float32")  # 16 kHz mono
text = model.generate(audio, processor, language="en", task="transcribe")
print(repr(text[0]))
```

```
' He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.'
```

Whisper is trained on cased, punctuated text, so it capitalizes "Christmas" and adds the
commas and full stop the all-caps LibriSpeech reference lacks. The leading space is
Whisper's own convention. It does slip once, hearing "similarly is drawn" for the
reference's "similes drawn", which is the kind of rare-word error a bigger variant tends to
fix.

### Batching

Pass a list of waveforms. Every clip is padded to the same 30 s window, so they need not
be the same length:

```python
clips = [audio, audio[: 3 * sr]]
for line in model.generate(clips, processor, language="en"):
    print(repr(line))
```

### Translating instead of transcribing

```python
text = model.generate(audio, processor, language="en", task="translate")
```

`task="translate"` always renders **English** output, whatever `language` says the input
is. Pass `language=None` to let the model detect the spoken language itself.

## Longer Audio

The encoder is fixed at a 30-second window, so anything longer must be split. Chunk the
waveform yourself and join the pieces:

```python
window = 30 * sr
chunks = [audio[i : i + window] for i in range(0, len(audio), window)]
text = " ".join(model.generate(chunks, processor, language="en"))
```

Cutting on silence rather than a fixed grid avoids clipping words in half.

## Audio Format

**Every model here takes the same thing: a 1-D float32 waveform in `[-1, 1]` at 16 kHz.**

| | What it expects |
|---|---|
| `generate` / processors | A 1-D `float32` array (or a list of them). `sampling_rate` tells it what rate you are handing over; it does not resample. |
| Models | `input_features`, the `(B, n_mels, 3000)` mel tensor the feature extractor produces. |

```python
import librosa
import soundfile as sf

audio, sr = sf.read(
    "assets/speech_festive_season.wav", dtype="float32"
)  # float32 in [-1, 1]
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo to mono
if sr != 16000:
    audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
```

Passing 44.1 kHz audio with `sampling_rate=16000` does not fail, it just transcribes
gibberish, so resample rather than relabel.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"whisper"` loads with the `hf:` prefix,
including the original OpenAI checkpoints and community fine-tunes.

```python
from zeromodels.models.whisper import WhisperProcessor, WhisperConditionalGenerate

model = WhisperConditionalGenerate.from_weights("hf:openai/whisper-small")
processor = WhisperProcessor.from_weights("hf:openai/whisper-small")

model = WhisperConditionalGenerate.from_weights("hf:<user>/whisper-small-finetuned")

# Architecture only, randomly initialized
model = WhisperConditionalGenerate.from_weights(
    "zeromodels/whisper_tiny", load_weights=False
)
```

Load the processor from the same source as the model: a fine-tune may ship a different
tokenizer or mel configuration.

See also [Moonshine](moonshine.md), which drops the fixed 30-second window, and
[Speech2Text](speech2text.md).
