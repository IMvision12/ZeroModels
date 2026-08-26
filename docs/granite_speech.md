# Granite Speech

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Granite Speech is a **speech-aware LLM**, not an ASR model with a language model bolted
on. A conformer CTC audio encoder and a BLIP-2 style Q-Former projector turn mel features
into audio embeddings, which are scattered into `<|audio|>` placeholder positions in a
Granite text decoder, exactly the way a vision-language model splices image embeddings
into the token stream.

The practical consequence: **you ask it for what you want in words**. "Transcribe this"
gets a transcript; a different instruction over the same audio gets something else. It
also keeps its text-only abilities, since in text mode it is just the Granite decoder with
the audio LoRA switched off.

**Paper**: [Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities](https://arxiv.org/abs/2505.08699)

## API

### GraniteSpeechConditionalGenerate

```python
GraniteSpeechConditionalGenerate(vocab_size=49160, embed_dim=2048, mlp_dim=8192,
                      num_layers=40, num_heads=32, num_kv_heads=8,
                      norm_eps=1e-05, rope_theta=1e7,
                      embedding_multiplier=12.0, residual_multiplier=0.22,
                      attention_multiplier=0.015625, logits_scaling=8.0,
                      tie_embeddings=True, eos_token_id=0,
                      audio_token_id=49159, downsample_rate=5, window_size=15,
                      has_lora_adapter=True, lora_rank=64, ...,
                      name="GraniteSpeechConditionalGenerate")
```

The audio encoder, projector, LoRA-adapted Granite decoder, and LM head. **This is the
class for audio-plus-text to text.**

**Parameters**

- **vocab_size** / **embed_dim** / **mlp_dim** / **num_layers** / **num_heads** / **num_kv_heads** (`int`, *optional*): the Granite decoder shape. Filled in by `from_weights` from the variant config.
- **rope_theta** (`float`, *optional*, defaults to `1e7`): RoPE base frequency.
- **embedding_multiplier** / **residual_multiplier** / **attention_multiplier** / **logits_scaling** (`float`, *optional*): Granite's scalar multipliers; the head divides by `logits_scaling`.
- **audio_token_id** (`int`, *optional*, defaults to `49159`): the `<|audio|>` placeholder the projector output is scattered into.
- **downsample_rate** / **window_size** (`int`, *optional*, defaults to `5` / `15`): projector windowing, which sets how many embeddings each second of audio becomes.
- **has_lora_adapter** (`bool`, *optional*, defaults to `True`): enable the query/value LoRA that is active in speech mode and off for text.
- **lora_rank** (`int`, *optional*, defaults to `64`): rank of that adapter.
- **tie_embeddings** (`bool`, *optional*, defaults to `True`): reuse the embedding matrix as the LM head.
- **name** (`str`, *optional*, defaults to `"GraniteSpeechConditionalGenerate"`): model name.

**generate**

```python
model.generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    **prefill_inputs,
)
```

The processor produces every tensor it needs, so the call is normally
`model.generate(**inputs, max_new_tokens=...)`. The audio tensors ride along in
`**prefill_inputs`. **Returns** token ids; decode them with `processor.tokenizer.decode`.

### GraniteSpeechModel

```python
GraniteSpeechModel(vocab_size=49160, embed_dim=2048, mlp_dim=8192,
                   num_layers=40, ..., name="GraniteSpeechModel")
```

The same stack without the LM head, for hidden states. Same arguments as
`GraniteSpeechConditionalGenerate`.

**Call** `model(inputs)` with the processor's dict. **Returns** the decoder hidden states
`(B, T, embed_dim)`.

### GraniteSpeechTextModel

The Granite decoder alone, for text-only use, with the audio branch and LoRA left out.

## Preprocessing

### GraniteSpeechFeatureExtractor

```python
GraniteSpeechFeatureExtractor(
    sampling_rate=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=80,
    projector_window_size=15,
    projector_downsample_rate=5,
)
```

Computes 80-bin mel features and reports how many audio embeddings the projector will
produce, which is what decides how far each `<|audio|>` placeholder is expanded.

**Parameters**

- **sampling_rate** (`int`, *optional*, defaults to `16000`): rate the model was trained at. Resample your audio to match; this does not resample for you.
- **n_fft** / **win_length** / **hop_length** (`int`, *optional*, defaults to `512` / `400` / `160`): STFT window and hop, 25 ms every 10 ms at 16 kHz.
- **n_mels** (`int`, *optional*, defaults to `80`): mel channels.
- **projector_window_size** / **projector_downsample_rate** (`int`, *optional*, defaults to `15` / `5`): must match the model's, since they decide the audio-embedding count.

`feat(raw_speech)` **returns** `input_features` and `input_features_mask`, plus the
`audio_embed_sizes` the processor uses to expand placeholders.

### GraniteSpeechProcessor

```python
processor(
    text=None,
    audio=None,
    conversation=None,
    messages=None,
    sampling_rate=16000,
    add_generation_prompt=True,
)
```

Renders the chat template and expands the audio placeholders. **Returns** `input_ids`,
`attention_mask`, `input_features`, and `input_features_mask`.

> **Audio goes in the `audio=` argument, not inside the conversation.** The conversation
> marks *where* the audio belongs with a bare `{"type": "audio"}` item; the waveform itself
> is passed separately. Put the array in the content dict and you get a text-only prompt
> back, with no `input_features` and a useless answer.

## Model Variants

| Variant id | Decoder | Params |
|---|---|---:|
| `granite_speech_3_3_2b` | Granite 3.3 2B | ~2 B |
| `granite_speech_3_3_8b` | Granite 3.3 8B | ~8 B |
| `granite_speech_4_1_2b` | Granite 4.1 2B | ~2 B |
| `granite_4_0_1b_speech` | Granite 4.0 1B | ~1 B |

These are decoder-sized: load in bf16 (`load_dtype="bfloat16"`) unless you have room for
fp32. See also [Granite Speech Plus](granite_speech_plus.md).

## Basic Usage: Transcription

The sample below is a LibriSpeech clip, 9.01 s of 16 kHz mono, kept in the repo at
`assets/speech_luminous_criticisms.wav`:

<audio controls src="../assets/speech_luminous_criticisms.wav"></audio>

Its reference transcript is *"IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS
THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION"*.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import keras
import numpy as np
import soundfile as sf
from zeromodels.models.granite_speech import (
    GraniteSpeechConditionalGenerate,
    GraniteSpeechProcessor,
)

model = GraniteSpeechConditionalGenerate.from_weights(
    "zeromodels/granite_speech_3_3_2b", load_dtype="bfloat16"
)
processor = GraniteSpeechProcessor.from_weights("zeromodels/granite_speech_3_3_2b")

audio, sr = sf.read(
    "assets/speech_luminous_criticisms.wav", dtype="float32"
)  # 16 kHz mono

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {
                "type": "text",
                "text": "can you transcribe the speech into a written format?",
            },
        ],
    }
]

inputs = processor(conversation=conversation, audio=audio, sampling_rate=sr)
out = model.generate(**inputs, max_new_tokens=64)

ids = np.asarray(keras.ops.convert_to_numpy(out))[0].tolist()
print(repr(processor.tokenizer.decode(ids)))
```

```
'it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression'
```

Word-for-word the reference, across nine seconds. Note the instruction is ordinary English,
not a flag: the model is being *asked* to transcribe.

### Asking for something other than a transcript

The same audio with a different instruction gives a different answer, which is what
separates a speech LLM from an ASR model:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {
                "type": "text",
                "text": "What is the speaker talking about? Answer in one sentence.",
            },
        ],
    }
]
inputs = processor(conversation=conversation, audio=audio, sampling_rate=sr)
out = model.generate(**inputs, max_new_tokens=64)
```

### Text only

Drop the audio and the audio branch stays off, leaving the plain Granite decoder:

```python
inputs = processor(conversation=[{"role": "user", "content": "Who wrote Dune?"}])
out = model.generate(**inputs, max_new_tokens=32)
```

## Audio Format

**A 1-D float32 waveform in `[-1, 1]` at 16 kHz.**

| | What it expects |
|---|---|
| Processor | The waveform in the `audio=` argument, plus a `{"type": "audio"}` marker in the conversation. `sampling_rate` tells it what rate you are handing over; it does not resample. |
| Models | `input_features` / `input_features_mask` from the extractor, alongside `input_ids`. |

```python
import librosa
import soundfile as sf

audio, sr = sf.read("assets/speech_luminous_criticisms.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo to mono
if sr != 16000:
    audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
```

The conformer encoder's cost grows with duration, so segment long recordings rather than
feeding an entire meeting in one call.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"granite_speech"` loads with the `hf:` prefix.

```python
from zeromodels.models.granite_speech import (
    GraniteSpeechConditionalGenerate,
    GraniteSpeechProcessor,
)

model = GraniteSpeechConditionalGenerate.from_weights(
    "hf:ibm-granite/granite-speech-3.3-2b"
)
processor = GraniteSpeechProcessor.from_weights("hf:ibm-granite/granite-speech-3.3-2b")

# Architecture only, randomly initialized
model = GraniteSpeechConditionalGenerate.from_weights(
    "zeromodels/granite_speech_3_3_2b", load_weights=False
)
```

See also [Granite Speech Plus](granite_speech_plus.md), and [Whisper](whisper.md) if you
want transcription without a decoder-sized language model.
