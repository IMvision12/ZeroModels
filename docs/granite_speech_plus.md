# Granite Speech Plus

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Granite Speech Plus is the **Granite 4.0-based** successor to
[Granite Speech](granite_speech.md). The architecture is the same idea, a conformer CTC
encoder and a BLIP-2 Q-Former projector feeding audio embeddings into `<|audio|>` slots of
a Granite decoder, so this port reuses the `granite_speech` implementation outright and
only the config and tokenizer differ.

Read [Granite Speech](granite_speech.md) for how the pieces fit together and how to call
the processor. This page covers the deltas and gives a runnable example.

**Model card**: [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
**Architecture paper**: [Granite-speech](https://arxiv.org/abs/2505.08699)

## What Changed

| | `granite_speech_3_3_2b` | `granite_speech_4_1_2b_plus` |
|---|---|---|
| LLM base | Granite 3.3 | **Granite 4.0** |
| Vocabulary | 49,160 | **100,353** |
| MLP dim | 8192 | **4096** |
| Heads (q / kv) | 32 / 8 | **16 / 4** |
| `rope_theta` | 1e7 | **10000** |
| `attention_multiplier` | 1/64 | **1/128** |
| `audio_token_id` | 49159 | **100352** |
| LoRA adapter | rank 64 | **none**, weights fully merged |
| `cat_hidden_layers` | `None` | **`[3]`** |
| Encoder `output_dim` | 256 | **348** |

`hidden_size` (2048) and `num_layers` (40) are unchanged.

**`cat_hidden_layers = [3]`** is the one structural change: the CTC encoder concatenates
its layer-3 intermediate output with the final output before the projector, so the
projector's `encoder_hidden_size` doubles to `1024 x 2 = 2048`. The model derives that for
you.

Dropping the LoRA adapter is the other practical difference. Granite Speech switches a
query/value LoRA on for audio and off for text; Plus has those weights merged, so there is
no adapter to toggle.

## API

The classes mirror Granite Speech exactly, under `zeromodels.models.granite_speech_plus`.

### GraniteSpeechPlusConditionalGenerate

```python
GraniteSpeechPlusConditionalGenerate(vocab_size=100353, embed_dim=2048, mlp_dim=4096,
                          num_layers=40, num_heads=16, num_kv_heads=4,
                          rope_theta=10000.0, attention_multiplier=1 / 128,
                          audio_token_id=100352, has_lora_adapter=False,
                          cat_hidden_layers=[3], ...,
                          name="GraniteSpeechPlusConditionalGenerate")
```

The audio encoder, projector, Granite 4.0 decoder, and LM head. **This is the class for
audio-plus-text to text.** The arguments are the same set as
[`GraniteSpeechConditionalGenerate`](granite_speech.md#granitespeechconditionalgenerate), with the defaults in
the table above; `from_weights` fills them in.

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

Call it as `model.generate(**inputs, max_new_tokens=...)` with the processor's output; the
audio tensors ride along in `**prefill_inputs`. **Returns** token ids, which you decode
with `processor.tokenizer.decode`.

### Other classes

| Class | What it is |
|---|---|
| `GraniteSpeechPlusModel` | the same stack without the LM head |
| `GraniteSpeechPlusProcessor` | chat template, audio placeholder expansion, tokenization |
| `GraniteSpeechPlusTokenizer` | the Granite 4.0 vocabulary |
| `GraniteSpeechFeatureExtractor` | reused unchanged from `granite_speech` |

## Preprocessing

`GraniteSpeechPlusProcessor` takes the same call as Granite Speech's:

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

**Returns** `input_ids`, `attention_mask`, `input_features`, and `input_features_mask`.
The feature extractor is `GraniteSpeechFeatureExtractor`, unchanged; see
[Granite Speech](granite_speech.md#preprocessing) for its parameters.

## Model Variants

| Variant id | Decoder | Params |
|---|---|---:|
| `granite_speech_4_1_2b_plus` | Granite 4.0 2B | ~2 B |

One variant. Load it in bf16 unless you have room for fp32.

## Basic Usage: Transcription

The sample below is a LibriSpeech clip, 9.90 s of 16 kHz mono, kept in the repo at
`assets/speech_leighton.wav`:

<audio controls src="../assets/speech_leighton.wav"></audio>

Its reference transcript is *"HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS
REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA"*.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

import keras
import numpy as np
import soundfile as sf
from zeromodels.models.granite_speech_plus import (
    GraniteSpeechPlusConditionalGenerate,
    GraniteSpeechPlusProcessor,
)

model = GraniteSpeechPlusConditionalGenerate.from_weights(
    "zeromodels/granite_speech_4_1_2b_plus", load_dtype="bfloat16"
)
processor = GraniteSpeechPlusProcessor.from_weights(
    "zeromodels/granite_speech_4_1_2b_plus"
)

audio, sr = sf.read("assets/speech_leighton.wav", dtype="float32")  # 16 kHz mono

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
"he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca"
```

Word-for-word the reference, including the proper nouns "Frederick Leighton" and "Ithaca"
and the possessive apostrophe, across nearly ten seconds of audio.

> **Audio goes in the `audio=` argument, not inside the conversation**, same as Granite
> Speech. The `{"type": "audio"}` item only marks the position.

## Audio Format

**A 1-D float32 waveform in `[-1, 1]` at 16 kHz.**

| | What it expects |
|---|---|
| Processor | The waveform in the `audio=` argument, plus a `{"type": "audio"}` marker in the conversation. `sampling_rate` tells it what rate you are handing over; it does not resample. |
| Models | `input_features` / `input_features_mask` from the extractor, alongside `input_ids`. |

```python
import librosa
import soundfile as sf

audio, sr = sf.read("assets/speech_leighton.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # stereo to mono
if sr != 16000:
    audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
```

## Loading Fine-tuned and Community Weights

```python
from zeromodels.models.granite_speech_plus import (
    GraniteSpeechPlusConditionalGenerate,
    GraniteSpeechPlusProcessor,
)

model = GraniteSpeechPlusConditionalGenerate.from_weights(
    "hf:ibm-granite/granite-speech-4.1-2b-plus"
)
processor = GraniteSpeechPlusProcessor.from_weights(
    "hf:ibm-granite/granite-speech-4.1-2b-plus"
)

# Architecture only, randomly initialized
model = GraniteSpeechPlusConditionalGenerate.from_weights(
    "zeromodels/granite_speech_4_1_2b_plus", load_weights=False
)
```

See also [Granite Speech](granite_speech.md) for the full architecture walkthrough.
