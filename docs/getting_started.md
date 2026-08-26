# Getting Started

Pretrained transformer models built entirely with **Keras 3**, so the same code runs on
JAX, PyTorch, and TensorFlow. Every model here is a pure-Keras port with weights converted
from the original checkpoints, no `transformers` or `torch` runtime dependency on the model
path.

```shell
pip install -U zeromodels
```

Every model follows the same two-call shape: build it with `from_weights`, and feed it
whatever its processor produces.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.detr import DETRDetect, DETRImageProcessor

model = DETRDetect.from_weights("zeromodels/detr-resnet-50")
processor = DETRImageProcessor.from_weights("zeromodels/detr-resnet-50")

image = Image.open("photo.jpg").convert("RGB")
output = model(processor(image)["pixel_values"], training=False)

results = processor.post_process_object_detection(
    output, threshold=0.9, target_sizes=[(image.height, image.width)]
)[0]
```

## Where to start

<div class="grid cards" markdown>

- **Vision**

    Detection, segmentation, depth, and self-supervised backbones.

    [DETR](detr.md) ┬╖ [SegFormer](segformer.md) ┬╖ [SAM3](sam3.md) ┬╖
    [Depth Anything V2](depth_anything_v2.md) ┬╖ [DINOv3](dinov3.md)

- **Text**

    Encoders and decoder LLMs, dense and mixture-of-experts.

    [BERT](bert.md) ┬╖ [Llama](llama.md) ┬╖ [Qwen3](qwen3.md) ┬╖
    [Gemma 4](gemma4.md) ┬╖ [DeepSeek-V3](deepseek_v3.md)

- **Multimodal**

    Vision-language generation and grounding.

    [Qwen3-VL](qwen3_vl.md) ┬╖ [InternVL](internvl.md) ┬╖
    [Kimi K2.5](kimi_k25.md) ┬╖ [LocateAnything](locateanything.md)

- **Speech**

    Transcription and speech-aware language models.

    [Whisper](whisper.md) ┬╖ [Moonshine](moonshine.md) ┬╖
    [Granite Speech](granite_speech.md)

</div>

## How the pages are organized

Each model page follows the same structure, so you can skim any of them the same way:

| Section | What it holds |
|---|---|
| **API** | Every class, its constructor signature, and what `call` returns. |
| **Preprocessing** | The processor or feature extractor, and its post-processing helpers. |
| **Model Variants** | The `from_weights` ids, with sizes and what each was trained on. |
| **Basic Usage** | A runnable example with its real, measured output. |
| **Data Format** | Layouts, `channels_last` vs `channels_first`, audio rates. |
| **Loading Fine-tuned Weights** | Hub Keras (`zeromodels/...`) and the `hf:` prefix for any compatible Hub repo. |

The outputs printed in those examples are **measured, not illustrative**: they come from
actually running the snippet on the image or audio clip shown beside it.

The pieces every page shares are documented once: [Main Classes](main_classes.md) for the
model, processor, and generation bases, and [Utilities](utils.md) for image and video
loading, plotting, and the class-name lists.

## Loading weights

Three sources, one call:

```python
# Preconverted Keras weights on the zeromodels Hub org (kf_config.json)
model = SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")

# Bare variant: on-the-fly conversion from an upstream Hub checkpoint (LLMs / VLMs)
model = Qwen3TextGenerate.from_weights("qwen3-8b")

# Any Hugging Face repo with a matching model_type (convert on the fly)
model = SegFormerSemanticSegment.from_weights(
    "hf:nvidia/segformer-b0-finetuned-ade-512-512"
)

# Architecture only, randomly initialized
model = SegFormerSemanticSegment.from_weights(
    "zeromodels/segformer_b0_ade_512", load_weights=False
)
```

Large checkpoints load in lower precision or weight-only quantized; see
[Loading Weights](loading_weights.md) and [Quantization](quantization.md).

```python
model = Qwen3TextGenerate.from_weights(
    "qwen3-8b", load_dtype="bfloat16", quantization="int8"
)
```

## Backends

Set `KERAS_BACKEND` before importing Keras. Models read
`keras.config.image_data_format()` when they are **constructed**, so set that first too if
you want `channels_first`.

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "torch" / "tensorflow"

import keras

keras.config.set_image_data_format("channels_first")
```

Source and issues: [github.com/IMvision12/ZeroModels](https://github.com/IMvision12/ZeroModels).
