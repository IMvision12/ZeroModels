---
template: home.html
---

<!-- Landing page. The shell (full width, no sidebars) comes from
     website/overrides/home.html, the styling from overrides/stylesheets/home.css.

     Two things to keep in mind when editing:

     * Raw HTML links are not rewritten by MkDocs, so internal hrefs are written
       as directory URLs ("detr/", not "detr.md"). Inside a Markdown-enabled
       block the normal "[DETR](detr.md)" form works and is preferred.
     * This page is served from the site root, so figures are "assets/..." with
       no "../" prefix, unlike the model pages. -->

<div class="kf-hero">
  <a class="kf-pill" href="sam3/">
    <span class="kf-pill__tag">New</span>
    SAM 3, Gemma 4 and GLM-5 have landed
  </a>
  <h1>Run <em>any</em> model,<br>on any backend</h1>
  <p class="kf-hero__lede">
    100+ model families ported to pure Keras 3, with weights converted from the
    original checkpoints. The same code runs on JAX, PyTorch and TensorFlow, and
    nothing from <code>transformers</code> or <code>torch</code> is needed at
    run time.
  </p>
  <div class="kf-actions">
    <a class="kf-btn kf-btn--primary" href="getting_started/">Get started</a>
    <a class="kf-btn kf-btn--ghost" href="https://github.com/IMvision12/ZeroModels">GitHub</a>
  </div>
</div>

<div class="kf-cards">
  <a class="kf-card" href="detr/">
    <span class="kf-card__media kf-card__media--detr">
      <img src="assets/detr_output.jpg" alt="DETR detections on a living room scene">
    </span>
    <span class="kf-card__body">
      <span class="kf-card__kicker">Object detection</span>
      <span class="kf-card__title">DETR</span>
      <span class="kf-card__meta">End-to-end, no anchors or NMS</span>
    </span>
  </a>
  <a class="kf-card" href="segformer/">
    <span class="kf-card__media kf-card__media--segformer">
      <img src="assets/segformer_seg_output.jpg" alt="SegFormer segmentation overlaid on a street corner at dusk">
    </span>
    <span class="kf-card__body">
      <span class="kf-card__kicker">Segmentation</span>
      <span class="kf-card__title">SegFormer</span>
      <span class="kf-card__meta">13 variants, ADE20K and Cityscapes</span>
    </span>
  </a>
  <a class="kf-card" href="depth_anything_v2/">
    <span class="kf-card__media kf-card__media--depth">
      <img src="assets/depth_anything_v2_single_output.jpg" alt="Depth Anything V2 depth map of a tennis match">
    </span>
    <span class="kf-card__body">
      <span class="kf-card__kicker">Depth estimation</span>
      <span class="kf-card__title">Depth Anything V2</span>
      <span class="kf-card__meta">Relative and metric depth</span>
    </span>
  </a>
  <a class="kf-card" href="sam3/">
    <span class="kf-card__media kf-card__media--sam3">
      <img src="assets/sam3_text_output.jpg" alt="SAM 3 masks on two elephants, from the text prompt 'elephant'">
    </span>
    <span class="kf-card__body">
      <span class="kf-card__kicker">Promptable masks</span>
      <span class="kf-card__title">SAM 3</span>
      <span class="kf-card__meta">Points, boxes and concepts</span>
    </span>
  </a>
  <a class="kf-card" href="owlv2/">
    <!-- Both OWLv2 panels are portrait, so cropping one to 3:2 loses a third of
         its height and cuts the subject. This nests a frame the exact shape of
         the panel, so the detection shows complete. -->
    <span class="kf-card__media kf-card__media--fit">
      <span class="kf-card__panel kf-card__panel--owlv2">
        <img src="assets/owlv2_batch_output.jpg" alt="OWLv2 detecting a real bear from the prompt 'a photo of a real bear'">
      </span>
    </span>
    <span class="kf-card__body">
      <span class="kf-card__kicker">Open-vocabulary detection</span>
      <span class="kf-card__title">OWLv2</span>
      <span class="kf-card__meta">Text prompts, no fixed class list</span>
    </span>
  </a>
  <!-- The only card that is not one big link: it holds an audio player, and a
       control nested inside a link would navigate away when you press play. The
       link wraps just the text block instead. The clip is the one whisper.md
       transcribes, and the text below it is that measured transcript. -->
  <div class="kf-card">
    <span class="kf-card__media kf-card__media--audio">
      <span class="kf-transcript">
        <span class="kf-transcript__label">whisper_base · 12.48 s clip</span>
        <span class="kf-transcript__text">He tells us that at this festive season of the year, with Christmas and roast beef looming before us…</span>
      </span>
      <audio class="kf-card__audio" controls preload="none" src="assets/speech_festive_season.wav"></audio>
    </span>
    <a class="kf-card__body" href="whisper/">
      <span class="kf-card__kicker">Speech recognition</span>
      <span class="kf-card__title">Whisper</span>
      <span class="kf-card__meta">Transcribe or translate</span>
    </a>
  </div>
</div>

<div class="kf-section" markdown="1">

## Two calls to a prediction

Build the model with `from_weights`, then feed it whatever its processor
produces. Every model in the library follows this shape, so moving between a
detector, a depth estimator and an LLM costs you nothing.

<div class="kf-start" markdown="1">

<div markdown="1">

```shell
pip install -U zeromodels
```

Weights come from the [`zeromodels`](https://huggingface.co/zeromodels) org
on the Hub, and the same identifier builds both the model and its processor, so
the resolution and normalization always match the checkpoint.

</div>

<div markdown="1">

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

</div>

</div>

</div>

<!-- markdown="1" has to be repeated down the whole chain: without it on this
     outer div, everything nested inside stays a raw HTML block and the headings,
     links and figures below never get parsed. -->
<div class="kf-panel" markdown="1">

<div class="kf-feature" markdown="1">

<div class="kf-feature__text" markdown="1">

### One call, three sources

`from_weights` dispatches on what you hand it: a preconverted Keras repo on the
Hub, a bare variant name that converts an upstream checkpoint on the fly, or any
compatible Hugging Face repo behind the `hf:` prefix. Architecture details,
including the class count of a fine-tune, are read from the repo config.

[Loading weights](loading_weights.md) · [Main classes](main_classes.md)

</div>

<div markdown="1">

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate
from zeromodels.models.segformer import SegFormerSemanticSegment

# Preconverted Keras weights (zm_config.json)
SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")

# Bare variant: converted from upstream on the fly
Qwen3TextGenerate.from_weights("qwen3-8b")

# Any Hub repo with a matching model_type
SegFormerSemanticSegment.from_weights("hf:nvidia/segformer-b0-finetuned-ade-512-512")

# Architecture only, randomly initialized
SegFormerSemanticSegment.from_weights(
    "zeromodels/segformer_b0_ade_512", load_weights=False
)
```

</div>

</div>

<div class="kf-feature" markdown="1">

<div class="kf-feature__text" markdown="1">

### Measured outputs, not illustrative ones

Every figure and every printed result on a model page comes from actually
running the snippet beside it on the image or audio clip shown. Nothing is
hand-written to look plausible, so what you read is what you get when you run
it yourself.

[Browse the model pages](main_classes.md)

</div>

<div markdown="1">

![SegFormer B5 on an open-plan kitchen and a herd in a field](assets/segformer_seg_batch_output.jpg)

</div>

</div>

<div class="kf-feature" markdown="1">

<div class="kf-feature__text" markdown="1">

### Any backend, either data format

Set `KERAS_BACKEND` before importing Keras and the rest is unchanged. Models
read `keras.config.image_data_format()` when they are **constructed**, so set
that first too if you want `channels_first`; processors take a per-instance
`data_format` argument.

[Configuration](configuration.md) · [Utilities](utils.md)

</div>

<div markdown="1">

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "torch" / "tensorflow"

import keras

keras.config.set_image_data_format("channels_first")
```

</div>

</div>

<div class="kf-feature" markdown="1">

<div class="kf-feature__text" markdown="1">

### Large checkpoints, as they ship

- GPT-OSS 120B loads at bfloat16 with its MoE experts left packed in MXFP4 and
  dequantized on the fly, so it stays near 66 GB instead of the ~130 GB an fp32
  expansion would cost.
- Weight-only int8, int4, fp8 and mxfp4 are arguments to the same `from_weights`
  call, on any model.

[Quantization](quantization.md) · [int8](quantization_int8.md) ·
[int4](quantization_int4.md) · [fp8](quantization_fp8.md) ·
[mxfp4](quantization_mxfp4.md)

</div>

<div markdown="1">

```python
from zeromodels.models.gpt_oss import GptOssTextGenerate

# Experts stay packed in MXFP4, dequantized in the expert layer's call
model = GptOssTextGenerate.from_weights("zeromodels/gpt-oss-120b")

# Quantize weight-only on the way in, for a smaller footprint again
model = GptOssTextGenerate.from_weights("zeromodels/gpt-oss-120b", quantization="int8")
```

</div>

</div>

</div>

<div class="kf-section" markdown="1">

## Where to start

<div class="kf-grid" markdown="1">

<div class="kf-tile" markdown="1">

### Vision

Detection, segmentation, depth, and self-supervised backbones.

[DETR](detr.md) · [SegFormer](segformer.md) · [SAM 3](sam3.md) ·
[Depth Anything V2](depth_anything_v2.md) · [DINOv3](dinov3.md)

</div>

<div class="kf-tile" markdown="1">

### Text

Encoders and decoder LLMs, dense and mixture-of-experts.

[BERT](bert.md) · [Llama](llama.md) · [Qwen3](qwen3.md) ·
[Gemma 4](gemma4.md) · [DeepSeek-V3](deepseek_v3.md)

</div>

<div class="kf-tile" markdown="1">

### Multimodal

Vision-language generation and grounding.

[Qwen3-VL](qwen3_vl.md) · [InternVL](internvl.md) ·
[Kimi K2.5](kimi_k25.md) · [LocateAnything](locateanything.md)

</div>

<div class="kf-tile" markdown="1">

### Speech

Transcription and speech-aware language models.

[Whisper](whisper.md) · [Moonshine](moonshine.md) ·
[Granite Speech](granite_speech.md)

</div>

</div>

</div>

<div class="kf-cta">
  <h2>Ready to use ZeroModels?</h2>
  <p>One install, 118 model families, three backends.</p>
  <div class="kf-actions">
    <a class="kf-btn kf-btn--primary" href="getting_started/">Get started</a>
    <a class="kf-btn kf-btn--ghost" href="https://huggingface.co/zeromodels">Weights on the Hub</a>
  </div>
</div>
