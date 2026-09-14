# Main Classes

Every model in ZeroModels is assembled from the same small set of base classes in
`zeromodels.base`. You rarely instantiate them directly, but knowing what they provide
explains why every model page looks alike: the same `from_weights`, the same processor
call, the same `generate`.

```python
from zeromodels.base import (
    BaseConfig,
    BaseModel,
    BaseGeneration,
    BaseSeq2SeqGeneration,
    BaseDiffusion,
    BaseScheduler,
    BaseTokenizer,
    BaseImageProcessor,
    BaseAudioFeatureExtractor,
    BaseProcessor,
    BaseQuantizer,
    fused_attention,
)
```

## Models

### BaseModel

```python
class BaseModel(keras.Model)
```

The single base every model builds on. A model assembles itself as a **Keras functional
graph** with `super().__init__(inputs=..., outputs=...)`. Vision models (CLIP, ViT, the
detectors, segmenters, depth estimators) trace a fixed input shape at construction, which
is why they take an `image_size` and why changing it means rebuilding. Text and multimodal
models declare their sequence inputs with an undefined length, so a language model still
takes any sequence length; the imperative KV-cache decode that autoregressive generation
needs lives on the task side, in a `BaseGeneration` / `BaseSeq2SeqGeneration` / `BaseDiffusion` mixin layered
over this backbone.

It carries the loading interface below.

## Configuration

### BaseConfig

```python
class BaseConfig
```

Every model carries a typed config, a `BaseConfig` subclass whose annotated fields are the
architecture hyperparameters. Model constructors stay flat while the config serializes
nested (the `text_config` / `vision_config` blocks you see in `zm_config.json`), and
BaseConfig bridges the two: `constructor_kwargs()` flattens a config for the model, and
`to_dict()` / `from_dict()` handle serialization. You rarely build one by hand, since
`from_weights` reconstructs it from the repo, but the [Configuration](configuration.md)
page covers the field system, the composite `sub_configs` layout, and the sub-config
classes each model exports.

## Loading Weights

Every model and preprocessor inherits these classmethods. What each source actually does
(Hub Keras repos, bare-variant on-the-fly conversion, `hf:` conversion, and caching) is
covered in [Loading Weights](loading_weights.md).

### from_weights

```python
Model.from_weights(
    identifier,
    load_weights=True,
    skip_mismatch=False,
    attn_implementation=None,
    quantization=None,
    load_dtype=None,
    cache_converted=False,
    **kwargs,
)
```

The one entry point you normally use. It dispatches on `identifier`:

- `"org/repo"` (for example `"zeromodels/segformer_b0_ade_512"`) → Hub Keras via
  `zm_config.json`
- a bare variant (for example `"qwen3-4b"`) → on-the-fly conversion from an upstream
  `hf_id`
- `"hf:org/repo"` → convert any compatible Hub checkpoint

**Parameters**

- **identifier** (`str`): a Hub Keras repo (`"zeromodels/segformer_b0_ade_512"`), a bare
  LLM/VLM variant (`"qwen3-4b"`), or an `hf:`-prefixed Hub repo
  (`"hf:nvidia/segformer-b0-finetuned-ade-512-512"`).
- **load_weights** (`bool`, *optional*, defaults to `True`): set `False` to build the architecture with random initialization.
- **skip_mismatch** (`bool`, *optional*, defaults to `False`): skip weights whose shapes disagree instead of raising, for partially compatible fine-tunes.
- **attn_implementation** (`str`, *optional*): attention kernel to use, see [`fused_attention`](#fused_attention).
- **quantization** (`str`, *optional*): quantize while loading, for example `"int8"`. The model builds at `load_dtype` and quantizes after. See [Quantization](quantization.md).
- **load_dtype** (`str`, *optional*): cast weights on load, typically `"bfloat16"`.
- **cache_converted** (`bool`, *optional*, defaults to `False`): keep the converted Keras weights so the next conversion load skips work.
- **kwargs**: forwarded to the constructor, so `image_size=448` or `as_backbone=True` go here.

```python
model = SegFormerSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")
model = SegFormerSemanticSegment.from_weights("hf:<user>/my-finetune")
model = Qwen3TextGenerate.from_weights(
    "qwen3-8b", load_dtype="bfloat16", quantization="int8"
)
```

### from_hub_repo, from_variant, and from_hf

```python
Model.from_hub_repo(
    repo_id,
    load_weights=True,
    skip_mismatch=False,
    **kwargs,
)

Model.from_variant(
    variant,
    load_weights=True,
    skip_mismatch=False,
    quantization=None,
    **kwargs,
)

Model.from_hf(
    hf_id,
    load_weights=True,
    variant=None,
    skip_mismatch=False,
    quantization=None,
    **kwargs,
)
```

The three halves `from_weights` dispatches to. Call them directly only when you want to be
explicit about the source. `from_hub_repo` reads `zm_config.json` (and optionally
`zm_preprocessor.json` for processors). `from_hf` reads the repo's `config.json`, so a
fine-tune with a different class count or vocabulary needs no extra arguments.

> **Load the processor from the same source as the model.** A fine-tune can ship a
> different tokenizer, label set, or normalization; mismatching them fails quietly with
> wrong output rather than loudly with an error.

### quantize

```python
model.quantize(mode=None, config=None, filters=None, **kwargs)
```

Quantize an already-built model in place. Passing `quantization=` to `from_weights` is
usually better, since it avoids materializing float weights first. See
[Quantization](quantization.md).

## Generation

### BaseGeneration

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

Backend-agnostic autoregressive decoding for **decoder-only** models, the counterpart to
Hugging Face's `GenerationMixin`. Any extra tensors a multimodal model needs, pixel values
or audio features, ride along in `**prefill_inputs`, which is why a VLM call looks like
`model.generate(**inputs, max_new_tokens=64)`.

**Parameters**

- **input_ids**: the prompt token ids from a tokenizer or processor.
- **attention_mask** (*optional*): padding mask for batched prompts.
- **max_new_tokens** (`int`, *optional*): decode budget.
- **eos_token_id** (`int`, *optional*): stop token, defaulting to the model's own.
- **sampler** (*optional*): a sampler from `zeromodels.samplers` (`GreedySampler`,
  `TopKSampler`, `TopPSampler`); greedy (deterministic `argmax`) if omitted. Stochastic
  samplers draw with an inverse-CDF categorical step.
- **seed** (`int`, *optional*): only affects stochastic samplers. With **no seed** each
  call draws fresh noise, so sampling **varies from call to call** (the usual `do_sample`
  behavior). Pass an explicit `seed` for a **reproducible** run (the same tokens every call
  on a given backend). `keras.random` is backend-specific, so a seeded stochastic run is
  reproducible per backend, not identical across torch / jax / tf; greedy is deterministic
  everywhere.

```python
from zeromodels.samplers import TopKSampler

model.generate(prompt_ids, max_new_tokens=64)  # greedy, deterministic
model.generate(prompt_ids, sampler=TopKSampler(k=50))  # sampled, varies each call
model.generate(prompt_ids, sampler=TopKSampler(k=50), seed=42)  # sampled, reproducible
```

### Samplers

The samplers live in `zeromodels.samplers`:

- **`GreedySampler`** — deterministic `argmax`; ignores the noise. Used when `sampler` is
  omitted.
- **`TopKSampler(k=50, temperature=1.0)`** — keep the `k` highest-logit tokens. `k` is
  clamped to `[1, vocab_size]`, so `k <= 0` falls back to greedy.
- **`TopPSampler(p=0.9, temperature=1.0, min_tokens_to_keep=1)`** — nucleus sampling: the
  smallest set of top tokens whose cumulative probability reaches `p`. At least
  `min_tokens_to_keep` top tokens are always kept, so `p <= 0` falls back to greedy.

`temperature` must be **strictly positive** — both stochastic samplers raise `ValueError`
on `temperature <= 0` (use `GreedySampler` for greedy decoding). Stochastic samplers draw
each token with an inverse-CDF categorical step from the pre-drawn `(steps, batch)` noise.

### BaseSeq2SeqGeneration

```python
model.generate(
    encoder_inputs,
    decoder_input_ids,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
)
model.encode(encoder_inputs)
```

The encoder-decoder flavor, used by [Whisper](whisper.md),
[Speech2Text](speech2text.md), and [Moonshine](moonshine.md). `encode` runs the encoder
once so you can decode repeatedly against the same audio.

Those speech models wrap this in a friendlier `generate(audio, processor, ...)` that owns
the whole pipeline; see their pages.

### BaseDiffusion

```python
model.generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    image=None,
    strength=None,
    denoising_start=None,
    denoising_end=None,
    output_type="image",
    **conditioning,
)
```

The diffusion flavor, used by [Stable Diffusion](stable_diffusion.md),
[Stable Diffusion 2](stable_diffusion_2.md), [Stable Diffusion XL](stable_diffusion_xl.md)
and [Stable Diffusion 3 / 3.5](stable_diffusion_3.md). Where the LM
mixins decode tokens, this one runs a scheduler's denoising loop with classifier-free
guidance over a latent and decodes it to `(batch, H, W, 3)` uint8 images. A model
supplies five hooks (`encode_prompt`, `unconditional_ids`, `predict_noise`,
`decode_latents`, `latent_shape`) and a `scheduler`; the mixin owns the guidance batching,
the initial latent, the loop and the postprocess. `encode_prompt` may return a nested
structure of tensors rather than one (SDXL's context, pooled embedding and size ids),
which the guidance batching and the compiled step carry through unchanged; a model whose
unconditional branch is not an encoded prompt overrides `encode_negative_prompt` (SDXL
zeroes it). The denoiser call is compiled per
backend (`jax.jit`, `tf.function(jit_compile=True)`, eager on Torch) and cached on the
instance, like the LM decode loop.

**Parameters**

- **input_ids**: `(batch, seq)` token ids from the family's tokenizer.
- **negative_input_ids** (*optional*): the tokenized negative prompt, one row per prompt
  or a single row for the batch; the empty prompt when omitted.
- **num_inference_steps** / **guidance_scale** (*optional*): default to the repo's
  `generate_args` (50 steps, 7.5 for Stable Diffusion); `guidance_scale <= 1` turns
  guidance off.
- **seed** (`int`, *optional*): seeds the initial latent, reproducible per backend.
- **latents** (*optional*): an explicit initial latent, for results identical across
  backends; with `strength` or `denoising_start`, the clean latent to start from.
- **image** / **strength** (*optional*): image-to-image (diffusers' `Img2ImgPipeline`):
  the `(batch, H, W, 3)` uint8 or `[0, 1]` float image is VAE-encoded (the
  `encode_latents` hook), noised to the `strength` point of the schedule and the
  remaining steps are run; `strength` defaults to the repo's `generate_args` (0.8, the
  SDXL refiner 0.3).
- **denoising_end** / **denoising_start** (*optional*): stop after, or resume from, a
  fraction of the schedule with no noise added: the SDXL base + refiner ensemble
  (`output_type="latent"` hands the base's latent over).
- **output_type** (*optional*): `"image"` (uint8 images) or `"latent"`.
- **conditioning** (*optional*): any further keyword argument is model-specific
  conditioning handed to `encode_prompt` (SDXL's `original_size` /
  `crops_coords_top_left` / `target_size`); a `negative_<name>` twin applies to the
  negative branch only and defaults to the positive value, the way `negative_input_ids`
  pairs with `input_ids`.

### BaseScheduler

The samplers a diffusion model steps with (`PNDMScheduler`, `DDIMScheduler`,
`EulerDiscreteScheduler`, `EulerAncestralDiscreteScheduler`,
`FlowMatchEulerDiscreteScheduler` (rectified flow, SD 3) in
`zeromodels.base.base_scheduler`) are weightless classes with the diffusers interface:
`set_timesteps(n)`, `scale_model_input(sample, t)`, `step(noise_pred, t, sample)` and
`init_noise_sigma`. `get_scheduler(config)` builds the one a diffusers scheduler config
dict names, which is what a repo's `scheduler_config` goes through; `model.scheduler` can
be swapped between calls. The Euler samplers take the diffusers `timestep_spacing`
(`linspace`, `leading`, `trailing`) and `interpolation_type` (`linear`, `log_linear`), and
the schedules (betas, `alphas_cumprod`, sigmas, timesteps) match diffusers to the bit.

## Preprocessing

All preprocessors share `PreprocessorMixin`, so they also get `from_weights`,
`from_hub_repo`, `from_variant`, and `from_hf`.

### BaseTokenizer

```python
tokenizer(inputs)
tokenizer.decode(ids, skip_special_tokens=True)
```

Text to token ids and back. Subclasses add `encode`, chat templating, and any
model-specific parsing (LocateAnything's `parse_boxes`, for example).

### BaseImageProcessor

```python
processor(images)
```

Images to `pixel_values`. Beyond the call, it carries the shared, backend-agnostic pixel
helpers every image model reuses: `resize`, `center_crop`, `pad`, `rescale`,
`normalize_image`, `rescale_and_normalize`, `preprocess_image`, and `stack_images` for
batching. Normalization constants live here too (`IMAGENET_STANDARD_MEAN`,
`OPENAI_CLIP_MEAN`, and friends).

Task-specific post-processing is added by subclasses:
`post_process_object_detection`, `post_process_semantic_segmentation`,
`post_process_depth_estimation`, `post_process_masks`.

### BaseAudioFeatureExtractor

```python
extractor(raw_speech, sampling_rate=16000)
```

Waveform to model input. What that means depends on the model: a log-mel spectrogram for
[Whisper](whisper.md) and [Granite Speech](granite_speech.md), normalized filterbanks for
[Speech2Text](speech2text.md), and the raw waveform itself for
[Moonshine](moonshine.md), which has no spectrogram step.

**`sampling_rate` tells the extractor what you are handing it; it does not resample.**
Feed 44.1 kHz audio while claiming 16 kHz and you get a confident, wrong transcript.

### BaseProcessor

```python
processor(text=None, images=None, audio=None, conversation=None, ...)
processor.decode(...)
```

The composite that bundles a tokenizer with an image processor or audio feature extractor
and renders chat templates. Multimodal models expose this as the single object you call.
Components are declared as class attributes, so `processor.tokenizer` and
`processor.image_processor` are always reachable.

## Quantization

### BaseQuantizer

```python
quantizer.quantize(weight, axis=0)
```

Base for the weight-only (tensor-level) quantizers. Helpers `normalize_axes(axis, ndim)`
and `single_axis(axis, ndim)` resolve contraction axes. The quantized layers built on this
(`QuantizedDense`, `QuantizedEinsumDense`, `QuantizedEmbedding`, `QuantizedExperts`) and
the `quantize_model` / `dequantize_model` entry points are covered in
[Quantization](quantization.md).

### ZmQuantizer

```python
get_zm_quantizer(quantization_config)  # {"quant_method": "mxfp4" | "int8" | ...}
```

Model-level quantizer, the transformers `HfQuantizer` analog. `from_weights` reads a
repo's `quantization_config` and runs the matching `ZmQuantizer` to swap in the packed /
int layers before the weights load, so the model stays quantization-agnostic. Dispatched
by `quant_method`: `Mxfp4ZmQuantizer` (GPT-OSS native experts) / `WeightOnlyZmQuantizer`
(int8 / int4 / fp8). Covered in [Quantization](quantization.md).

## Attention

### fused_attention

```python
fused_attention(
    query,
    key,
    value,
    scale,
    attention_mask=None,
    soft_cap=None,
    dropout=None,
    training=None,
    attn_implementation=None,
)
```

Scaled dot-product attention with a selectable backend kernel, used by every attention
layer in the library so a single implementation choice applies everywhere.

- **scale** (`float`): the `1/sqrt(head_dim)` factor, applied inside.
- **attention_mask** (*optional*): additive mask broadcastable to `(B, heads, T_q, T_kv)`.
- **soft_cap** (`float`, *optional*): logit soft-capping, used by Gemma 2.
- **attn_implementation** (`str`, *optional*): `"sdpa"`, `"fused"` or `"flash"` (below); `None` uses the implementation active in the current context, else the layer's own default.

**Implementations.** All three compute the same `softmax(QKᵀ · scale + mask) · V`; they differ in
memory, speed and portability:

| | `"sdpa"` (library default) | `"fused"` | `"flash"` |
|---|---|---|---|
| What runs | hand-written `matmul`, float32 `softmax`, `matmul` | `keras.ops.dot_product_attention`, the backend picks the kernel | the same op with `flash_attention=True` |
| torch | the math on any device / dtype | `scaled_dot_product_attention`: flash or memory-efficient kernel on a CUDA GPU in fp16 / bf16, else its math kernel | the flash kernel, or an error |
| JAX | the math | the XLA reference implementation (same memory as the math) | cuDNN flash on a capable GPU, or an error |
| TensorFlow | the math | falls back to the math | falls back to the math |
| Logits memory | the full `(B, heads, T_q, T_kv)` matrix, materialized in fp16 and again in float32 for the softmax | tiled, never materialized when a fused kernel applies | tiled |
| Masks / soft-cap / dropout | all supported | additive masks yes; a soft-cap or attention dropout falls back to the math | no masks; soft-cap / dropout fall back |

The math path is what every parity number in this library was measured with and what runs
everywhere identically. Its cost is quadratic memory: at 1024px the Stable Diffusion 3 joint
attention (4429 tokens) needs about 3.8 GB of float32 logits per block, which runs an 8 GB GPU
out of memory, while `"fused"` runs the same model at a 6.9 GB peak, 1.0 s/step on an RTX
4060 Laptop (the SDXL 1024px step drops from a 10.3 GB to a 7.6 GB peak). The fused kernels
accumulate in float32 and differ from the math only by rounding (about 3e-7 in float32).

**Precedence.** `Model.from_weights(attn_implementation=...)` activates the choice for the
model's build and every forward / generation step (a `ContextVar`, restored on exit, so it
never leaks to another model). A layer may carry its own default for when nothing is
chosen: the SD 3 MMDiT's `StableDiffusion3JointAttention` defaults to `"fused"` because of the sequence
length above; every other layer defaults to `"sdpa"`. An explicit choice always wins over a
layer default. Outside `from_weights`, `zeromodels.base.base_attention.use_attn_implementation("fused")`
wraps a build or a forward the same way.

See also [Utilities](utils.md) for the image, video, visualization, and label helpers.
