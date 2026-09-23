# Qwen-Image

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + sharded
<code>*.weights.json</code> / <code>*.weights.h5</code> +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Qwen-Image, ported to pure Keras 3: latent text-to-image flow-matching with a
60-layer double-stream MMDiT denoiser, a Wan-derived KL autoencoder and the
Qwen2.5-VL-7B Instruct text tower. The whole model is **one container**,
`QwenImageModel`, a single functional graph with the three towers as disconnected
sub-graphs (the way `CLIPModel` holds its two), so a hosted repo is one set of
sharded weights and one `zm_config.json`. `QwenImageTextToImage` is that same
graph plus `generate`: the true-CFG denoising loop of `BaseDiffusion`, the
diffusion counterpart of `BaseGeneration`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Packed latents**: the denoiser sees `(B, H/2 · W/2, 64)` tokens; pack / unpack
  match Diffusers' 2×2 patching around the 16-channel VAE latent grid
  (`AutoencoderKLQwenImage`, 8× spatial, channels-last NDHWC with `T=1`).
- **True CFG**: when `guidance_scale > 1`, two transformer forwards and Diffusers'
  prediction-norm renormalization (`true_cfg_scale`); a space negative prompt is
  the default when none is passed.
- **bf16-safe dual-stream clip**: Diffusers clips dual-stream activations to the
  fp16 finite range only under float16; the port matches that, so bf16 runs keep
  the ~1e7 magnitudes Qwen-Image produces instead of collapsing to noise.
- **Schedulers match Diffusers**: `FlowMatchEulerDiscreteScheduler` with dynamic
  resolution shifting (`mu` from packed sequence length) so timesteps agree with
  the reference within float rounding.
- **Verified against Diffusers**: transformer, text encoder, VAE and scheduler
  outputs agree with the reference within float noise (see
  [Verified against Diffusers](#verified-against-diffusers)).

Links:

- Paper: [Qwen-Image Technical Report (arXiv:2508.02324)](https://arxiv.org/abs/2508.02324)
- Reference implementation: [diffusers `QwenImagePipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- License: [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image/blob/main/LICENSE)

See also [qwen2_5_vl.md](qwen2_5_vl.md) (the text tower),
[stable_diffusion_3.md](stable_diffusion_3.md) (flow-match / MMDiT-style diffusion
in ZeroModels).

## Variants

Preconverted, bfloat16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: MMDiT
transformer + VAE + Qwen2.5-VL text tower, ~53 GiB at 16-bit. Released under
Apache-2.0.

| Variant | Hub | Source |
|---|---|---|
| `qwen-image` | [`zeromodels/qwen-image`](https://huggingface.co/zeromodels/qwen-image) | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |

Default `generate_args`: 50 flow-match steps, `guidance_scale=4.0`, 1024×1024.

## API

Configs are typed: `QwenImageConfig` (composite, `model_type` `"qwen_image"`) over
`QwenImageTransformerConfig`, `QwenImageVAEConfig` and `QwenImageTextConfig`, plus
the checkpoint's `scheduler_config` and the Qwen special token ids. Each repo's
`zm_config.json` parses through it; the constructor stays flat, with the
sub-config fields prefixed `transformer_` / `vae_` / `text_`.

### `QwenImageTextToImage`

The text-to-image task: the `QwenImageModel` container plus `BaseDiffusion`'s
`generate`. It supplies the hooks the mixin needs (`encode_prompt` on the text
tower with ChatML template drop, `unconditional_ids`, `predict_noise` on the
transformer, `decode_latents` on the VAE with mean/std un-normalization) and a
`scheduler`, built from the config's `scheduler_config`
(`FlowMatchEulerDiscreteScheduler` with dynamic shifting).

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    negative_attention_mask=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    height=None,
    width=None,
    output_type="image",
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | ChatML-templated token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | padding mask from the tokenizer |
| `negative_input_ids` | `None` | tokenized negative prompt for true CFG; a space / empty prompt when unset |
| `negative_attention_mask` | `None` | mask for the negative ids |
| `num_inference_steps` | `None` | scheduler steps; the repo's `generate_args` (50) when unset |
| `guidance_scale` | `None` | true CFG strength; `generate_args` (4.0) when unset, `<= 1` disables it |
| `seed` | `None` | seed for the initial latent, reproducible per backend |
| `latents` | `None` | explicit packed initial latent; identical results across backends |
| `height` / `width` | `None` | output pixel size; `default_sample_size * 8` (1024) when unset; rounded to VAE×pack multiples |
| `output_type` | `"image"` | `"image"` for uint8 RGB, `"latent"` for packed latents |

Returns `(batch, height, width, 3)` uint8 numpy images.

| Constructor arg | Default | Meaning |
|---|---|---|
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `transformer_sample_size` | `128` | latent side the transformer graph is built for (image / 8) |
| `vae_sample_size` | `1024` | image size the VAE graphs are built for |
| `transformer_*` / `vae_*` / `text_*` | Qwen-Image | the flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `151643` / `151645` / `151643` | Qwen special tokens |
| `scheduler_config` | `None` | the Diffusers scheduler dict (`_class_name`, shift, dynamic shifting, ...) |
| `prompt_template_encode_start_idx` | `34` | ChatML template tokens dropped after the text encoder |
| `max_sequence_length` | `512` | prompt embed length after the template drop |

### `QwenImageModel`

The container: one functional model whose graph is three disconnected paths, one per
component. Inputs cover the transformer (`sample`, `timestep`,
`encoder_hidden_states`, `encoder_hidden_states_mask`), the VAE (`image`, `latent`)
and the text tower (`token_ids`, `padding_mask`); outputs include the packed
velocity, VAE moments / sample, and prompt embeds. The components are exposed as
`.transformer`, `.vae` and `.text_encoder` and share the container's weights, so
loading the container loads all three. It loads the same repo as the task class.

### `QwenImageTransformer2DModel`

The denoiser (Diffusers' `QwenImageTransformer2DModel`): a 60-layer double-stream
MMDiT over **packed** latents and text features, with MS-RoPE
(`axes_dims_rope=(16, 56, 56)`). Inputs
`{"sample": (B, seq, 64), "timestep": (B,), "encoder_hidden_states": (B, text_seq, 3584)}`
(optional `encoder_hidden_states_mask`), output `{"sample": (B, seq, 64)}` packed
velocity.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `2` | latent pack side |
| `in_channels` | `64` | packed token width (`16 × 2 × 2`) |
| `out_channels` | `16` | unpacked latent channels |
| `num_layers` | `60` | dual-stream DiT blocks |
| `attention_head_dim` / `num_attention_heads` | `128` / `24` | head geometry (inner dim 3072) |
| `joint_attention_dim` | `3584` | text feature width from Qwen2.5-VL |
| `axes_dims_rope` | `(16, 56, 56)` | MS-RoPE axis splits |
| `sample_size` | `128` | latent spatial side the graph is built for |
| `text_seq_len` | `512` | static text length after the template drop |

### `AutoencoderKLQwenImage`

The VAE (Diffusers' `AutoencoderKLQwenImage`): `encode(image, sample=False)` returns
the posterior mean (or a sample) of a `(B, H/8, W/8, 16)` latent and
`decode(latent)` the `(B, H, W, 3)` image in `[-1, 1]`. Latent normalisation uses
`latents_mean` / `latents_std` around the denoiser, as `generate` does. One
functional graph with the encoder and decoder as two disconnected paths
(channels-last NDHWC, `T=1` for still images).

| Arg | Default | Meaning |
|---|---|---|
| `base_dim` | `96` | base channel width |
| `z_dim` | `16` | latent channels |
| `dim_mult` | `(1, 2, 4, 4)` | width multipliers per level |
| `num_res_blocks` | `2` | residual blocks per level |
| `temperal_downsample` | `(False, True, True)` | which levels downsample time (unused at `T=1`) |
| `sample_size` | `1024` | image size the graphs are built for |
| `latents_mean` / `latents_std` | Qwen-Image | per-channel latent normalisation |

### Schedulers

`zeromodels.base.base_scheduler` has the flow-match sampler with the Diffusers
interface (`set_timesteps`, `step`, `init_noise_sigma`), configured the Diffusers
way:

| Scheduler | Notes |
|---|---|
| `FlowMatchEulerDiscreteScheduler` | Qwen-Image default; `use_dynamic_shifting=True`, exponential time shift |

`get_scheduler(config)` builds the one named by a Diffusers scheduler config dict,
which is what the model does with its `scheduler_config`. At generate time the
task sets timesteps with resolution-dependent `mu` and a linspace sigma schedule,
matching Diffusers.

## Preprocessing

### `QwenImageTokenizer`

The Qwen2 BPE tokenizer with the Diffusers ChatML prompt template wrapped around
each string. Calling it returns `{"input_ids", "attention_mask"}` for a string or
a list of strings. `generate` / `encode_prompt` drop the template prefix
(`prompt_template_encode_start_idx=34`) after the text encoder and keep at most
`max_sequence_length` (512) tokens.

```python
QwenImageTokenizer(hf_id=None, tokenizer_file=None, max_seq_len=1024)
```

There is no image processor and no processor class: text-to-image takes token ids
in and hands uint8 images out.

## End-to-end example

### Single prompt

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.qwen_image import (
    QwenImageTextToImage,
    QwenImageTokenizer,
)

model = QwenImageTextToImage.from_weights("zeromodels/qwen-image")
tokenizer = QwenImageTokenizer.from_weights("zeromodels/qwen-image")

inputs = tokenizer(
    "a coffee shop entrance with a chalkboard sign reading 'Open', warm afternoon light"
)
images = model.generate(
    **inputs, height=1024, width=1024, num_inference_steps=50, guidance_scale=4.0, seed=1
)

Image.fromarray(images[0]).save("coffee.png")  # (1024, 1024, 3) uint8
```

### Negative prompt and batching

Prompts batch through the tokenizer; a negative prompt is tokenized the same way
and passed as `negative_input_ids` (one per prompt, or one broadcast row).

```python
prompts = [
    "a watercolor painting of a lighthouse at sunset",
    "a red vintage car on a coastal road",
]
inputs = tokenizer(prompts)
negative = tokenizer(["blurry, low quality"] * len(prompts))["input_ids"]

images = model.generate(
    **inputs,
    negative_input_ids=negative,
    height=512,
    width=512,
    num_inference_steps=30,
)
for i, image in enumerate(images):
    Image.fromarray(image).save(f"out_{i}.png")
```

### Reproducible across backends

`seed` reproduces a run on a given backend, but `keras.random` differs between
torch, jax and tensorflow. For an image that is identical everywhere, pass the
initial packed latent yourself. At 1024px the spatial noise before packing is
`(batch, 128, 128, 16)`; after packing it is `(batch, 4096, 64)`:

```python
import numpy as np

from zeromodels.models.qwen_image.qwen_image_model import pack_latents

h, w, channels = 128, 128, 16
noise = np.random.default_rng(0).standard_normal((1, h, w, channels)).astype("float32")
latents = pack_latents(noise, h, w)
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

### Verified against Diffusers

Component checks against Diffusers / Transformers with the same weights (fp32
unless noted):

```
scheduler timesteps / mu     within 1e-4 / identical
transformer (fp32, N blocks) relative error ~8e-6
transformer (bf16, N blocks) cosine similarity 0.99998
text encoder (fp32)          relative error ~1.7e-6
VAE encode / decode          matches Diffusers (NCDHW <-> NDHWC transpose)
```

End-to-end image parity depends on dtype and resolution; prefer bf16 (or fp32)
over fp16 — Qwen-Image activations overflow the fp16 range without the Diffusers
clip path.

### Container only

```python
from zeromodels.models.qwen_image import QwenImageModel

qi = QwenImageModel.from_weights("zeromodels/qwen-image")
latent = qi.vae.encode(image)  # (1, H, W, 3) in [-1, 1] -> (1, H/8, W/8, 16)
text = qi.text_encoder({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]
```

### Other resolutions

The graphs are built for a fixed size, the weights are not. Rebuild at another
size with the constructor overrides (multiples of 16px after VAE×pack; the
checkpoint targets 1024px):

```python
model = QwenImageTextToImage.from_weights(
    "zeromodels/qwen-image",
    transformer_sample_size=64,
    vae_sample_size=512,
)
images = model.generate(
    **tokenizer("a mountain lake at dawn"), height=512, width=512
)  # (1, 512, 512, 3)
```

## Data Format

**Channels-last only for the VAE.** `AutoencoderKLQwenImage` is built as NDHWC
`Conv3D` (with `T=1` for still images); the transformer works on packed sequence
tokens and has no spatial layout. `generate` always hands back
`(batch, height, width, 3)` uint8 images.

| | Shape |
|---|---|
| `latents` passed to `generate` (packed) | `(batch, (H/16)·(W/16), 64)` |
| VAE encode input / decode output | `(batch, H, W, 3)` |
| VAE latent | `(batch, H/8, W/8, 16)` |
| Transformer `sample` | `(batch, seq, 64)` |

## Memory and speed

The bf16 / float16 container is about 53 GiB. Building the full graph on a ~40 GB
GPU OOMs; build on CPU (or load with component offload) and move one tower at a
time. A guided 1024px run needs substantial activation headroom on top of the
~38 GiB transformer; 512px is the practical default on 40 GB cards. Prefer the
fused attention path (`keras.ops.dot_product_attention` / torch SDPA) over the
plain matmul softmax default.

## Loading Fine-tuned Weights

The hosted checkpoint is the supported weight; any repo laid out like it
(`zm_config.json` declaring `QwenImageModel`, sharded `*.weights.json` /
`*.weights.h5`, `tokenizer.json`) loads with `from_weights("<org>/<repo>")`. The
`hf:` prefix raises for diffusion models: convert a Diffusers-format checkpoint
once with
`zeromodels/models/qwen_image/convert_qwen_image_diffusers_to_keras.py`
(`transfer_qwen_image(repo)`, `pip install zeromodels[conversion]`) and host the
result.
