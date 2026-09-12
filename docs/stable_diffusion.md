# Stable Diffusion

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Stable Diffusion 1.x, ported to pure Keras 3: latent text-to-image diffusion with a
UNet denoiser, a KL-regularized autoencoder and the CLIP ViT-L/14 text encoder. The
whole model is **one container**, `StableDiffusionModel`, a single functional graph with
the three towers as disconnected sub-graphs (the way `CLIPModel` holds its two), so a
hosted repo is one `model.weights.h5` and one `zm_config.json`.
`StableDiffusionTextToImage` is that same graph plus `generate`: the classifier-free
guidance denoising loop of `BaseDiffusion`, the diffusion counterpart of
`BaseGeneration`.

All five SD 1.x checkpoints (v1-1 to v1-5) share this architecture; only the weights
differ. The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Both data formats**: the UNet and VAE follow `keras.config.image_data_format()`
  (every conv takes it, GroupNorm normalizes the layout's channel axis, and the two
  token <-> grid boundaries transpose around their flatten under `channels_first`).
  Weights are layout-independent, so one hosted checkpoint serves both, and the
  converter is a `(O, I, H, W) -> (H, W, I, O)` kernel transpose and nothing else.
- **Block-level layers**: the UNet and VAE are built from composite Keras layers
  (`ResnetBlock2D`, `Transformer2DModel`, `CrossAttention`, ...). A functional graph keeps
  every node's output alive until the forward ends, and at 512px the UNet's per-op
  intermediates (4096x4096 attention maps above all) would need several GB; a layer's
  internals are freed when its call returns.
- **Schedulers match diffusers to the bit**: the training noise schedule is built in
  float32 the way torch does it, so `alphas_cumprod` and the Euler sigmas are identical
  and a 50-step run stays within float rounding of the reference.
- **Verified end to end**: at 512px, fp32, PNDM 50 steps and the same initial latent, the
  generated image matches diffusers' to within **1 uint8 level** (99.3% of pixels
  identical, PSNR 75 dB); the UNet, VAE and text encoder outputs agree to ~1e-6.

Links:

- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models (arXiv:2112.10752)](https://arxiv.org/abs/2112.10752)
- Reference implementation: [diffusers `StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- License: [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

See also [clip.md](clip.md) (the text encoder is `CLIPTextModel`).

## Variants

Preconverted, float32 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: UNet (860M) + VAE
(84M) + CLIP text encoder (123M), 1.07B parameters, 3.97 GB. All checkpoints are released
under the CreativeML OpenRAIL-M license, which carries use-based restrictions.

| Variant | Hub | Training |
|---|---|---|
| `stable-diffusion-v1-1` | [`zeromodels/stable-diffusion-v1-1`](https://huggingface.co/zeromodels/stable-diffusion-v1-1) | 237k steps at 256px on laion2B-en, then 194k steps at 512px on laion-high-resolution |
| `stable-diffusion-v1-2` | [`zeromodels/stable-diffusion-v1-2`](https://huggingface.co/zeromodels/stable-diffusion-v1-2) | v1-1 + 515k steps at 512px on laion-aesthetics v2 5+ |
| `stable-diffusion-v1-3` | [`zeromodels/stable-diffusion-v1-3`](https://huggingface.co/zeromodels/stable-diffusion-v1-3) | v1-2 + 195k steps at 512px, 10% text-conditioning dropout |
| `stable-diffusion-v1-4` | [`zeromodels/stable-diffusion-v1-4`](https://huggingface.co/zeromodels/stable-diffusion-v1-4) | v1-2 + 225k steps at 512px, 10% text-conditioning dropout |
| `stable-diffusion-v1-5` | [`zeromodels/stable-diffusion-v1-5`](https://huggingface.co/zeromodels/stable-diffusion-v1-5) | v1-2 + 595k steps at 512px, 10% text-conditioning dropout |

The 10% text-conditioning dropout of v1-3 onwards is what makes classifier-free guidance
work well; v1-5 is the usual default.

## API

Configs are typed: `StableDiffusionConfig` (composite) over `UNet2DConditionConfig`,
`AutoencoderKLConfig` and `StableDiffusionTextConfig` (a `CLIPTextConfig` with the
ViT-L/14 defaults), plus the checkpoint's `scheduler_config` and the CLIP special token
ids. Each repo's `zm_config.json` parses through it; the constructor stays flat, with the
sub-config fields prefixed `unet_` / `vae_` / `text_`.

### `StableDiffusionTextToImage`

The text-to-image task: the `StableDiffusionModel` container plus `BaseDiffusion`'s
`generate`. It supplies the hooks the mixin needs (`encode_prompt` on the CLIP tower,
the empty prompt's `unconditional_ids`, `predict_noise` on the UNet, `decode_latents` on
the VAE) and a `scheduler`, built from the config's `scheduler_config` (PNDM for every
SD 1.x checkpoint).

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, 77)` token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | accepted alongside the tokenizer's output; SD attends its padding tokens, so it is unused |
| `negative_input_ids` | `None` | `(batch, 77)` tokenized negative prompt for guidance; the empty prompt when unset |
| `num_inference_steps` | `None` | scheduler steps; the repo's `generate_args` (50) when unset |
| `guidance_scale` | `None` | classifier-free guidance strength; `generate_args` (7.5) when unset, `<= 1` disables it |
| `seed` | `None` | seed for the initial latent, reproducible per backend |
| `latents` | `None` | explicit `(batch, 64, 64, 4)` initial latent; identical results across backends |

Returns `(batch, 512, 512, 3)` uint8 numpy images. The denoiser call (the UNet plus the
guidance combine) runs as a per-backend compiled function cached on the instance,
`jax.jit` on JAX and `tf.function(jit_compile=True)` on TensorFlow, eager on Torch; its
shapes are the same at every step, so one compile serves the whole loop.

| Constructor arg | Default | Meaning |
|---|---|---|
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `unet_sample_size` | `64` | latent size the UNet graph is built for (image size / 8) |
| `vae_sample_size` | `512` | image size the VAE graphs are built for |
| `unet_*` / `vae_*` / `text_*` | SD 1.x | the flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `49406` / `49407` / `49407` | CLIP's special tokens, used to build the empty prompt |
| `scheduler_config` | `None` | the diffusers scheduler dict (`_class_name`, betas, `steps_offset`, ...) |

### `StableDiffusionModel`

The container: one functional model whose graph is three disconnected paths, one per
component. Inputs `{"sample", "timestep", "encoder_hidden_states"}` (UNet),
`{"image", "latent"}` (VAE encode / decode) and `{"token_ids", "padding_mask"}` (text);
outputs `{"noise_pred", "moments", "image", "text_embeds"}`. The components are exposed as
`.unet`, `.vae` and `.text_encoder` and share the container's weights, so loading the
container loads all three. It loads the same repo as the task class.

### `UNet2DConditionModel`

The denoiser (diffusers' `UNet2DConditionModel`): a symmetric down / mid / up UNet whose
ResNet blocks are conditioned on the sinusoidal timestep embedding and whose
`CrossAttn` blocks attend the latent to the text context. Inputs
`{"sample": (B, h, w, 4), "timestep": (B,), "encoder_hidden_states": (B, 77, 768)}`,
output `{"sample": (B, h, w, 4)}`.

| Arg | Default | Meaning |
|---|---|---|
| `sample_size` | `64` | latent spatial size the graph is built for |
| `in_channels` / `out_channels` | `4` | latent channels |
| `down_block_types` / `up_block_types` | SD 1.x | `CrossAttnDownBlock2D` x3 + `DownBlock2D`, mirrored |
| `block_out_channels` | `(320, 640, 1280, 1280)` | width per level |
| `layers_per_block` | `2` | ResNet blocks per level |
| `cross_attention_dim` | `768` | width of the text context |
| `num_attention_heads` | `8` | heads in the cross-attention blocks |
| `norm_num_groups` | `32` | GroupNorm groups |
| `text_seq_len` | `77` | static text length |

### `AutoencoderKL`

The VAE (diffusers' `AutoencoderKL`): `encode(image, sample=False)` returns the posterior
mean (or a sample) of a `(B, H/8, W/8, 4)` latent and `decode(latent)` the `(B, H, W, 3)`
image in `[-1, 1]`; multiply / divide by `scaling_factor` (0.18215) around the UNet, as
`generate` does. One functional graph with the encoder and decoder as two disconnected
paths.

| Arg | Default | Meaning |
|---|---|---|
| `in_channels` / `out_channels` | `3` | image channels |
| `latent_channels` | `4` | latent channels |
| `block_out_channels` | `(128, 256, 512, 512)` | width per level, x8 spatial compression |
| `layers_per_block` | `2` | ResNet blocks per encoder level |
| `norm_num_groups` | `32` | GroupNorm groups |
| `sample_size` | `512` | image size the graphs are built for |
| `scaling_factor` | `0.18215` | latent scaling |

### Schedulers

`zeromodels.base.base_scheduler` has the SD samplers as weightless classes with the
diffusers interface (`set_timesteps`, `scale_model_input`, `step`,
`init_noise_sigma`), configured the diffusers way:

| Scheduler | Notes |
|---|---|
| `PNDMScheduler` | the SD 1.x default (PLMS, `skip_prk_steps=True`) |
| `DDIMScheduler` | deterministic, `eta=0` |
| `EulerDiscreteScheduler` | Karras-style sigma sampler, `linspace` spacing |
| `EulerAncestralDiscreteScheduler` | stochastic Euler (`seed` for reproducibility) |

`get_scheduler(config)` builds the one named by a diffusers scheduler config dict, which
is what the model does with its `scheduler_config`; swap at any time:

```python
from zeromodels.base.base_scheduler import EulerDiscreteScheduler

model.scheduler = EulerDiscreteScheduler.from_config(model.config.scheduler_config)
```

## Preprocessing

### `StableDiffusionTokenizer`

The CLIP ViT-L/14 BPE tokenizer: `<|startoftext|>` / `<|endoftext|>` framing, truncated
and `<|endoftext|>`-padded to 77 tokens, the fixed context length the UNet's
cross-attention is built for. Calling it returns `{"input_ids", "attention_mask"}`
for a string or a list of strings.

```python
StableDiffusionTokenizer(hf_id=None, tokenizer_file=None, max_seq_len=77)
```

There is no image processor and no processor class: text-to-image takes token ids in
and hands uint8 images out.

## End-to-end example

### Single prompt

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.stable_diffusion import (
    StableDiffusionTextToImage,
    StableDiffusionTokenizer,
)

model = StableDiffusionTextToImage.from_weights("zeromodels/stable-diffusion-v1-5")
tokenizer = StableDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-v1-5")

inputs = tokenizer(
    "a red fox sitting in a field of lavender, golden hour, wildlife photography, sharp focus"
)
images = model.generate(**inputs, num_inference_steps=50, guidance_scale=7.5, seed=1)

Image.fromarray(images[0]).save("fox.png")  # (512, 512, 3) uint8
```

<img src="../assets/stable_diffusion_fox.jpg" alt="Stable Diffusion v1-5: a red fox sitting in a field of lavender at golden hour, 512px" width="380">

### Negative prompt and batching

Prompts batch through the tokenizer; a negative prompt is tokenized the same way and
passed as `negative_input_ids` (one per prompt, or one broadcast row).

```python
prompts = [
    "a watercolor painting of a lighthouse at sunset",
    "a red vintage car on a coastal road",
]
inputs = tokenizer(prompts)  # {"input_ids": (2, 77), "attention_mask": (2, 77)}
negative = tokenizer(["blurry, low quality"] * len(prompts))["input_ids"]

images = model.generate(**inputs, negative_input_ids=negative, num_inference_steps=30)
for i, image in enumerate(images):
    Image.fromarray(image).save(f"out_{i}.png")
```

### Reproducible across backends

`seed` reproduces a run on a given backend, but `keras.random` differs between torch,
jax and tensorflow. For an image that is identical everywhere, pass the initial latent
yourself (`(batch, 64, 64, 4)` at 512px):

```python
import numpy as np

latents = np.random.default_rng(0).standard_normal((1, 64, 64, 4)).astype("float32")
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

### Verified against diffusers

The fox prompt above with the same initial latent, PNDM 50 steps and guidance 7.5
through `StableDiffusionTextToImage` (left) and diffusers' `StableDiffusionPipeline`
(right), both fp32:

<img src="../assets/stable_diffusion_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of a red fox in lavender from the same latent" width="760">

```
tokenizer input_ids           identical
text_encoder                  max|d|=9.5e-06
unet noise_pred (t=981)       max|d|=9.5e-07
vae decode                    max|d|=3.5e-06
final image (uint8)           max|d|=1  mean|d|=0.0022  PSNR=74.7 dB  identical_pixels=99.34%
```

### Container only

```python
from zeromodels.models.stable_diffusion import StableDiffusionModel

sd = StableDiffusionModel.from_weights("zeromodels/stable-diffusion-v1-5")
latent = sd.vae.encode(image)  # (1, 512, 512, 3) in [-1, 1] -> (1, 64, 64, 4)
text = sd.text_encoder({"token_ids": ids, "padding_mask": mask})["last_hidden_state"]
```

### Other resolutions

The graphs are built for a fixed size, the weights are not. Rebuild at another size with
the constructor overrides (multiples of 64px; SD 1.x was trained at 512px):

```python
model = StableDiffusionTextToImage.from_weights(
    "zeromodels/stable-diffusion-v1-5", unet_sample_size=96, vae_sample_size=768
)
images = model.generate(**tokenizer("a mountain lake at dawn"))  # (1, 768, 768, 3)
```

## Data Format

**Both `channels_last` and `channels_first` are supported.** The models read
`keras.config.image_data_format()` when they are constructed (there is no
`data_format` argument), so set it once, before `from_weights`, and every tensor that
crosses the model boundary follows it:

| | `channels_last` (default) | `channels_first` |
|---|---|---|
| `latents` passed to `generate` | `(batch, 64, 64, 4)` | `(batch, 4, 64, 64)` |
| `AutoencoderKL.encode` input / `decode` output | `(batch, 512, 512, 3)` | `(batch, 3, 512, 512)` |
| UNet `sample` | `(batch, 64, 64, 4)` | `(batch, 4, 64, 64)` |

`generate` itself always hands back `(batch, height, width, 3)` uint8 images, whatever
the layout, so the PIL step is the same in both. The same hosted `model.weights.h5`
loads under either layout (conv kernels are `(kh, kw, in, out)` in both), and the two
runs agree to within 1 uint8 level (99.7% of pixels identical, the rest being the
NCHW / NHWC convolution kernels' rounding).

```python
import keras

keras.config.set_image_data_format("channels_first")

model = StableDiffusionTextToImage.from_weights("zeromodels/stable-diffusion-v1-5")
images = model.generate(
    **tokenizer("a bowl of ramen"), seed=0
)  # still (1, 512, 512, 3)
```

Note that `keras.config.set_image_data_format` is global state. Set it once at the top
of a script rather than toggling it between calls, since already-built models keep the
layout they were constructed with.

## Memory and speed

The fp32 container is 3.97 GB; a 512px, batch-1, guided run peaks at 6.5 GB of GPU
memory (the guided UNet call runs batch 2). On an RTX 4060 laptop the 50-step run takes
~25 s in eager Torch, about 1.5x diffusers' eager time. On JAX the compiled denoiser step
measured 2-3x faster than an op-by-op run; TensorFlow uses the same mechanism.

## Loading Fine-tuned Weights

The five hosted checkpoints are the supported weights; any repo laid out like them
(`zm_config.json` declaring `StableDiffusionModel`, `model.weights.h5`, `tokenizer.json`)
loads with `from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models:
convert a diffusers-format checkpoint once with
`zeromodels/models/stable_diffusion/convert_stable_diffusion_diffusers_to_keras.py`
(`build_from_diffusers(repo)`, `pip install zeromodels[conversion]`) and host the result.
