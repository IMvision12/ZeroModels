# Configuration

Every model in ZeroModels carries a typed **config**: a `BaseConfig` subclass that holds
the architecture hyperparameters as annotated fields. You rarely build one by hand, since
`from_weights` reads it from a repo's `zm_config.json` for you, but the config is what
turns a set of numbers into the right model, and it explains the shape of every
`zm_config.json` and every model constructor.

```python
from zeromodels.base import BaseConfig
```

## The one idea: flat constructor, nested serialize

A config lives a double life:

- **Model constructors are flat.** `CLIPModel(embed_dim=512, vision_hidden_dim=768, text_hidden_dim=512, ...)` takes every hyperparameter as a plain keyword. No nesting, no config object required.
- **Configs serialize nested.** `to_dict()` groups those same fields into `text_config` / `vision_config` blocks, mirroring the upstream Hugging Face config layout, and that nested dict is what gets written to `zm_config.json`.

`BaseConfig` is the bridge. It flattens a nested config down to the flat keyword set a
model wants (`constructor_kwargs()`), and serializes a flat set of fields back up into
nested blocks (`to_dict()`).

```python
from zeromodels.models.gemma3 import Gemma3Config

config = Gemma3Config()  # every field at its default
config.text_config  # a Gemma3TextConfig object
config.text_config.num_layers  # 26
```

## Building and reading a config

Each field is an annotated class attribute with a default, so a bare `Config()` is fully
populated, and any field is overridable by keyword. Sub-configs accept either a nested
config object or a plain dict:

```python
from zeromodels.models.gemma3 import Gemma3Config, Gemma3TextConfig

# override sub-config fields with a dict...
config = Gemma3Config(text_config={"embed_dim": 2560, "num_layers": 34})

# ...or with a real sub-config object
config = Gemma3Config(text_config=Gemma3TextConfig(embed_dim=2560, num_layers=34))
```

Two methods do the round trip:

- **`to_dict()`** returns the nested serialization, the block written to `zm_config.json`.
- **`from_dict(data)`** rebuilds a config from that dict. It accepts both the nested (v2) form and the older flat (v1) form, so every repo keeps loading.

```python
config.to_dict()
# {
#   "model_type": "gemma3",
#   "text_config": {"vocab_size": 262144, "embed_dim": 1152, ...},
# }
Gemma3Config.from_dict(config.to_dict())  # the same config back
```

## How a config feeds the model

The flat model constructor is fed by `constructor_kwargs()`, which flattens the
sub-configs to the keyword names the model expects:

```python
Gemma3Config().constructor_kwargs()
# {
#   "vocab_size": 262144, "embed_dim": 1152, ...,            # text_config, unprefixed
#   "vision_embed_dim": 1152, "vision_num_layers": 0, ...,   # vision_config, prefixed
#   "image_size": 896, "patch_size": 14,                     # kept native (group_extras)
#   "mm_tokens_per_image": 256, "image_token_id": 262144,    # top-level glue
# }
```

You almost never call this yourself: `from_weights` builds the config, calls
`constructor_kwargs()`, and hands the result to the model constructor.

## Two ways to declare the nesting

A single-tower config (one `text_config`, or one `vision_config`) needs no extra wiring;
its fields serialize under one block. Multi-tower models pick one of two mechanisms.

### Composite sub-configs (preferred)

Declare `sub_configs`, a `{block_name: SubConfigClass}` map. The config then holds real
sub-config objects, each its own `BaseConfig` with native field names, and `to_dict`
recurses into them. This is the form used by CLIP, Whisper, the DeepSeek-VL family, and
Gemma 3.

```python
class Gemma3Config(BaseConfig):
    model_type = "gemma3"

    sub_configs = {"text_config": Gemma3TextConfig, "vision_config": Gemma3VisionConfig}
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {"vision_config": ("image_size", "patch_size")}
    optional_sub_configs = ("vision_config",)  # 1B is text-only

    text_config: Gemma3TextConfig | dict | None = None
    vision_config: Gemma3VisionConfig | dict | None = None
    mm_tokens_per_image: int = 256  # top-level glue, in neither block
    image_token_id: int = 262144
```

- **`sub_config_prefixes`** says how each block's native fields map to the flat model constructor. `""` for the primary tower (its fields keep their names); `"vision_"` so `Gemma3VisionConfig.embed_dim` becomes the model's `vision_embed_dim`.
- **`group_extras`** lists a block's fields that keep their own name in the flat constructor anyway. Gemma 3's `image_size` and `patch_size` live in the vision block, but the model takes them unprefixed.
- **`optional_sub_configs`** names secondary blocks dropped from `to_dict()` when they are entirely default. The text-only Gemma 3 1B has a default (empty) vision tower, so its serialized config omits `vision_config` and the top-level image glue. Mandatory towers (CLIP's two encoders, Whisper's encoder and decoder) are always emitted, so leave this empty for them.

Fields not in any sub-config (`mm_tokens_per_image`, `image_token_id`) are **global
glue**: they stay at the top level of the serialized config, alongside `model_type`.

### Prefix auto-grouping

The lighter alternative, `config_groups`, keeps a single flat config and collapses fields
that share a prefix into a nested block only at serialize time. There are no separate
sub-config classes.

```python
class SomeConfig(BaseConfig):
    config_groups = {
        "vision_config": "vision_"
    }  # vision_embed_dim -> vision_config: {"embed_dim": ...}
    group_extras = {"vision_config": ("image_size",)}
    top_level_fields = ("image_token_id",)
```

Reach for this when a model has one optional group and a whole sub-config class would be
overkill. Prefer composite `sub_configs` otherwise, especially when you expect to rename
fields, since the sub-config class keeps them in one place.

## The primary block

Whichever mechanism is used, the config has one **primary block**, chosen automatically:
the sub-config whose prefix is `""`, or `text_config` if present, otherwise the first
sub-config. For a non-composite config it is `text_config` when the fields include a token
`vocab_size`, else `vision_config`. Set `main_config_key` to override.

## Class attributes

Declare these on a `BaseConfig` subclass to shape its serialization.

- **model_type** (`str`): the tag written as `"model_type"`, also used to match `hf:` checkpoints.
- **sub_configs** (`dict`): `{block_name: SubConfigClass}` for the composite form.
- **sub_config_prefixes** (`dict`): `{block_name: flat_prefix}`; `""` for the primary tower.
- **optional_sub_configs** (`tuple`): secondary blocks omitted from `to_dict()` when all-default.
- **config_groups** (`dict`): `{block_name: prefix}` for the prefix auto-grouping form.
- **group_extras** (`dict`): `{block_name: (fields,)}` kept native in the flat constructor.
- **top_level_fields** (`tuple`): glue fields kept top-level in the auto-grouping form.
- **main_config_key** (`str`, *optional*): override for the primary block.

## Methods

- **`to_dict()`**: nested serialization, the config block of `zm_config.json`.
- **`from_dict(data)`** (*classmethod*): rebuild from a nested (v2) or flat (v1) dict.
- **`constructor_kwargs()`**: the flat keyword set fed to the model constructor.
- **`field_names()`** (*classmethod*): every field name, sub-configs included.

## Sub-config classes are public

Each composite model exports its sub-config classes, so you can build and inspect a tower
on its own:

```python
from zeromodels.models.clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

CLIPTextConfig()  # just the text tower's fields
CLIPConfig(vision_config=CLIPVisionConfig(patch_size=16))
```

## Where the config comes from at load time

You do not construct a config to load a model. `from_weights("org/repo")` reads
`zm_config.json`, whose model fields are a `to_dict()` payload sitting alongside the repo
metadata (`weights`, `weight_dtype`, and friends), rebuilds the config with `from_dict`,
and flattens it through `constructor_kwargs()` into the model constructor. The `hf:` path
does the same starting from an upstream `config.json`. See
[Loading Weights](loading_weights.md) and [Main Classes](main_classes.md).
