import keras

from zeromodels.base import BaseModel
from zeromodels.models.siglip.siglip_model import (
    SigLIPImageClassify,
    SigLIPModel,
    SigLIPTextModel,
    SigLIPVisionModel,
    siglip_head,
)

from .siglip2_config import Siglip2Config

# SigLIP 2 has its own weights repos (separate from SigLIP). The full SigLIP2Model
# plus its heads all load from one repo per variant, whose zm_config.json declares
# the canonical SigLIP2ZeroShotClassify.
SIGLIP2_HUB_SIBLINGS = frozenset(
    {
        "SigLIP2Model",
        "SigLIP2VisionModel",
        "SigLIP2TextModel",
        "SigLIP2ZeroShotClassify",
        "SigLIP2ImageClassify",
    }
)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2VisionModel(SigLIPVisionModel):
    """SigLIP 2 vision tower as a standalone model.

    Thin subclass of :class:`SigLIPVisionModel`, architecture is
    identical; only the weights differ (SigLIP 2 has its own repos), and repo-id
    loading warm-starts the
    encoder from a :class:`SigLIP2Model` checkpoint instead of
    :class:`SigLIPModel`.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["last_hidden_state"]   # (B, num_patches, vision_hidden_dim)
        out["pooler_output"]       # (B, vision_hidden_dim): attention-pooled

    Construction:

    >>> SigLIP2VisionModel.from_weights("siglip2_base_p16_224")
    >>> SigLIP2VisionModel.from_weights("hf:google/siglip2-base-patch16-224")

    Reference:
        - `SigLIP 2: Multilingual Vision-Language Encoders with
          Improved Semantic Understanding, Localization, and Dense
          Features <https://arxiv.org/abs/2502.14786>`_

    Args (identical to :class:`SigLIPVisionModel`):
        image_size, patch_size, vision_hidden_dim,
        vision_num_layers, vision_num_heads, vision_mlp_dim,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Siglip2Config
    HUB_REPO_SIBLINGS = SIGLIP2_HUB_SIBLINGS

    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIP2Model

    def __init__(self, *args, name="SigLIP2VisionModel", **kwargs):
        super().__init__(*args, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2TextModel(SigLIPTextModel):
    """SigLIP 2 text tower as a standalone model.

    Thin subclass of :class:`SigLIPTextModel`, architecture is
    identical; differs only in the weights (SigLIP 2 has its own repos) and the
    Gemma-style ``vocab_size`` of
    256000 set by the SigLIP 2 config entries. ``from_weights``
    warm-starts the encoder from a :class:`SigLIP2Model` checkpoint.

    Output dict:

    .. code-block:: python

        out = model(token_ids)
        out["last_hidden_state"]   # (B, sequence_length, text_hidden_dim)
        out["pooler_output"]       # (B, embed_dim): last-token + Dense head

    Construction:

    >>> SigLIP2TextModel.from_weights("siglip2_base_p16_224")
    >>> SigLIP2TextModel.from_weights("hf:google/siglip2-base-patch16-224")

    Reference:
        - `SigLIP 2 <https://arxiv.org/abs/2502.14786>`_

    Args (identical to :class:`SigLIPTextModel`):
        vocab_size, embed_dim, text_hidden_dim, text_num_layers,
        text_num_heads, text_mlp_dim, max_seq_len,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Siglip2Config
    HUB_REPO_SIBLINGS = SIGLIP2_HUB_SIBLINGS

    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIP2Model

    def __init__(self, *args, name="SigLIP2TextModel", **kwargs):
        super().__init__(*args, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2Model(SigLIPModel):
    """SigLIP 2 dual encoder (no contrastive head).

    Thin subclass of :class:`SigLIPModel`, architecture is identical;
    only the weights (SigLIP 2 has its own repos)
    differs. Composes :class:`SigLIP2VisionModel` and
    :class:`SigLIP2TextModel` via the inherited ``__init__``.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ...})
        out["image_embeddings"]   # (B, vision_hidden_dim)
        out["text_embeddings"]    # (B, embed_dim)

    Construction:

    >>> SigLIP2Model.from_weights("siglip2_base_p16_224")
    >>> SigLIP2Model.from_weights("hf:google/siglip2-base-patch16-224")

    Reference:
        - `SigLIP 2 <https://arxiv.org/abs/2502.14786>`_

    Args (identical to :class:`SigLIPModel`):
        image_size, patch_size, vision_hidden_dim,
        vision_num_layers, vision_num_heads, vision_mlp_dim,
        vocab_size, embed_dim, text_hidden_dim, text_num_layers,
        text_num_heads, text_mlp_dim, max_seq_len,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Siglip2Config
    HUB_REPO_SIBLINGS = SIGLIP2_HUB_SIBLINGS

    HF_MODEL_TYPE = "siglip"

    def __init__(self, *args, name="SigLIP2Model", **kwargs):
        super().__init__(*args, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2ZeroShotClassify(BaseModel):
    """SigLIP 2 + sigmoid-similarity head for zero-shot classification.

    Composes :class:`SigLIP2Model` and adds the standard SigLIP head:
    L2-normalize both sides, compute the pairwise cosine-similarity
    matrix, then apply a learnable ``logit_scale`` and ``logit_bias``.
    Output is the ``(B, B)`` image-vs-text similarity logits.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ...})
        out["image_logits"]   # (B, B): image[i] vs text[j], scaled+biased
        out["text_logits"]    # (B, B): transpose of image_logits

    Construction:

    >>> SigLIP2ZeroShotClassify.from_weights("siglip2_base_p16_224")
    >>> SigLIP2ZeroShotClassify.from_weights("hf:google/siglip2-base-patch16-224")

    Reference:
        - `SigLIP 2 <https://arxiv.org/abs/2502.14786>`_

    Args (identical to :class:`SigLIPModel`):
        image_size, patch_size, vision_hidden_dim,
        vision_num_layers, vision_num_heads, vision_mlp_dim,
        vocab_size, embed_dim, text_hidden_dim, text_num_layers,
        text_num_heads, text_mlp_dim, max_seq_len,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Siglip2Config
    HUB_REPO_SIBLINGS = SIGLIP2_HUB_SIBLINGS

    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIP2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_mlp_dim=3072,
        vocab_size=256000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_mlp_dim=3072,
        max_seq_len=64,
        input_tensor=None,
        name="SigLIP2ZeroShotClassify",
        **kwargs,
    ):
        base = SigLIP2Model(
            image_size=image_size,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_mlp_dim=vision_mlp_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_mlp_dim=text_mlp_dim,
            max_seq_len=max_seq_len,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = siglip_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.image_size = base.image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_mlp_dim = text_mlp_dim
        self.max_seq_len = max_seq_len
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_mlp_dim": self.text_mlp_dim,
                "max_seq_len": self.max_seq_len,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2ImageClassify(SigLIPImageClassify):
    """SigLIP 2 vision tower + linear image-classification head.

    Thin subclass of :class:`SigLIPImageClassify`, architecture is
    identical; only the weights differ (SigLIP 2 has its own repos), and repo-id
    loading warm-starts the
    encoder from a :class:`SigLIP2Model` checkpoint instead of
    :class:`SigLIPModel`.

    .. code-block:: python

        model = SigLIP2ImageClassify.from_weights(
            "hf:<user>/siglip2-finetune-imagenet"
        )
        logits = model(images)              # (B, num_classes)

    Reference:
        - `SigLIP 2 <https://arxiv.org/abs/2502.14786>`_

    Args (identical to :class:`SigLIPImageClassify`):
        num_classes, image_size, patch_size, vision_hidden_dim,
        vision_num_layers, vision_num_heads, vision_mlp_dim,
        input_tensor, name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Siglip2Config
    HUB_REPO_SIBLINGS = SIGLIP2_HUB_SIBLINGS

    HF_MODEL_TYPE = "siglip"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "SigLIP2ImageClassify")
        super().__init__(*args, **kwargs)

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIP2Model
