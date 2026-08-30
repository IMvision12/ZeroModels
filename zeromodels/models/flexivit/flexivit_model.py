import keras
from keras import layers

from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.vit.vit_model import ViTImageClassify, ViTModel

from .flexivit_config import FlexiViTConfig

# The backbone (FlexiViTModel) and classifier (FlexiViTImageClassify) share the
# variant's weights repo, whose zm_config.json declares FlexiViTImageClassify.
FLEXIVIT_HUB_SIBLINGS = frozenset({"FlexiViTModel", "FlexiViTImageClassify"})


@keras.saving.register_keras_serializable(package="zeromodels")
class FlexiViTModel(ViTModel):
    """Instantiates the FlexiViT (Flexible Vision Transformer) backbone.

    FlexiViT is a thin :class:`ViTModel` subclass that loads FlexiViT timm
    weights. The architecture is a standard ViT: patch embedding, CLS
    token, position embedding, and ``depth`` transformer encoder blocks,
    but is trained with randomized patch sizes per minibatch, enabling
    flexible resampling of the patch embedding kernel and positional
    embeddings at inference for any patch size from a single set of
    weights. FlexiViT checkpoints use ``no_embed_class=True`` so position
    embeddings cover only patch tokens (the CLS token gets none), which is
    what makes positional-embedding interpolation across patch sizes well
    defined.

    Output is the last layer output before the classifier head: the
    final-LN normalized token sequence ``(B, num_tokens, embed_dim)`` where the
    first token is the class token and the rest are spatial patch tokens.
    :class:`FlexiViTImageClassify` composes this model and reads
    ``backbone.output[:, 0]`` to produce logits.

    References:
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-block feature maps ending with the final-LN output.
            Defaults to `False`.
        name: String, the name of the model. Defaults to
            `"FlexiViTModel"`.
        **kwargs: All architectural parameters of :class:`ViTModel`
            (``patch_size``, ``embed_dim``, ``depth``, ``num_heads``,
            ``mlp_ratio``, ``qkv_bias``, ``qk_norm``, ``drop_rate``,
            ``attn_drop_rate``, ``no_embed_class``, ``use_distillation``,
            ``layer_scale_init``, ``input_tensor``)
            are forwarded to the parent class.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = FlexiViTConfig
    HUB_REPO_SIBLINGS = FLEXIVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with FlexiViTImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = FlexiViTImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from zeromodels.models.vit.convert_vit_timm_to_keras import (
            transfer_vit_weights,
        )

        transfer_vit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        image_size=240,
        name="FlexiViTModel",
        **kwargs,
    ):
        super().__init__(
            as_backbone=as_backbone,
            image_size=image_size,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class FlexiViTImageClassify(ViTImageClassify):
    """Instantiates the FlexiViT (Flexible Vision Transformer) classifier.

    This classifier wraps a :class:`FlexiViTModel` backbone and attaches a
    single Dense layer on the CLS token (index 0 of the backbone's
    output) to produce ``num_classes`` class logits. FlexiViT checkpoints
    are trained with ``no_embed_class=True`` so the positional embedding
    can be resampled for any patch size at inference. All architectural
    parameters are forwarded to the underlying :class:`FlexiViTModel`;
    only ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Args:
        patch_size: Integer, conv-stem patch size in pixels. Can be set
            at inference to any value supported by the resampled
            positional embedding. Defaults to `16`.
        embed_dim: Integer, token embedding dimension. Defaults to `768`.
        depth: Integer, number of transformer encoder blocks in the
            backbone. Defaults to `12`.
        num_heads: Integer, number of attention heads per block.
            Defaults to `12`.
        mlp_ratio: Float, hidden expansion ratio for the MLP sub-block.
            Defaults to `4.0`.
        qkv_bias: Boolean, whether to include bias in the QKV projection.
            Defaults to `True`.
        qk_norm: Boolean, whether to apply LayerNorm to Q and K inside
            attention. Defaults to `False`.
        drop_rate: Float, dropout rate after the position embedding,
            inside the MLP sub-block, and before the classifier head.
            Defaults to `0.0`.
        attn_drop_rate: Float, dropout rate applied to attention weights.
            Defaults to `0.0`.
        no_embed_class: Boolean, if `True`, position embeddings do not
            cover the class / distillation prefix tokens: required for
            the flexible patch-size positional-embedding resampling.
            Defaults to `False` here for ``__init__`` compatibility;
            FlexiViT checkpoints set this to `True`.
        use_distillation: Boolean, if `True`, prepend a separate
            distillation token alongside the class token. Defaults to
            `False`.
        layer_scale_init: Optional float, initial gamma value for LayerScale
            applied on both residual branches. If `None`, LayerScale is
            disabled. Defaults to `None`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `240`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"FlexiViTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = FlexiViTConfig
    HUB_REPO_SIBLINGS = FLEXIVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from zeromodels.models.vit.convert_vit_timm_to_keras import (
            transfer_vit_weights,
        )

        transfer_vit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        layer_scale_init=None,
        image_size=240,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="FlexiViTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = FlexiViTModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            layer_scale_init=layer_scale_init,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = backbone.output
        if use_distillation:
            cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
            dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(x)
            cls_token = layers.Dropout(drop_rate)(cls_token)
            dist_token = layers.Dropout(drop_rate)(dist_token)
            cls_head = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(cls_token)
            dist_head = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions_dist",
            )(dist_token)
            out = (cls_head + dist_head) / 2
        else:
            tok = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
            tok = layers.Dropout(drop_rate)(tok)
            out = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(tok)

        super(ViTImageClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.layer_scale_init = layer_scale_init
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super(ViTImageClassify, self).get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "layer_scale_init": self.layer_scale_init,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config
