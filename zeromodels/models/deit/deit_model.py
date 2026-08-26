import keras
from keras import layers

from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.vit.vit_model import ViTImageClassify, ViTModel

from .deit_config import DeiTConfig

# The backbone (DeiTModel) and classifier (DeiTImageClassify) share the variant's
# weights repo, whose kf_config.json declares DeiTImageClassify.
DEIT_HUB_SIBLINGS = frozenset({"DeiTModel", "DeiTImageClassify"})


@keras.saving.register_keras_serializable(package="zeromodels")
class DeiTModel(ViTModel):
    """Instantiates the Data-efficient Image Transformer (DeiT) backbone.

    DeiT is a thin :class:`ViTModel` subclass that loads DeiT and DeiT III
    timm weights. The architecture mirrors ViT: patch embedding, learnable
    CLS token, position embedding, and ``depth`` standard transformer
    encoder blocks, but is paired with a data-efficient training recipe
    that enables strong ImageNet-only training. The distilled variants
    additionally prepend a learnable distillation token alongside the CLS
    token, which is trained against a teacher's predictions; this is
    enabled via ``use_distillation=True``.

    Output is the last layer output before the classifier head: the
    final-LN normalized token sequence ``(B, num_tokens, embed_dim)`` where the
    first 1 (or 2 if ``use_distillation=True``) tokens are class /
    distillation tokens and the rest are spatial patch tokens.
    :class:`DeiTImageClassify` composes this model and reads the class token(s)
    via ``backbone.output[:, 0]`` (and ``[:, 1]`` for the distillation
    token) to produce logits.

    References:
    - [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
    - [DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-block feature maps ending with the final-LN output.
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"DeiTModel"`.
        **kwargs: All architectural parameters of :class:`ViTModel`
            (``patch_size``, ``embed_dim``, ``depth``, ``num_heads``,
            ``mlp_ratio``, ``qkv_bias``, ``qk_norm``, ``drop_rate``,
            ``attn_drop_rate``, ``no_embed_class``, ``use_distillation``,
            ``layer_scale_init``, ``image_size``, ``include_normalization``,
            ``normalization_mode``, ``input_tensor``)
            are forwarded to the parent class.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = DeiTConfig
    HUB_REPO_SIBLINGS = DEIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with DeiTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = DeiTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_deit_timm_to_keras import transfer_deit_weights

        transfer_deit_weights(keras_model, state_dict)

    def __init__(self, as_backbone=False, name="DeiTModel", **kwargs):
        super().__init__(as_backbone=as_backbone, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeiTImageClassify(ViTImageClassify):
    """Instantiates the Data-efficient Image Transformer (DeiT) classifier.

    This classifier wraps a :class:`DeiTModel` backbone and attaches a
    single Dense layer on the CLS token (index 0 of the backbone's
    output) to produce ``num_classes`` class logits. When
    ``use_distillation=True``, a second Dense head is attached to the
    distillation token (index 1) and the two head outputs are averaged:
    matching the DeiT-distilled inference recipe. All architectural
    parameters are forwarded to the underlying :class:`DeiTModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
    - [DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
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
            cover the class / distillation prefix tokens. Defaults to
            `False`.
        use_distillation: Boolean, if `True`, prepend a separate
            distillation token alongside the class token and attach a
            second prediction head whose output is averaged with the CLS
            head: the DeiT-distilled inference recipe. Defaults to
            `False`.
        layer_scale_init: Optional float, initial gamma value for LayerScale
            applied on both residual branches (used by DeiT III). If
            `None`, LayerScale is disabled. Defaults to `None`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
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
            named `f"{name}_backbone"`. Defaults to `"DeiTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = DeiTConfig
    HUB_REPO_SIBLINGS = DEIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_deit_timm_to_keras import transfer_deit_weights

        transfer_deit_weights(keras_model, state_dict)

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
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="DeiTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = DeiTModel(
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
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
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
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
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
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
