import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.vit.vit_layers import (
    ViTAddPositionEmbs,
    ViTClassDistToken,
    ViTLayerScale,
    ViTMultiHeadSelfAttention,
    ViTSwiGLUGate,
)
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .vit_config import ViTConfig

VIT_HUB_SIBLINGS = frozenset({"ViTModel", "ViTImageClassify"})


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_idx=0):
    x = layers.Dense(
        hidden_features, use_bias=True, name=f"blocks_{block_idx}_dense_1"
    )(inputs)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"blocks_{block_idx}_dense_2")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_2")(x)
    return x


def swiglu_block(inputs, hidden_features, out_features, drop=0.0, block_idx=0):
    x = layers.Dense(
        2 * hidden_features, use_bias=True, name=f"blocks_{block_idx}_weights_in"
    )(inputs)
    x = ViTSwiGLUGate(name=f"blocks_{block_idx}_swiglu")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_1")(x)
    x = layers.Dense(
        out_features, use_bias=True, name=f"blocks_{block_idx}_weights_out"
    )(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_2")(x)
    return x


def swiglu_hidden_features(embed_dim, mlp_ratio):
    hidden = int(embed_dim * mlp_ratio)
    return (int(hidden * 2 / 3) + 7) // 8 * 8


def transformer_block(
    inputs,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    qkv_bias=False,
    qk_norm=False,
    proj_drop=0.0,
    attn_drop=0.0,
    block_idx=0,
    layer_scale_init=None,
    use_swiglu=False,
):
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    x = ViTMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        block_prefix=f"blocks_{block_idx}",
    )(x)
    if layer_scale_init:
        x = ViTLayerScale(
            layer_scale_init=layer_scale_init, name=f"blocks_{block_idx}_layerscale_1"
        )(x)
    x = keras.layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    y = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    if use_swiglu:
        y = swiglu_block(
            y,
            hidden_features=swiglu_hidden_features(embed_dim, mlp_ratio),
            out_features=embed_dim,
            drop=proj_drop,
            block_idx=block_idx,
        )
    else:
        y = mlp_block(
            y,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            drop=proj_drop,
            block_idx=block_idx,
        )
    if layer_scale_init:
        y = ViTLayerScale(
            layer_scale_init=layer_scale_init, name=f"blocks_{block_idx}_layerscale_2"
        )(y)
    return keras.layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])


def vit_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    qkv_bias,
    qk_norm,
    drop_rate,
    attn_drop_rate,
    no_embed_class,
    use_distillation,
    layer_scale_init,
    image_size,
    data_format,
    resize_mode="bilinear",
    return_intermediates=False,
    return_stages=False,
    use_swiglu=False,
):
    if data_format == "channels_first":
        _, height, width = inputs.shape[1:]
    else:
        height, width, _ = inputs.shape[1:]
    if height is None:
        height = image_size
    if width is None:
        width = image_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="conv1",
    )(inputs)
    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((-1, embed_dim))(x)
    x = ViTClassDistToken(use_distillation=use_distillation, name="cls_token")(x)
    x = ViTAddPositionEmbs(
        name="pos_embed",
        no_embed_class=no_embed_class,
        use_distillation=use_distillation,
        grid_h=grid_h,
        grid_w=grid_w,
        resize_mode=resize_mode,
    )(x)
    intermediates = [x]
    x = layers.Dropout(drop_rate)(x)

    stages = []
    for i in range(depth):
        x = transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=drop_rate,
            attn_drop=attn_drop_rate,
            layer_scale_init=layer_scale_init,
            block_idx=i,
            use_swiglu=use_swiglu,
        )
        intermediates.append(x)
        stages.append(x)

    if return_intermediates:
        return intermediates
    x = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_layernorm")(x)
    stages.append(x)
    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ViTModel(BaseModel):
    """Instantiates the Vision Transformer (ViT) backbone.

    ViT splits the input image into fixed-size patches via a convolutional
    stem, linearly embeds each patch into a token, prepends a learnable CLS
    token (and optionally a distillation token) plus position embeddings,
    and processes the resulting sequence through ``depth`` standard
    transformer encoder blocks consisting of multi-head self-attention and
    MLP sub-blocks with residual connections and LayerNorm.

    Output is the last layer output before the classifier head: the
    final-LN normalized token sequence ``(B, num_tokens, embed_dim)`` where the
    first 1 (or 2 if ``use_distillation=True``) tokens are class /
    distillation tokens and the rest are spatial patch tokens.
    :class:`ViTImageClassify` composes this model and reads the class token(s)
    via ``backbone.output[:, 0]`` to produce logits.

    References:
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-block feature maps ending with the final-LN output.
            Defaults to `False`.
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, token embedding dimension. Defaults to `768`.
        depth: Integer, number of transformer encoder blocks.
            Defaults to `12`.
        num_heads: Integer, number of attention heads per block.
            Defaults to `12`.
        mlp_ratio: Float, hidden expansion ratio for the MLP sub-block.
            Defaults to `4.0`.
        qkv_bias: Boolean, whether to include bias in the QKV projection.
            Defaults to `True`.
        qk_norm: Boolean, whether to apply LayerNorm to Q and K inside
            attention. Defaults to `False`.
        drop_rate: Float, dropout rate after the position embedding and
            inside the MLP sub-block. Defaults to `0.0`.
        attn_drop_rate: Float, dropout rate applied to attention weights.
            Defaults to `0.0`.
        no_embed_class: Boolean, if `True`, position embeddings do not
            cover the class / distillation prefix tokens. Defaults to
            `False`.
        use_distillation: Boolean, if `True`, prepend a separate
            distillation token alongside the class token (DeiT-distilled
            style). Defaults to `False`.
        layer_scale_init: Optional float, initial gamma value for ViTLayerScale
            applied on both residual branches. If `None`, ViTLayerScale is
            disabled. Defaults to `None`.
        resize_mode: String, interpolation method used to resize the position
            embeddings when the input resolution differs from the pretrained
            grid. One of `"bilinear"` (default) or `"bicubic"`; use
            `"bicubic"` to match timm at non-native resolutions. Defaults to
            `"bilinear"`.
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
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``. Every released ViT
            checkpoint was trained with 0.5/0.5 statistics, which is
            `'inception'`; `'imagenet'` misclassifies.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model. Defaults to `"ViTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ViTConfig
    HUB_REPO_SIBLINGS = VIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ViTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ViTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_vit_timm_to_keras import transfer_vit_weights

        transfer_vit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
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
        resize_mode="bilinear",
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        name="ViTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

        input_shape = standardize_input_shape(image_size, data_format)
        image_size = (
            input_shape[0] if data_format == "channels_last" else input_shape[1]
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = vit_backbone_feature(
            x,
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
            resize_mode=resize_mode,
            image_size=image_size,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
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
        self.resize_mode = resize_mode
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
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
                "resize_mode": self.resize_mode,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ViTImageClassify(BaseModel):
    """Instantiates the Vision Transformer (ViT) classifier.

    This classifier wraps a :class:`ViTModel` backbone and attaches a
    single Dense layer on the CLS token (index 0 of the backbone's
    output) to produce ``num_classes`` class logits. When
    ``use_distillation=True``, two prediction heads are used (one on the
    CLS token, one on the distillation token) and their outputs are
    averaged. All architectural parameters are forwarded to the
    underlying :class:`ViTModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
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
            head. Defaults to `False`.
        layer_scale_init: Optional float, initial gamma value for ViTLayerScale
            applied on both residual branches. If `None`, ViTLayerScale is
            disabled. Defaults to `None`.
        resize_mode: String, interpolation method used to resize the position
            embeddings when the input resolution differs from the pretrained
            grid. One of `"bilinear"` (default) or `"bicubic"`; use
            `"bicubic"` to match timm at non-native resolutions. Defaults to
            `"bilinear"`.
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
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``. Every released ViT
            checkpoint was trained with 0.5/0.5 statistics, which is
            `'inception'`; `'imagenet'` misclassifies.
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
            named `f"{name}_backbone"`. Defaults to `"ViTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ViTConfig
    HUB_REPO_SIBLINGS = VIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_vit_timm_to_keras import transfer_vit_weights

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
        resize_mode="bilinear",
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ViTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = ViTModel(
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
            resize_mode=resize_mode,
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

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
        self.resize_mode = resize_mode
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
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
                "resize_mode": self.resize_mode,
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
