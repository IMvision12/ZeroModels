import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.cait.cait_layers import (
    CaiTAddPositionEmbs,
    CaiTClassAttention,
    CaiTClassDistToken,
    CaiTLayerScale,
    CaiTStochasticDepth,
    CaiTTalkingHeadAttention,
)
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .cait_config import CaiTConfig

CAIT_HUB_SIBLINGS = frozenset({"CaiTModel", "CaiTImageClassify"})


def mlp_block(x, hidden_dim, out_dim, drop_rate=0.0, block_prefix=None):
    """Two-layer MLP block: Dense -> GELU -> Drop -> Dense -> Drop.

    Args:
        x: Input token tensor of shape ``(B, N, D)``.
        hidden_dim: Output dimension of the first Dense layer.
        out_dim: Output dimension of the second Dense layer.
        drop_rate: Dropout rate applied after each Dense.
        block_prefix: Optional prefix used to name the inner Dense layers.

    Returns:
        Tensor of shape ``(B, N, out_dim)``.
    """
    x = layers.Dense(
        hidden_dim,
        activation="gelu",
        name=f"{block_prefix}_dense_1" if block_prefix else None,
    )(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(out_dim, name=f"{block_prefix}_dense_2" if block_prefix else None)(
        x
    )
    x = layers.Dropout(drop_rate)(x)
    return x


def layer_scale_talking_head_block(
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    drop_rate=0.0,
    layer_scale_init=1e-5,
    block_prefix="block",
):
    """CaiT main block: LN -> TalkingHeadAttn -> CaiTLayerScale -> SD -> Add -> LN -> MLP -> CaiTLayerScale -> SD -> Add.

    Args:
        x: Input token tensor of shape ``(B, N, embed_dim)``.
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP block.
        drop_rate: Stochastic-depth drop rate applied to each residual branch.
        layer_scale_init: Initial value for the CaiTLayerScale per-channel gamma.
        block_prefix: Prefix used to name layers inside the block.

    Returns:
        Tensor of shape ``(B, N, embed_dim)`` after both residual branches.
    """
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(x)
    attn = CaiTTalkingHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)
    attn = CaiTLayerScale(
        layer_scale_init=layer_scale_init, name=f"{block_prefix}_layerscale_1"
    )(attn)
    if drop_rate > 0:
        attn = CaiTStochasticDepth(drop_rate)(attn)
    x = layers.Add(name=f"{block_prefix}_add_1")([x, attn])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(x)
    mlp = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )
    mlp = CaiTLayerScale(
        layer_scale_init=layer_scale_init, name=f"{block_prefix}_layerscale_2"
    )(mlp)
    if drop_rate > 0:
        mlp = CaiTStochasticDepth(drop_rate)(mlp)
    return layers.Add(name=f"{block_prefix}_add_2")([x, mlp])


def layer_scale_class_attn_block(
    cls_token,
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    layer_scale_init=1e-5,
    block_prefix="block_token_only",
):
    """Class-attention-only block: cls_token attends to patch tokens, then MLP.

    Args:
        cls_token: Class token tensor of shape ``(B, 1, embed_dim)``.
        x: Patch tokens of shape ``(B, N, embed_dim)``.
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP block.
        layer_scale_init: Initial value for the CaiTLayerScale per-channel gamma.
        block_prefix: Prefix used to name layers inside the block.

    Returns:
        Updated ``cls_token`` tensor of shape ``(B, 1, embed_dim)``.
    """
    concat = layers.Concatenate(axis=1)([cls_token, x])
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(
        concat
    )
    cls = CaiTClassAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)
    cls = CaiTLayerScale(
        layer_scale_init=layer_scale_init, name=f"{block_prefix}_layerscale_1"
    )(cls)
    cls_token = layers.Add(name=f"{block_prefix}_add_1")([cls_token, cls])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(
        cls_token
    )
    mlp = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )
    mlp = CaiTLayerScale(
        layer_scale_init=layer_scale_init, name=f"{block_prefix}_layerscale_2"
    )(mlp)
    return layers.Add(name=f"{block_prefix}_add_2")([cls_token, mlp])


def cait_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    drop_path_rate,
    data_format,
    depth_token_only=2,
    return_stages=False,
):
    """CaiT stem + talking-head blocks + class-attn blocks.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        patch_size: Conv-stem patch size in pixels.
        embed_dim: Token embedding dimension.
        depth: Number of TalkingHead transformer blocks.
        num_heads: Number of attention heads per block.
        drop_path_rate: Maximum stochastic-depth drop rate (linearly scaled
            across the ``depth`` blocks).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        depth_token_only: Number of trailing class-attention blocks.
        return_stages: If ``True``, return a list of per-block (talking-head
            + class-attn) intermediate outputs ending with the final-LN
            output. Otherwise return only the final-LN output.

    Returns:
        ``(B, 1+N, D)`` tensor of final-LN normalized tokens: CLS at index 0
        followed by ``N = (H/patch_size) * (W/patch_size)`` patch tokens.
        When ``return_stages=True``, returns a list of intermediate tensors;
        the last entry is the same final-LN output.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        grid_h = inputs.shape[2] // patch_size
        grid_w = inputs.shape[3] // patch_size
        x = ops.transpose(x, (0, 2, 3, 1))
    else:
        grid_h = inputs.shape[1] // patch_size
        grid_w = inputs.shape[2] // patch_size

    x = layers.Reshape((-1, embed_dim))(x)
    x = CaiTAddPositionEmbs(
        grid_h=grid_h, grid_w=grid_w, no_embed_class=True, name="pos_embed"
    )(x)

    stages = []
    dpr = list(ops.linspace(0.0, drop_path_rate, depth))
    for i in range(depth):
        x = layer_scale_talking_head_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=dpr[i],
            layer_scale_init=1e-5,
            block_prefix=f"blocks_{i}",
        )
        stages.append(x)

    cls_token = CaiTClassDistToken(name="cls_token")(x)
    for i in range(depth_token_only):
        cls_token = layer_scale_class_attn_block(
            cls_token,
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            layer_scale_init=1e-5,
            block_prefix=f"blocks_token_only_{i}",
        )
        stages.append(cls_token)

    x = layers.Concatenate(axis=1, name="cat_cls_patch")([cls_token, x])
    x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)
    stages.append(x)
    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class CaiTModel(BaseModel):
    """Instantiates the Class-Attention in Image Transformers (CaiT) backbone.

    CaiT refines the vanilla ViT recipe in two ways that make very deep
    transformers trainable for image classification: (1) talking-head
    self-attention paired with a learnable per-channel CaiTLayerScale
    (initialized at ``1e-5``) plus stochastic depth on every residual
    branch, so deep stacks converge without divergence; and (2) a
    dedicated class-attention stage where the model first runs ``depth``
    blocks on patch tokens alone, then appends a class token and updates
    it with ``depth_token_only`` extra class-attention blocks while the
    patch tokens are frozen, so the CLS token aggregates information
    without contaminating the patch representation.

    Output is the last layer output before the classifier head:
    the final-LN normalized token sequence ``(B, 1+N, D)`` with the CLS
    token at index 0 followed by ``N = (H/patch_size) * (W/patch_size)``
    patch tokens. :class:`CaiTImageClassify` composes this model and reads
    ``[:, 0]`` from the output to produce logits.

    References:
    - [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            feature maps after each talking-head block, each class-attn
            block, and the final-LN output. Defaults to `False`.
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, token embedding dimension. Determines model
            width: 192 (XXS), 288 (XS), 384 (S), 768 (M).
            Defaults to `192`.
        depth: Integer, number of patch-only talking-head transformer
            blocks. Defaults to `24`.
        num_heads: Integer, number of attention heads per block (both
            patch-only and class-attention). Defaults to `4`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across the
            ``depth`` patch blocks. Defaults to `0.0`.
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
        name: String, the name of the model. Defaults to `"CaiTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = CaiTConfig
    HUB_REPO_SIBLINGS = CAIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with CaiTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = CaiTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_cait_timm_to_keras import transfer_cait_weights

        transfer_cait_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        name="CaiTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = cait_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
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
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class CaiTImageClassify(BaseModel):
    """Instantiates the Class-Attention in Image Transformers (CaiT) classifier.

    This classifier wraps a :class:`CaiTModel` backbone and attaches a
    single Dense layer on the CLS token (index 0 of the backbone's
    output) to produce ``num_classes`` class logits. All architectural
    parameters are forwarded to the underlying :class:`CaiTModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, token embedding dimension. Determines model
            width: 192 (XXS), 288 (XS), 384 (S), 768 (M).
            Defaults to `192`.
        depth: Integer, number of patch-only talking-head transformer
            blocks in the backbone. Defaults to `24`.
        num_heads: Integer, number of attention heads per block (both
            patch-only and class-attention). Defaults to `4`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across the
            ``depth`` patch blocks. Defaults to `0.0`.
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
            named `f"{name}_backbone"`. Defaults to `"CaiTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = CaiTConfig
    HUB_REPO_SIBLINGS = CAIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_cait_timm_to_keras import transfer_cait_weights

        transfer_cait_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="CaiTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = CaiTModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(backbone.output[:, 0])

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
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
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
