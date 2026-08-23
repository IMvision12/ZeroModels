import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.vit.vit_layers import (
    ViTAddPositionEmbs,
    ViTClassDistToken,
    ViTMultiHeadSelfAttention,
)
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .pit_config import PiTConfig

# The backbone (PiTModel) and classifier (PiTImageClassify) share the variant's
# weights repo, whose kf_config.json declares PiTImageClassify.
PIT_HUB_SIBLINGS = frozenset({"PiTModel", "PiTImageClassify"})


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_prefix=None):
    """Standard transformer MLP block: Dense -> GELU -> Drop -> Dense -> Drop.

    Args:
        inputs: Input token tensor of shape ``(B, N, D)``.
        hidden_features: Hidden expansion dimension of the first Dense.
        out_features: Output dimension of the second Dense.
        drop: Dropout rate applied after each Dense.
        block_prefix: Prefix used to name the inner Dense layers.

    Returns:
        Tensor of shape ``(B, N, out_features)``.
    """
    x = layers.Dense(hidden_features, use_bias=True, name=block_prefix + "_dense_1")(
        inputs
    )
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(out_features, use_bias=True, name=block_prefix + "_dense_2")(x)
    x = layers.Dropout(drop)(x)
    return x


def transformer_block(inputs, embed_dim, num_heads, mlp_ratio, block_prefix=None):
    """PiT transformer block: LN -> MHSA -> Add -> LN -> MLP -> Add.

    Args:
        inputs: Input token tensor of shape ``(B, N, embed_dim)``.
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP sub-block.
        block_prefix: Prefix used to name layers inside the block.

    Returns:
        Tensor of shape ``(B, N, embed_dim)`` after both residual branches.
    """
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=block_prefix + "_layernorm_1"
    )(inputs)
    x = ViTMultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=block_prefix.replace("pit", "transformers"),
    )(x)
    x = layers.Add()([inputs, x])

    y = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=block_prefix + "_layernorm_2"
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(embed_dim * mlp_ratio),
        out_features=embed_dim,
        block_prefix=block_prefix,
    )
    return layers.Add()([x, y])


def conv_pooling(
    x, nb_tokens, in_channels, out_channels, stride, data_format, block_prefix
):
    """Depthwise-conv downsampling for spatial tokens + Dense projection for class tokens.

    Args:
        x: ``(tokens, (height, width))`` pair where ``tokens`` has shape
            ``(B, nb_tokens + height*width, in_channels)`` and ``(height,
            width)`` is the spatial grid for the patch tokens.
        nb_tokens: Number of class/distillation prefix tokens (1 or 2).
        in_channels: Channel dimension of the input tokens.
        out_channels: Channel dimension after projection.
        stride: Spatial stride for the depthwise conv downsample.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        block_prefix: Prefix used to name the conv and dense layers.

    Returns:
        Pair ``(output, (new_height, new_width))`` where ``output`` has
        shape ``(B, nb_tokens + new_height*new_width, out_channels)``.
    """
    input_tensor, (height, width) = x
    tokens = input_tensor[:, :nb_tokens]
    spatial = input_tensor[:, nb_tokens:]

    new_height = (height + stride - 1) // stride
    new_width = (width + stride - 1) // stride

    spatial = layers.Reshape((height, width, in_channels))(spatial)
    if data_format == "channels_first":
        spatial = layers.Permute((3, 1, 2))(spatial)
    spatial = layers.ZeroPadding2D(data_format=data_format, padding=stride // 2)(
        spatial
    )
    spatial = layers.Conv2D(
        filters=out_channels,
        kernel_size=stride + 1,
        strides=stride,
        groups=in_channels,
        data_format=data_format,
        name=block_prefix + "_conv",
    )(spatial)

    tokens = layers.Dense(units=out_channels, name=block_prefix + "_dense")(tokens)
    if data_format == "channels_first":
        spatial = layers.Permute((2, 3, 1))(spatial)
    spatial = layers.Reshape((new_height * new_width, out_channels))(spatial)
    output = layers.Concatenate(axis=1)([tokens, spatial])
    return output, (new_height, new_width)


def pit_backbone_feature(
    inputs,
    *,
    patch_size,
    stride,
    embed_dim,
    depth,
    heads,
    mlp_ratio,
    distilled,
    drop_rate,
    data_format,
    return_stages=False,
    return_final_spatial=False,
):
    """PiT stem + pooling-attention stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        patch_size: Conv-stem kernel size in pixels.
        stride: Conv-stem stride in pixels.
        embed_dim: Per-stage embedding dimensions (one int per stage).
        depth: Per-stage number of transformer blocks.
        heads: Per-stage number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP sub-block.
        distilled: If ``True``, also use a distillation token in addition to
            the class token.
        drop_rate: Dropout rate applied after the position embedding.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If ``True``, return a list of intermediate stage
            outputs as described below.
        return_final_spatial: If ``True``, return the final stage's spatial
            feature map instead of the class/dist tokens.

    Returns:
        By default, the LN-normalized class/dist tokens of shape
        ``(B, 1 or 2, embed_dim[-1])``. If ``return_stages=True``, returns
        ``[stem, stage1, ..., stageN, cls_dist_norm]``. If
        ``return_final_spatial=True``, returns the final stage's spatial
        feature map of shape ``(B, H, W, C)`` (or channels-first equivalent).
    """
    if data_format == "channels_first":
        _, height, width = inputs.shape[1:]
    else:
        height, width, _ = inputs.shape[1:]

    x = layers.Conv2D(
        filters=embed_dim[0],
        kernel_size=patch_size,
        strides=stride,
        data_format=data_format,
        name="patch_embed_conv",
    )(inputs)

    grid_h = (height - patch_size) // stride + 1
    grid_w = (width - patch_size) // stride + 1
    input_size = (grid_h, grid_w)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name="patch_to_nhwc")(x)
    x = layers.Reshape((grid_h * grid_w, embed_dim[0]), name="patch_tokens_reshape")(x)

    x = ViTAddPositionEmbs(
        grid_h=grid_h,
        grid_w=grid_w,
        no_embed_class=True,
        use_distillation=distilled,
        name="pos_embed",
    )(x)
    x = ViTClassDistToken(
        use_distillation=distilled,
        combine_tokens=True,
        name="class_dist_token",
    )(x)

    stages = [x]
    x = layers.Dropout(drop_rate, name="pos_drop")(x)

    for stage_idx in range(len(depth)):
        for block_idx in range(depth[stage_idx]):
            x = transformer_block(
                x,
                embed_dim=embed_dim[stage_idx],
                num_heads=heads[stage_idx],
                mlp_ratio=mlp_ratio,
                block_prefix=f"pit_{stage_idx}_blocks_{block_idx}",
            )
        if stage_idx < len(depth) - 1:
            x, input_size = conv_pooling(
                (x, input_size),
                nb_tokens=2 if distilled else 1,
                in_channels=embed_dim[stage_idx],
                out_channels=embed_dim[stage_idx + 1],
                stride=2,
                data_format=data_format,
                block_prefix=f"pit_{stage_idx + 1}_pool",
            )
        stages.append(x)

    cls_dist = x[:, : 2 if distilled else 1]
    cls_dist = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="norm")(cls_dist)

    if return_final_spatial:
        nb_tokens = 2 if distilled else 1
        final_channels = embed_dim[-1]
        spatial = x[:, nb_tokens:]
        spatial = layers.Reshape(
            (input_size[0], input_size[1], final_channels),
            name="final_spatial_reshape",
        )(spatial)
        if data_format == "channels_first":
            spatial = layers.Permute((3, 1, 2), name="final_spatial_to_cf")(spatial)
        return spatial

    if return_stages:
        stages.append(cls_dist)
        return stages
    return cls_dist


@keras.saving.register_keras_serializable(package="kerasformers")
class PiTModel(BaseModel):
    """Instantiates the Pooling-based Vision Transformer (PiT) backbone.

    PiT is a hierarchical Vision Transformer that progressively shrinks
    the spatial token grid via depthwise-conv pooling layers placed
    between transformer stages, while expanding the channel dimension:
    analogous to the spatial-reduction / channel-expansion pattern of
    classical CNN backbones. The class (and optional distillation) tokens
    are pooled in parallel with the patch tokens via a Dense projection,
    so the prefix tokens stay shape-compatible across stages.

    Output is the last layer output before the classifier head: the
    final LN-normalized class (and distillation) tokens of shape
    ``(B, 1 or 2, embed_dim[-1])``. :class:`PiTImageClassify` composes this
    model and reads the class token via ``backbone.output[:, 0]`` (and
    ``[:, 1]`` for the distillation token) to produce logits.

    References:
    - [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (post-pos-embed, after each stage, and
            the final LN-normalized class/dist tokens). Defaults to
            `False`.
        patch_size: Integer, conv-stem kernel size in pixels.
            Defaults to `16`.
        stride: Integer, conv-stem stride in pixels. Defaults to `8`.
        embed_dim: Tuple of integers, per-stage token embedding
            dimensions (one entry per stage). Defaults to
            `(64, 128, 256)`.
        depth: Tuple of integers, per-stage number of transformer blocks.
            Defaults to `(2, 6, 4)`.
        heads: Tuple of integers, per-stage number of attention heads.
            Defaults to `(2, 4, 8)`.
        mlp_ratio: Float, hidden expansion ratio for the MLP sub-block
            inside every transformer block. Defaults to `4.0`.
        distilled: Boolean, if `True`, also use a distillation token in
            addition to the class token (DeiT-distilled style).
            Defaults to `False`.
        drop_rate: Float, dropout rate applied after the position
            embedding. Defaults to `0.0`.
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
        name: String, the name of the model. Defaults to `"PiTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PiTConfig
    HUB_REPO_SIBLINGS = PIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with PiTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = PiTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_pit_timm_to_keras import transfer_pit_weights

        transfer_pit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        name="PiTModel",
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
        x = pit_backbone_feature(
            x,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            distilled=distilled,
            drop_rate=drop_rate,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
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
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
class PiTImageClassify(BaseModel):
    """Instantiates the Pooling-based Vision Transformer (PiT) classifier.

    This classifier wraps a :class:`PiTModel` backbone and attaches a
    single Dense layer on the CLS token (index 0 of the backbone's
    output) to produce ``num_classes`` class logits. When
    ``distilled=True``, a second Dense head is attached to the
    distillation token (index 1) and the two head outputs are averaged
    before the classifier activation. All architectural parameters are
    forwarded to the underlying :class:`PiTModel`; only ``num_classes``
    and ``classifier_activation`` are head-specific.

    References:
    - [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)

    Args:
        patch_size: Integer, conv-stem kernel size in pixels.
            Defaults to `16`.
        stride: Integer, conv-stem stride in pixels. Defaults to `8`.
        embed_dim: Tuple of integers, per-stage token embedding
            dimensions (one entry per stage). Defaults to
            `(64, 128, 256)`.
        depth: Tuple of integers, per-stage number of transformer blocks.
            Defaults to `(2, 6, 4)`.
        heads: Tuple of integers, per-stage number of attention heads.
            Defaults to `(2, 4, 8)`.
        mlp_ratio: Float, hidden expansion ratio for the MLP sub-block
            inside every transformer block. Defaults to `4.0`.
        distilled: Boolean, if `True`, also use a distillation token in
            addition to the class token and attach a second prediction
            head whose output is averaged with the CLS head. Defaults to
            `False`.
        drop_rate: Float, dropout rate applied after the position
            embedding and before the classifier head. Defaults to `0.0`.
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
            named `f"{name}_backbone"`. Defaults to `"PiTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PiTConfig
    HUB_REPO_SIBLINGS = PIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_pit_timm_to_keras import transfer_pit_weights

        transfer_pit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PiTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = PiTModel(
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            distilled=distilled,
            drop_rate=drop_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        cls_dist = backbone.output
        if distilled:
            cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(
                cls_dist
            )
            dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(
                cls_dist
            )
            cls_token = layers.Dropout(drop_rate)(cls_token)
            dist_token = layers.Dropout(drop_rate)(dist_token)
            cls_head = layers.Dense(num_classes, name="predictions")(cls_token)
            dist_head = layers.Dense(num_classes, name="predictions_dist")(dist_token)
            out = layers.Average()([cls_head, dist_head])
            if classifier_activation is not None:
                out = layers.Activation(
                    classifier_activation, name="predictions_activation"
                )(out)
        else:
            tok = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(cls_dist)
            tok = layers.Dropout(drop_rate)(tok)
            out = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(tok)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
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
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
