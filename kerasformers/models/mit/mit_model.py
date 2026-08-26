import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.mit.mit_layers import (
    MiTEfficientMultiheadSelfAttention,
    MiTStochasticDepth,
)
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .mit_config import MiTConfig

MIT_HUB_SIBLINGS = frozenset({"MiTModel", "MiTImageClassify"})


def mlp_block(x, H, W, channels, mid_channels, data_format, name_prefix):
    """MiT Mix-FFN: Dense -> spatial DWConv -> GELU -> Dense.

    Args:
        x: Input token tensor of shape ``(B, H*W, channels)``.
        H: Spatial height of the token grid.
        W: Spatial width of the token grid.
        channels: Output channel dimension.
        mid_channels: Hidden dimension of the first Dense (and DWConv).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name_prefix: Prefix used to name the inner layers.

    Returns:
        Tensor of shape ``(B, H*W, channels)``.
    """
    x = layers.Dense(mid_channels, name=f"{name_prefix}_dense_1")(x)
    x = layers.Reshape((H, W, mid_channels))(x)
    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 3, 1, 2))

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_dwconv",
    )(x)

    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((H * W, mid_channels))(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(channels, name=f"{name_prefix}_dense_2")(x)
    return x


def overlap_patch_embedding_block(
    x,
    channels_axis,
    data_format,
    out_channels=32,
    patch_size=7,
    stride=4,
    stage_idx=1,
):
    """Overlapping patch embedding: ZeroPad -> Conv2D(patch_size, stride) -> Reshape -> LN.

    Args:
        x: Input image/feature tensor for the current stage.
        channels_axis: Channel axis index (``-1`` for channels-last,
            ``1`` for channels-first).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        out_channels: Output channel dimension of the patch projection.
        patch_size: Conv kernel size (7 for stage 1, 3 elsewhere).
        stride: Conv stride (4 for stage 1, 2 elsewhere).
        stage_idx: 1-based stage index; mapped to the 0-based timm name
            in the layer prefixes.

    Returns:
        Tuple ``(tokens, H, W)`` where ``tokens`` has shape
        ``(B, H*W, out_channels)`` and ``H, W`` are the new spatial dims.
    """
    pytorch_stage_idx = stage_idx - 1

    x = keras.layers.ZeroPadding2D(padding=(patch_size // 2, patch_size // 2))(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=patch_size,
        strides=stride,
        padding="valid",
        data_format=data_format,
        name=f"patch_embed_{pytorch_stage_idx}_conv_proj",
    )(x)
    if data_format == "channels_first":
        H, W = x.shape[2], x.shape[3]
        x = keras.ops.transpose(x, (0, 2, 3, 1))
    else:
        H, W = x.shape[1], x.shape[2]
    x = layers.Reshape((-1, out_channels))(x)
    x = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"patch_embed_{pytorch_stage_idx}_layernorm",
    )(x)
    return x, H, W


def hierarchical_transformer_encoder_block(
    x,
    H,
    W,
    project_dim,
    num_heads,
    stage_idx,
    block_idx,
    channels_axis,
    data_format,
    qkv_bias=False,
    sr_ratio=1,
    drop_prob=0.0,
):
    """MiT block: LN -> efficient self-attn -> Add -> LN -> Mix-FFN -> Add.

    Args:
        x: Input token tensor of shape ``(B, H*W, project_dim)``.
        H: Spatial height of the token grid.
        W: Spatial width of the token grid.
        project_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        stage_idx: 1-based stage index; mapped to the 0-based timm name
            in the layer prefixes.
        block_idx: Block index within the stage.
        channels_axis: Channel axis index (``-1`` or ``1``).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        qkv_bias: Whether to include bias in the QKV projection.
        sr_ratio: Spatial reduction ratio for the key/value tokens.
        drop_prob: Stochastic-depth drop rate for each residual branch.

    Returns:
        Tensor of shape ``(B, H*W, project_dim)`` after both residual branches.
    """
    pytorch_stage_idx = stage_idx - 1
    drop_path_layer = MiTStochasticDepth(drop_prob)

    norm1 = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_1",
    )(x)

    attn_layer = MiTEfficientMultiheadSelfAttention(
        project_dim,
        sr_ratio,
        block_prefix=f"block_{pytorch_stage_idx}_{block_idx}",
        qkv_bias=qkv_bias,
        num_heads=num_heads,
    )

    attn_out = attn_layer(norm1)
    attn_out = drop_path_layer(attn_out)
    add1 = layers.Add()([x, attn_out])

    norm2 = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_2",
    )(add1)

    mlp_out = mlp_block(
        norm2,
        H,
        W,
        channels=project_dim,
        mid_channels=int(project_dim * 4),
        data_format=data_format,
        name_prefix=f"block_{pytorch_stage_idx}_{block_idx}_mlp",
    )

    mlp_out = drop_path_layer(mlp_out)
    return layers.Add()([add1, mlp_out])


def mit_backbone_feature(
    inputs,
    *,
    embed_dim,
    depths,
    drop_path_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """MiT 4-stage hierarchical transformer encoder (SegFormer backbone).

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        embed_dim: 4-tuple of per-stage embedding dimensions.
        depths: 4-tuple of per-stage block counts.
        drop_path_rate: Maximum stochastic-depth drop rate (linearly scaled
            across all blocks in the network).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index (``-1`` or ``1``).
        return_stages: If ``True``, return the full list of four per-stage
            spatial feature maps. Otherwise return only the final stage.

    Returns:
        By default, the final stage's spatial feature map of shape
        ``(B, H_4, W_4, embed_dim[-1])``. When ``return_stages=True``,
        returns the list of four per-stage feature maps.
    """
    num_stages = 4
    blockwise_num_heads = [1, 2, 5, 8]
    blockwise_sr_ratios = [8, 4, 2, 1]

    total_blocks = sum(depths)
    dpr = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

    x = inputs
    features = []
    cur_block = 0

    for i in range(num_stages):
        x, H, W = overlap_patch_embedding_block(
            x,
            out_channels=embed_dim[i],
            channels_axis=channels_axis,
            data_format=data_format,
            patch_size=7 if i == 0 else 3,
            stride=4 if i == 0 else 2,
            stage_idx=i + 1,
        )

        for j in range(depths[i]):
            x = hierarchical_transformer_encoder_block(
                x,
                H,
                W,
                project_dim=embed_dim[i],
                num_heads=blockwise_num_heads[i],
                stage_idx=i + 1,
                block_idx=j,
                sr_ratio=blockwise_sr_ratios[i],
                drop_prob=dpr[cur_block],
                qkv_bias=True,
                channels_axis=channels_axis,
                data_format=data_format,
            )
            cur_block += 1

        x = layers.LayerNormalization(
            name=f"final_layernorm_{i}", axis=-1, epsilon=1e-5
        )(x)
        x = layers.Reshape((H, W, embed_dim[i]))(x)
        if data_format == "channels_first":
            x = keras.ops.transpose(x, (0, 3, 1, 2))
        features.append(x)

    if return_stages:
        return features
    return features[-1]


@keras.saving.register_keras_serializable(package="kerasformers")
class MiTModel(BaseModel):
    """Instantiates the Mix Transformer (MiT) backbone: the SegFormer encoder.

    MiT is a hierarchical transformer backbone built from four stages.
    Each stage begins with an overlapping patch embedding (a strided
    convolution with kernel ``> stride``) that downsamples and re-tokenizes
    the feature map, followed by a stack of transformer blocks that use
    efficient self-attention with spatial reduction of the key/value
    tokens (large reduction ratio in early stages, none in the final
    stage) and a Mix-FFN feed-forward network that injects a 3x3 depthwise
    convolution between the two Dense layers to encode positional
    information without explicit positional embeddings. The four stages
    produce a multi-scale feature pyramid used by SegFormer.

    Output is the last layer output before the classifier head: by default
    the final stage's spatial feature map of shape
    ``(B, H_4, W_4, embed_dim[-1])`` (or channels-first equivalent).
    When constructed with ``as_backbone=True``, returns the list of all
    four per-stage feature maps instead. :class:`MiTImageClassify` composes
    this model with ``as_backbone=False`` and reads the final feature map
    through a ``GlobalAveragePooling2D`` + Dense head.

    References:
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of four
            per-stage spatial feature maps (the SegFormer feature pyramid).
            Defaults to `False`.
        embed_dim: Tuple of four integers, per-stage channel dimensions.
            Determines model width. Defaults to `(32, 64, 160, 256)`.
        depths: Tuple of four integers, per-stage number of transformer
            blocks. Defaults to `(2, 2, 2, 2)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across all
            blocks in the network. Defaults to `0.1`.
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
        name: String, the name of the model. Defaults to `"MiTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MiTConfig
    HUB_REPO_SIBLINGS = MIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MiTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MiTImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dim": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mit_hf_to_keras import transfer_mit_weights

        transfer_mit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        embed_dim=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        name="MiTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "hf_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = mit_backbone_feature(
            x,
            embed_dim=embed_dim,
            depths=depths,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.embed_dim = list(embed_dim)
        self.depths = list(depths)
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
                "embed_dim": self.embed_dim,
                "depths": self.depths,
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
class MiTImageClassify(BaseModel):
    """Instantiates the Mix Transformer (MiT) classifier.

    This classifier wraps a :class:`MiTModel` backbone and attaches a
    ``GlobalAveragePooling2D`` followed by a single Dense layer on the
    final stage feature map to produce ``num_classes`` class logits. All
    architectural parameters are forwarded to the underlying
    :class:`MiTModel`; only ``num_classes`` and ``classifier_activation``
    are head-specific.

    References:
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

    Args:
        embed_dim: Tuple of four integers, per-stage channel dimensions.
            Determines model width. Defaults to `(32, 64, 160, 256)`.
        depths: Tuple of four integers, per-stage number of transformer
            blocks. Defaults to `(2, 2, 2, 2)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across all
            blocks in the network. Defaults to `0.1`.
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
            named `f"{name}_backbone"`. Defaults to `"MiTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MiTConfig
    HUB_REPO_SIBLINGS = MIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dim": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_classes": hf_config.get("num_labels", 1000),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mit_hf_to_keras import transfer_mit_weights

        transfer_mit_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dim=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MiTImageClassify",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)

        data_format = keras.config.image_data_format()

        backbone = MiTModel(
            embed_dim=embed_dim,
            depths=depths,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.embed_dim = list(embed_dim)
        self.depths = list(depths)
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
                "embed_dim": self.embed_dim,
                "depths": self.depths,
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
