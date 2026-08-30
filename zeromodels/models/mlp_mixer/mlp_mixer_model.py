import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .mlp_mixer_config import MLPMixerConfig

# The backbone (MLPMixerModel) and classifier (MLPMixerImageClassify) share the
# variant's repo, whose zm_config.json declares MLPMixerImageClassify.
MLP_MIXER_HUB_SIBLINGS = frozenset({"MLPMixerModel", "MLPMixerImageClassify"})


def mixer_block(
    x,
    patches,
    filters,
    token_mlp_dim,
    channel_mlp_dim,
    channels_axis,
    drop_rate=0.0,
    block_idx=None,
):
    """A building block for the MLP-Mixer architecture.

    Args:
        x: input tensor.
        patches: int, the number of patches (sequence length) for token mixing.
        filters: int, the number of output filters for channel mixing.
        token_mlp_dim: int, hidden dimension for token mixing MLP.
        channel_mlp_dim: int, hidden dimension for channel mixing MLP.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """

    inputs = x

    x = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_1"
    )(x)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        token_mlp_dim,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_1")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = layers.Dense(
        patches, name=f"blocks_{block_idx}_dense_2", kernel_initializer="glorot_uniform"
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    x = layers.Dense(
        channel_mlp_dim,
        name=f"blocks_{block_idx}_dense_3",
        kernel_initializer="glorot_uniform",
    )(x)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_2")(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = layers.Dense(
        filters, name=f"blocks_{block_idx}_dense_4", kernel_initializer="glorot_uniform"
    )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


def mlp_mixer_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depths,
    mlp_ratio,
    drop_path_rate,
    data_format,
    channels_axis,
    drop_rate=0.0,
    return_stages=False,
):
    """MLP-Mixer stem (patch embed) + N mixer blocks + final LayerNorm.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        patch_size: Side length of each square patch.
        embed_dim: Per-patch embedding (channel) dimension.
        depths: Number of mixer blocks.
        mlp_ratio: Pair ``(token_mlp, channel_mlp)`` of ratios scaling
            ``embed_dim`` to the two hidden dims.
        drop_path_rate: Maximum stochastic-depth-style dropout rate (scaled linearly
            with block index).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis of the channel dimension.
        return_stages: If True, return a list of per-block (post-residual)
            features (one per mixer block, ``depths`` total). Spatial shape
            is constant across blocks (Mixer is isotropic). If False (default),
            return the single post-final-LayerNorm sequence.

    Returns:
        Post-LayerNorm patch sequence of shape ``(B, num_patches, embed_dim)``,
        or a list of ``depths`` per-block outputs when ``return_stages=True``.
    """
    if data_format == "channels_first":
        height, width = inputs.shape[2], inputs.shape[3]
    else:
        height, width = inputs.shape[1], inputs.shape[2]

    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)

    num_patches = (height // patch_size) * (width // patch_size)
    x = layers.Reshape((num_patches, embed_dim))(x)

    token_mlp_dim = int(embed_dim * mlp_ratio[0])
    channel_mlp_dim = int(embed_dim * mlp_ratio[1])

    stages = []
    for i in range(depths):
        drop_path = drop_rate + drop_path_rate * (i / depths)
        x = mixer_block(
            x,
            num_patches,
            embed_dim,
            token_mlp_dim,
            channel_mlp_dim,
            channels_axis,
            drop_rate=drop_path,
            block_idx=i,
        )
        stages.append(x)

    if return_stages:
        return stages

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MLPMixerModel(BaseModel):
    """Instantiates the MLP-Mixer backbone.

    MLP-Mixer is an all-MLP architecture for vision: after a patch
    embedding stem, patches go through alternating "token-mixing" MLPs
    (acting across spatial positions) and "channel-mixing" MLPs (acting
    across channels). It contains no self-attention and no convolutions
    beyond the patch-embed stem, demonstrating that competitive image
    classification can be built from MLPs alone.

    Output is the last layer output before the classifier head: the
    final-LN normalized patch sequence ``(B, N, D)`` where
    ``N = (H/patch_size) * (W/patch_size)``. :class:`MLPMixerImageClassify`
    composes this model and applies a GlobalAveragePooling1D + Dense
    head (mean-pool over tokens).

    References:
    - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, per-patch embedding (channel) dimension.
            Defaults to `768`.
        depths: Integer, number of mixer blocks.
            Defaults to `12`.
        mlp_ratio: Tuple of two floats, ``(token_mlp, channel_mlp)``
            hidden-dim ratios applied to ``embed_dim``.
            Defaults to `(0.5, 4.0)`.
        drop_rate: Float, dropout rate. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth-style dropout
            rate (scaled linearly with block index). Defaults to `0.0`.
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
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-block outputs (one per mixer block). Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"MLPMixerModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MLPMixerConfig
    HUB_REPO_SIBLINGS = MLP_MIXER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with MLPMixerImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = MLPMixerImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mlpmixer_timm_to_keras import transfer_mlp_mixer_weights

        transfer_mlp_mixer_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depths=12,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="MLPMixerModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        for k in ("num_classes", "classifier_activation", "timm_id"):
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

        x = img_input
        x = mlp_mixer_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class MLPMixerImageClassify(BaseModel):
    """Instantiates the MLP-Mixer classifier.

    This classifier wraps an :class:`MLPMixerModel` backbone and
    attaches a GlobalAveragePooling1D + Dense head (mean-pool over
    tokens) on the patch sequence to produce ``num_classes`` class
    logits. All architectural parameters are forwarded to the
    underlying :class:`MLPMixerModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, per-patch embedding (channel) dimension.
            Defaults to `768`.
        depths: Integer, number of mixer blocks.
            Defaults to `12`.
        mlp_ratio: Tuple of two floats, ``(token_mlp, channel_mlp)``
            hidden-dim ratios applied to ``embed_dim``.
            Defaults to `(0.5, 4.0)`.
        drop_rate: Float, dropout rate. Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth-style dropout
            rate (scaled linearly with block index). Defaults to `0.0`.
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
            named `f"{name}_backbone"`. Defaults to `"MLPMixerImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MLPMixerConfig
    HUB_REPO_SIBLINGS = MLP_MIXER_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mlpmixer_timm_to_keras import transfer_mlp_mixer_weights

        transfer_mlp_mixer_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depths=12,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MLPMixerImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        backbone = MLPMixerModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = ops.mean(backbone.output, axis=1)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
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
