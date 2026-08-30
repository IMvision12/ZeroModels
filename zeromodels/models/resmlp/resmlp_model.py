import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.resmlp.resmlp_layers import ResMLPAffine, ResMLPLayerScale
from zeromodels.utils import standardize_input_shape

from .resmlp_config import ResMLPConfig

# The backbone (ResMLPModel) and classifier (ResMLPImageClassify) share the variant's
# weights repo, whose zm_config.json declares ResMLPImageClassify.
RESMLP_HUB_SIBLINGS = frozenset({"ResMLPModel", "ResMLPImageClassify"})


def resmlp_block(
    x,
    embed_dim,
    seq_len,
    mlp_ratio=4,
    layer_scale_init=1e-4,
    drop_rate=0.0,
    block_idx=None,
):
    """A building block for the ResMLP architecture.

    Args:
        x: input tensor.
        embed_dim: int, dimension of the input features.
        seq_len: int, length of the input sequence for cross-patch mixing.
        mlp_ratio: float, ratio of the hidden dimension in the MLP to the input
            dimension (default: 4).
        layer_scale_init: float, initial value for layer scale parameters
            (default: 1e-4).
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """
    inputs = x

    x = ResMLPAffine(name=f"blocks_{block_idx}_affine_1")(inputs)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        seq_len,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = ResMLPLayerScale(layer_scale_init, name=f"blocks_{block_idx}_scale_1")(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = ResMLPAffine(name=f"blocks_{block_idx}_affine_2")(x)
    x = layers.Dense(
        embed_dim * mlp_ratio,
        activation="gelu",
        name=f"blocks_{block_idx}_dense_2",
    )(x)
    x = layers.Dense(
        embed_dim,
        name=f"blocks_{block_idx}_dense_3",
    )(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = ResMLPLayerScale(layer_scale_init, name=f"blocks_{block_idx}_scale_2")(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


def resmlp_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depth,
    mlp_ratio,
    layer_scale_init,
    drop_path_rate,
    data_format,
    drop_rate=0.0,
    return_stages=False,
):
    """ResMLP stem (patch embed) + ``depth`` ResMLP blocks + final ResMLPAffine.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        patch_size: Side length of each square patch.
        embed_dim: Token (channel) embedding dimension.
        depth: Number of ResMLP blocks.
        mlp_ratio: Hidden-embed_dim multiplier inside each block's channel MLP.
        layer_scale_init: Initial ResMLPLayerScale value applied at the end of each residual branch.
        drop_path_rate: Maximum stochastic-depth-style dropout rate (scaled linearly
            with block index).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-block (post-residual)
            outputs (one per ResMLP block, ``depth`` total). ResMLP is
            isotropic: shape is constant across blocks. If False (default),
            return the single post-final-ResMLPAffine sequence.

    Returns:
        Post-ResMLPAffine patch sequence of shape ``(B, num_patches, embed_dim)``,
        or a list of ``depth`` per-block outputs when ``return_stages=True``.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        height = inputs.shape[2]
        width = inputs.shape[3]
    else:
        height = inputs.shape[1]
        width = inputs.shape[2]

    num_patches = (height // patch_size) * (width // patch_size)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)
    x = layers.Reshape((num_patches, embed_dim))(x)

    stages = []
    for i in range(depth):
        drop_path = drop_rate + drop_path_rate * (i / depth)
        x = resmlp_block(
            x,
            embed_dim,
            num_patches,
            mlp_ratio,
            layer_scale_init,
            drop_path,
            block_idx=i,
        )
        stages.append(x)

    if return_stages:
        return stages

    x = ResMLPAffine(name="Final_affine")(x)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class ResMLPModel(BaseModel):
    """Instantiates the ResMLP (Residual MLP) backbone.

    ResMLP is a Mixer-style architecture where per-channel learnable
    ResMLPAffine layers replace LayerNorm and a residual structure (ResMLPAffine
    pre-norm + ResMLPLayerScale + residual add on each branch) scales to very
    deep models with stable, data-efficient training. Like Mixer it
    alternates a cross-patch linear and a channel MLP, but the
    normalization-free ResMLPAffine design and ResMLPLayerScale residuals are what
    make deep stacks trainable.

    Output is the last layer output before the classifier head: the
    final-ResMLPAffine normalized patch sequence ``(B, N, D)`` where
    ``N = (H/patch_size) * (W/patch_size)``. :class:`ResMLPImageClassify`
    composes this model and applies a GlobalAveragePooling1D + Dense
    head.

    References:
    - [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, token (channel) embedding dimension.
            Defaults to `384`.
        depth: Integer, number of ResMLP blocks. Defaults to `12`.
        mlp_ratio: Integer, hidden-embed_dim multiplier inside each block's
            channel MLP. Defaults to `4`.
        layer_scale_init: Float, initial ResMLPLayerScale value applied at the end
            of each residual branch. Defaults to `1e-4`.
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
            per-block outputs (one per ResMLP block). Defaults to `False`.
        name: String, the name of the model. Defaults to `"ResMLPModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResMLPConfig
    HUB_REPO_SIBLINGS = RESMLP_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ResMLPImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ResMLPImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resmlp_timm_to_keras import transfer_resmlp_weights

        transfer_resmlp_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        layer_scale_init=1e-4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="ResMLPModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
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

        x = img_input
        x = resmlp_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.layer_scale_init = layer_scale_init
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
                "depth": self.depth,
                "mlp_ratio": self.mlp_ratio,
                "layer_scale_init": self.layer_scale_init,
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
class ResMLPImageClassify(BaseModel):
    """Instantiates the ResMLP classifier.

    This classifier wraps a :class:`ResMLPModel` backbone and attaches a
    GlobalAveragePooling1D + Dense head on the patch sequence to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`ResMLPModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404)

    Args:
        patch_size: Integer, conv-stem patch size in pixels.
            Defaults to `16`.
        embed_dim: Integer, token (channel) embedding dimension.
            Defaults to `384`.
        depth: Integer, number of ResMLP blocks. Defaults to `12`.
        mlp_ratio: Integer, hidden-embed_dim multiplier inside each block's
            channel MLP. Defaults to `4`.
        layer_scale_init: Float, initial ResMLPLayerScale value applied at the end
            of each residual branch. Defaults to `1e-4`.
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
            named `f"{name}_backbone"`. Defaults to `"ResMLPImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ResMLPConfig
    HUB_REPO_SIBLINGS = RESMLP_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_resmlp_timm_to_keras import transfer_resmlp_weights

        transfer_resmlp_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        layer_scale_init=1e-4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResMLPImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        backbone = ResMLPModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            layer_scale_init=layer_scale_init,
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
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.layer_scale_init = layer_scale_init
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
                "depth": self.depth,
                "mlp_ratio": self.mlp_ratio,
                "layer_scale_init": self.layer_scale_init,
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
