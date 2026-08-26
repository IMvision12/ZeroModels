import keras
from keras import layers

from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.convnext.convnext_model import (
    ConvNeXtImageClassify,
    ConvNeXtModel,
)

from .convnextv2_config import ConvNeXtV2Config

# The backbone (ConvNeXtV2Model) and classifier (ConvNeXtV2ImageClassify) share the
# variant's weights repo, whose zm_config.json declares ConvNeXtV2ImageClassify.
CONVNEXTV2_HUB_SIBLINGS = frozenset({"ConvNeXtV2Model", "ConvNeXtV2ImageClassify"})


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtV2Model(ConvNeXtModel):
    """Instantiates the ConvNeXtV2 backbone.

    ConvNeXtV2 augments :class:`ConvNeXtModel` with Global Response
    Normalization (GRN) inside each block and is pre-trained with the
    Fully Convolutional Masked Autoencoder (FCMAE) recipe before
    supervised fine-tuning. Output is the last layer output before the
    classifier head: the final stage feature map ``(B, H, W, C)``.
    :class:`ConvNeXtV2ImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + LayerNorm + Dense head to produce logits.

    References:
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one per ConvNeXt stage).
            Defaults to `False`.
        name: String, the name of the model.
            Defaults to `"ConvNeXtV2Model"`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`ConvNeXtModel` (e.g. ``depths``, ``projection_dim``,
            ``use_grn``, ``image_size``, ``include_normalization``).

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvNeXtV2Config
    HUB_REPO_SIBLINGS = CONVNEXTV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with ConvNeXtV2ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ConvNeXtV2ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from zeromodels.models.convnext.convert_convnext_timm_to_keras import (
            transfer_convnext_weights,
        )

        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, as_backbone=False, name="ConvNeXtV2Model", **kwargs):
        super().__init__(as_backbone=as_backbone, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtV2ImageClassify(ConvNeXtImageClassify):
    """Instantiates the ConvNeXtV2 classifier.

    This classifier wraps a :class:`ConvNeXtV2Model` backbone and
    attaches a GlobalAveragePooling2D + LayerNorm + Dense head to
    produce ``num_classes`` class logits. All architectural parameters
    are forwarded to the underlying :class:`ConvNeXtV2Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        depths: Tuple of 4 integers, number of ConvNeXt blocks per stage.
            Defaults to `(3, 3, 9, 3)`.
        projection_dim: Tuple of 4 integers, channel count per stage.
            Defaults to `(96, 192, 384, 768)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly scaled from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init: Float, initial value for per-channel
            LayerScale. Pass ``None`` to disable LayerScale.
            Defaults to `1e-6`.
        use_conv: Boolean, if True, use 1x1 Conv2D layers inside each
            block's MLP; otherwise use Dense layers. Defaults to `False`.
        use_grn: Boolean, whether to apply ConvNeXtGlobalResponseNorm inside each
            block (ConvNeXtV2 recipe). Defaults to `False`.
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
            named `f"{name}_backbone"`. Defaults to `"ConvNeXtV2ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = ConvNeXtV2Config
    HUB_REPO_SIBLINGS = CONVNEXTV2_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from zeromodels.models.convnext.convert_convnext_timm_to_keras import (
            transfer_convnext_weights,
        )

        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dim=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvNeXtV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvNeXtV2Model(
            depths=depths,
            projection_dim=projection_dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            use_conv=use_conv,
            use_grn=use_grn,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super(ConvNeXtImageClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.depths = list(depths)
        self.projection_dim = list(projection_dim)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.use_conv = use_conv
        self.use_grn = use_grn
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
