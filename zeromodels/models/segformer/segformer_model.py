import keras
from keras import layers

from zeromodels.base import BaseModel
from zeromodels.base.base_model import hf_num_classes
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.mit.mit_model import MiTModel
from zeromodels.utils import standardize_input_shape

from .segformer_config import SegformerConfig


def segformer_head(
    features,
    decode_head_dim=256,
    num_classes=19,
    dropout_rate=0.1,
    name="head",
):
    """All-MLP SegFormer decode head.

    Projects each backbone stage to ``decode_head_dim`` channels with
    a per-stage ``Dense``, bilinearly resamples to the highest-
    resolution stage, concatenates, fuses through a 1x1 conv + BN +
    ReLU + Dropout, and classifies through a 1x1 conv to
    ``num_classes`` channels.

    Reference:
        - `SegFormer: Simple and Efficient Design for Semantic
          Segmentation with Transformers <https://arxiv.org/abs/2105.15203>`_

    Args:
        features: List of four multi-scale feature maps from the MiT
            backbone (highest resolution first).
        decode_head_dim: Channel width of the per-stage projection and
            the fused representation. ``256`` for B0/B1, ``768`` for
            B2-B5.
        num_classes: Number of output classes.
        dropout_rate: Dropout applied before the final classifier.
        name: Name prefix for all layers in the head.

    Returns:
        Logits tensor at the highest backbone stage resolution
        (``H / 4, W / 4``).
    """
    data_format = keras.config.image_data_format()
    channels_axis = 1 if data_format == "channels_first" else -1
    if data_format == "channels_first":
        target_height = features[0].shape[2]
        target_width = features[0].shape[3]
    else:
        target_height = features[0].shape[1]
        target_width = features[0].shape[2]

    projected_features = []
    for i, feature in enumerate(features):
        if data_format == "channels_first":
            feature = keras.ops.transpose(feature, (0, 2, 3, 1))
        x = layers.Dense(decode_head_dim, name=f"{name}_linear_c{i + 1}")(feature)
        if data_format == "channels_first":
            x = keras.ops.transpose(x, (0, 3, 1, 2))

        x = layers.Resizing(
            height=target_height,
            width=target_width,
            interpolation="bilinear",
            data_format=data_format,
            name=f"{name}_resize_c{i + 1}",
        )(x)
        projected_features.append(x)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")(
        projected_features[::-1]
    )

    x = layers.Conv2D(
        filters=decode_head_dim,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_fusion_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.9, name=f"{name}_fusion_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_fusion_relu")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)

    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        data_format=data_format,
        name=f"{name}_classifier",
    )(x)

    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class SegFormerModel(BaseModel):
    """SegFormer hierarchical Transformer backbone (no decode head).

    Wraps the MiT (Mix Transformer) backbone in ``as_backbone=True``
    mode and exposes its four multi-scale feature maps as a list
    output. Use this when you want SegFormer's hierarchical features
    to feed into a custom head; use :class:`SegFormerSemanticSegment` for the
    full semantic-segmentation model.

    Reference:
        - `SegFormer: Simple and Efficient Design for Semantic
          Segmentation with Transformers <https://arxiv.org/abs/2105.15203>`_

    Args:
        embed_dim: Per-stage hidden dimensions for the four MiT
            stages.
        depths: Per-stage transformer-block counts.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `512`.
        input_tensor: Optional pre-existing Keras tensor to use as the
            model input.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dim": list(hf_config["hidden_sizes"]),
            "depths": list(hf_config["depths"]),
        }

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = SegFormerSemanticSegment.from_hf(hf_id, skip_mismatch=skip_mismatch)
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model

    def __init__(
        self,
        embed_dim=None,
        depths=None,
        image_size=512,
        input_tensor=None,
        name="SegFormerModel",
        **kwargs,
    ):
        if embed_dim is None:
            embed_dim = [32, 64, 160, 256]
        if depths is None:
            depths = [2, 2, 2, 2]

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        backbone = MiTModel(
            embed_dim=embed_dim,
            depths=depths,
            image_size=image_size,
            input_tensor=input_tensor,
            as_backbone=True,
            name=f"{name}_backbone",
        )

        super().__init__(
            inputs=backbone.input, outputs=backbone.output, name=name, **kwargs
        )

        self.backbone = backbone
        self.embed_dim = list(embed_dim)
        self.depths = list(depths)
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class SegFormerSemanticSegment(BaseModel):
    """SegFormer full semantic segmentation model (backbone + decode head).

    Composes :class:`SegFormerModel` and adds the all-MLP decode head,
    a classifier, and a bilinear upsample back to the input
    resolution. Output shape is
    ``(B, H, W, num_classes)`` in ``channels_last`` (and
    ``(B, num_classes, H, W)`` in ``channels_first``).

    Reference:
        - `SegFormer: Simple and Efficient Design for Semantic
          Segmentation with Transformers <https://arxiv.org/abs/2105.15203>`_

    Args:
        embed_dim: Per-stage hidden dimensions for the four MiT
            stages.
        depths: Per-stage transformer-block counts.
        decode_head_dim: Channel width of the decode-head projection
            and fused representation. ``256`` for B0/B1, ``768`` for
            B2-B5.
        dropout_rate: Dropout applied before the final classifier.
        num_classes: Number of segmentation classes.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `512`.
        input_tensor: Optional pre-existing Keras tensor to use as the
            model input.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    config_class = SegformerConfig
    # Weights load by Hub repo id, e.g. from_weights("zeromodels/segformer_b0_ade_512"),
    # via zm_config.json on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def config_from_hf(cls, hf_config):
        # SegformerConfig.image_size is a vestigial 224 default unrelated to the
        # checkpoint's training resolution: keep the class default (override
        # with an explicit image_size kwarg if needed).
        return {
            "embed_dim": list(hf_config["hidden_sizes"]),
            "depths": list(hf_config["depths"]),
            "decode_head_dim": hf_config["decoder_hidden_size"],
            "num_classes": hf_num_classes(hf_config),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.segformer.convert_segformer_hf_to_keras import (
            transfer_segformer_weights,
        )

        transfer_segformer_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=None,
        depths=None,
        decode_head_dim=256,
        dropout_rate=0.1,
        num_classes=19,
        image_size=512,
        input_tensor=None,
        name="SegFormerSemanticSegment",
        **kwargs,
    ):
        if embed_dim is None:
            embed_dim = [32, 64, 160, 256]
        if depths is None:
            depths = [2, 2, 2, 2]

        base = SegFormerModel(
            embed_dim=embed_dim,
            depths=depths,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )

        x = segformer_head(
            base.output,
            decode_head_dim=decode_head_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            name="head",
        )

        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            upsample_h, upsample_w = (
                base.image_size[1],
                base.image_size[2],
            )
        else:
            upsample_h, upsample_w = (
                base.image_size[0],
                base.image_size[1],
            )
        x = layers.Resizing(
            height=upsample_h,
            width=upsample_w,
            interpolation="bilinear",
            data_format=data_format,
            name="final_upsampling",
        )(x)

        super().__init__(inputs=base.input, outputs=x, name=name, **kwargs)

        self.backbone = base.backbone
        self.embed_dim = list(embed_dim)
        self.depths = list(depths)
        self.decode_head_dim = decode_head_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "decode_head_dim": self.decode_head_dim,
                "dropout_rate": self.dropout_rate,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
