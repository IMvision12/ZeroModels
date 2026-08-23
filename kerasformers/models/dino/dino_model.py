import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.models.resnet.resnet_model import (
    bottleneck_block,
    resnet_backbone_feature,
)
from kerasformers.models.vit.vit_model import vit_backbone_feature
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .dino_config import DinoResNetConfig, DinoViTConfig


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoViTModel(BaseModel):
    """DINO Vision Transformer model.

    Standard ViT pretrained with the DINO self-supervised method.

    When ``as_backbone=False`` (default), returns the final
    LayerNorm-normalized token sequence ``(B, num_tokens, embed_dim)`` (CLS at
    index 0). When ``as_backbone=True``, returns the list of
    intermediate feature maps from each transformer block (with the
    last LayerNorm-normalized), suitable for feeding into detection /
    segmentation / depth necks.

    Reference:
        - `Emerging Properties in Self-Supervised Vision Transformers
          <https://arxiv.org/abs/2104.14294>`_

    Args:
        as_backbone: If ``True``, output the list of per-block
            intermediate features (last LayerNorm-normalized) for use as
            a backbone. If ``False`` (default), output only the final
            LayerNorm-normalized token sequence.
        patch_size: ViT patch size (8 or 16).
        embed_dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        qkv_bias: Whether to use bias in QKV projections. Defaults to
            ``True``.
        qk_norm: Whether to use QK normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.0``.
        attn_drop_rate: Attention dropout rate. Defaults to ``0.0``.
        include_normalization: Whether to prepend
            image normalization.
        normalization_mode: Normalization preset.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DinoViTConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        include_normalization=True,
        normalization_mode="imagenet",
        image_size=224,
        input_tensor=None,
        name="DinoViTModel",
        **kwargs,
    ):
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
        features = vit_backbone_feature(
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
            no_embed_class=False,
            use_distillation=False,
            layer_scale_init=None,
            image_size=image_size,
            data_format=data_format,
            return_intermediates=True,
        )
        final_ln = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name="final_layernorm"
        )
        features[-1] = final_ln(features[-1])

        outputs = features if as_backbone else features[-1]
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

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
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.image_size = image_size
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
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoResNetModel(BaseModel):
    """DINO ResNet model.

    ResNet-50 pretrained with the DINO self-supervised method.

    When ``as_backbone=False`` (default), returns the final-stage
    feature map ``(B, H/32, W/32, C)``. When ``as_backbone=True``,
    returns the list of per-stage feature maps, suitable for feeding
    into detection / segmentation / depth necks.

    Reference:
        - `Emerging Properties in Self-Supervised Vision Transformers
          <https://arxiv.org/abs/2104.14294>`_

    Args:
        as_backbone: If ``True``, output the list of per-stage feature
            maps for use as a backbone. If ``False`` (default), output
            only the final-stage feature map.
        depths: Per-stage block counts.
        filters: Per-stage filter counts.
        include_normalization: Whether to prepend
            image normalization.
        normalization_mode: Normalization preset.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DinoResNetConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        as_backbone=False,
        depths=None,
        filters=None,
        include_normalization=True,
        normalization_mode="imagenet",
        image_size=224,
        input_tensor=None,
        name="DinoResNetModel",
        **kwargs,
    ):
        if depths is None:
            depths = [3, 4, 6, 3]
        if filters is None:
            filters = [64, 128, 256, 512]

        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = resnet_backbone_feature(
            x,
            block_fn=bottleneck_block,
            depths=depths,
            filters=filters,
            channels_axis=channels_axis,
            data_format=data_format,
            groups=1,
            senet=False,
            width_factor=1,
            return_stages=True,
        )

        outputs = features if as_backbone else features[-1]
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.depths = list(depths)
        self.filters = list(filters)
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "depths": self.depths,
                "filters": self.filters,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
