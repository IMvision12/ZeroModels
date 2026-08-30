import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.models.vit.vit_model import vit_backbone_feature
from zeromodels.utils import standardize_input_shape

from .dino_v2_config import DinoV2Config


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoV2Model(BaseModel):
    """DINOv2 Vision Transformer model.

    Standard ViT with LayerScale pretrained with the DINOv2
    self-supervised method.

    When ``as_backbone=False`` (default), returns the final
    LayerNorm-normalized token sequence ``(B, num_tokens, embed_dim)`` (CLS at
    index 0). When ``as_backbone=True``, returns the list of
    intermediate feature maps from each transformer block (with the
    last LayerNorm-normalized), suitable for feeding into detection /
    segmentation / depth necks (e.g. Depth Anything, OwlViT).

    Reference:
        - `DINOv2: Learning Robust Visual Features without Supervision
          <https://arxiv.org/abs/2304.07193>`_

    Args:
        as_backbone: If ``True``, output the list of per-block
            intermediate features (last LayerNorm-normalized) for use as
            a backbone. If ``False`` (default), output only the final
            LayerNorm-normalized token sequence.
        patch_size: ViT patch size. DINOv2 uses 14.
        embed_dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        qkv_bias: Whether to use bias in QKV projections. Defaults to
            ``True``.
        qk_norm: Whether to use QK normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.0``.
        attn_drop_rate: Attention dropout rate. Defaults to ``0.0``.
        layer_scale_init: LayerScale init value. Defaults to ``1.0``.
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
    config_class = DinoV2Config
    HF_MODEL_TYPE = "dinov2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "patch_size": hf_config.get("patch_size", 14),
            "embed_dim": hf_config["hidden_size"],
            "depth": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_ratio": hf_config.get("mlp_ratio", 4.0),
            "layer_scale_init": hf_config.get("layerscale_value", 1.0),
            "use_swiglu": hf_config.get("use_swiglu_ffn", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.dino_v2.convert_dino_v2_hf_to_keras import (
            transfer_dinov2_weights,
        )

        transfer_dinov2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale_init=1.0,
        use_swiglu=False,
        image_size=224,
        input_tensor=None,
        name="DinoV2Model",
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

        x = img_input
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
            layer_scale_init=layer_scale_init,
            image_size=image_size,
            data_format=data_format,
            return_intermediates=True,
            use_swiglu=use_swiglu,
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
        self.layer_scale_init = layer_scale_init
        self.use_swiglu = use_swiglu
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
                "layer_scale_init": self.layer_scale_init,
                "use_swiglu": self.use_swiglu,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
