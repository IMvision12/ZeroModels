import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.beit.beit_layers import (
    BeitAttention,
    BeitClsToken,
    BeitLayerScale,
)
from zeromodels.models.pvt_v2.pvt_v2_layers import adaptive_pool_matrix
from zeromodels.utils import standardize_input_shape

from .beit_config import BeitConfig

BEIT_HUB_SIBLINGS = frozenset({"BeitModel", "BeitImageClassify"})


def beit_encoder_layer(
    x,
    hidden_size,
    num_heads,
    intermediate_size,
    window_size,
    layer_scale_init,
    layer_norm_eps,
    block_prefix,
):
    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_layernorm_before"
    )(x)
    h = BeitAttention(
        hidden_size,
        num_heads,
        window_size,
        block_prefix=f"{block_prefix}_attn",
        name=f"{block_prefix}_attn",
    )(h)
    h = BeitLayerScale(layer_scale_init, name=f"{block_prefix}_layerscale_1")(h)
    x = layers.Add(name=f"{block_prefix}_add_1")([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_layernorm_after"
    )(x)
    h = layers.Dense(intermediate_size, name=f"{block_prefix}_fc1")(h)
    h = layers.Activation("gelu", name=f"{block_prefix}_gelu")(h)
    h = layers.Dense(hidden_size, name=f"{block_prefix}_fc2")(h)
    h = BeitLayerScale(layer_scale_init, name=f"{block_prefix}_layerscale_2")(h)
    x = layers.Add(name=f"{block_prefix}_add_2")([residual, h])
    return x


def beit_backbone_feature(
    inputs,
    hidden_size,
    depth,
    num_heads,
    intermediate_size,
    patch_size,
    layer_scale_init,
    layer_norm_eps,
    image_size,
    data_format,
    return_intermediates=False,
):
    grid = image_size // patch_size
    x = layers.Conv2D(
        hidden_size,
        patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="patch_embed",
    )(inputs)
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((grid * grid, hidden_size))(x)
    x = BeitClsToken(hidden_size, name="cls_token")(x)

    window_size = (grid, grid)
    intermediates = [x]
    for i in range(depth):
        x = beit_encoder_layer(
            x,
            hidden_size,
            num_heads,
            intermediate_size,
            window_size,
            layer_scale_init,
            layer_norm_eps,
            block_prefix=f"beit_layer_{i}",
        )
        intermediates.append(x)

    if return_intermediates:
        return intermediates
    return x


def beit_common_config(hf_config):
    image_size = hf_config["image_size"]
    if not isinstance(image_size, int):
        image_size = image_size[0]
    return {
        "hidden_size": hf_config["hidden_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "intermediate_size": hf_config["intermediate_size"],
        "patch_size": hf_config["patch_size"],
        "image_size": image_size,
        "layer_scale_init_value": hf_config.get("layer_scale_init_value", 0.1),
        "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-12),
    }


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitModel(BaseModel):
    """BEiT backbone: patch stem, CLS token, and pre-norm transformer blocks.

    Each block runs relative-position-bias self-attention and an MLP, both on a
    layer-scaled residual branch. There are no absolute position embeddings and no
    final backbone LayerNorm (BEiT normalizes only inside the mean-pooling head).
    Output is the token sequence ``(B, num_patches + 1, hidden_size)`` (the first
    token is the CLS token). With ``as_backbone=True`` it returns the per-block
    token sequences (the initial embedding plus one per block), which the
    segmentation head reshapes into 2D feature maps.

    Reference:
        - `BEiT: BERT Pre-Training of Image Transformers
          <https://arxiv.org/abs/2106.08254>`_

    Args:
        as_backbone: If True, return the list of per-block token sequences instead
            of only the final one. Defaults to False.
        hidden_size: Transformer width. Defaults to 768.
        num_hidden_layers: Number of transformer blocks. Defaults to 12.
        num_attention_heads: Attention heads per block. Defaults to 12.
        intermediate_size: MLP inner dimension. Defaults to 3072.
        patch_size: Patch (conv-stem) size. Defaults to 16.
        layer_scale_init_value: Initial layer-scale value. Defaults to 0.1.
        layer_norm_eps: Epsilon of every LayerNorm. Defaults to 1e-12.
        image_size: Square input resolution the model is built for. Defaults to 224.
        input_tensor: Optional pre-existing Keras input tensor. Defaults to None.
        name: Model name. Defaults to ``"BeitModel"``.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = BeitConfig
    HUB_REPO_SIBLINGS = BEIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "beit"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = BeitImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return beit_common_config(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_beit_hf_to_keras import transfer_beit_weights

        transfer_beit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=16,
        layer_scale_init_value=0.1,
        layer_norm_eps=1e-12,
        image_size=224,
        input_tensor=None,
        name="BeitModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "hf_id"):
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

        x = img_input
        x = beit_backbone_feature(
            x,
            hidden_size=hidden_size,
            depth=num_hidden_layers,
            num_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            patch_size=patch_size,
            layer_scale_init=layer_scale_init_value,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            data_format=data_format,
            return_intermediates=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.layer_scale_init_value = layer_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "patch_size": self.patch_size,
                "layer_scale_init_value": self.layer_scale_init_value,
                "layer_norm_eps": self.layer_norm_eps,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitImageClassify(BaseModel):
    """BEiT image classifier: [`BeitModel`] backbone + mean-pool head.

    The classifier mean-pools the final patch tokens (excluding the CLS token),
    applies a LayerNorm (the BEiT "pooler"), and a single Dense to produce
    ``num_classes`` logits.

    Reference:
        - `BEiT: BERT Pre-Training of Image Transformers
          <https://arxiv.org/abs/2106.08254>`_

    Args:
        See [`BeitModel`]. ``num_classes`` / ``classifier_activation`` are
        head-specific; all other args forward to the backbone.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = BeitConfig
    HUB_REPO_SIBLINGS = BEIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "beit"

    @classmethod
    def config_from_hf(cls, hf_config):
        config = beit_common_config(hf_config)
        config["num_classes"] = len(hf_config.get("id2label", {})) or 1000
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_beit_hf_to_keras import transfer_beit_weights

        transfer_beit_weights(keras_model, state_dict)

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=16,
        layer_scale_init_value=0.1,
        layer_norm_eps=1e-12,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="BeitImageClassify",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)

        backbone = BeitModel(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            patch_size=patch_size,
            layer_scale_init_value=layer_scale_init_value,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = backbone.output
        pooled = layers.Lambda(
            lambda v: ops.mean(v[:, 1:, :], axis=1), name="mean_pool"
        )(x)
        pooled = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="pooler_layernorm"
        )(pooled)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(pooled)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.layer_scale_init_value = layer_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "patch_size": self.patch_size,
                "layer_scale_init_value": self.layer_scale_init_value,
                "layer_norm_eps": self.layer_norm_eps,
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


def beit_conv_bn(x, out_channels, kernel_size, name, activation="relu"):
    padding = "same" if kernel_size > 1 else "valid"
    x = layers.Conv2D(
        out_channels,
        kernel_size,
        padding=padding,
        use_bias=False,
        data_format="channels_last",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, name=f"{name}_bn")(x)
    if activation == "relu":
        x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def beit_resize(x, size):
    return ops.image.resize(
        x, size, interpolation="bilinear", data_format="channels_last"
    )


def beit_adaptive_avg_pool(x, out_size):
    mh = adaptive_pool_matrix(x.shape[1], out_size)  # (out, h)
    mw = adaptive_pool_matrix(x.shape[2], out_size)  # (out, w)
    x = ops.einsum("oh,bhwc->bowc", mh, x)
    x = ops.einsum("pw,bowc->bopc", mw, x)
    return x


def beit_uperhead(feature_maps, hidden, pool_scales, num_labels):
    last = feature_maps[-1]
    size = (last.shape[1], last.shape[2])
    psp_outs = []
    for i, scale in enumerate(pool_scales):
        p = beit_adaptive_avg_pool(last, scale)
        p = beit_conv_bn(p, hidden, 1, name=f"psp_{i}")
        psp_outs.append(beit_resize(p, size))
    psp = ops.concatenate([last, *psp_outs], axis=-1)
    psp = beit_conv_bn(psp, hidden, 3, name="psp_bottleneck")

    laterals = [
        beit_conv_bn(feature_maps[j], hidden, 1, name=f"lateral_{j}") for j in range(3)
    ]
    laterals.append(psp)

    for i in range(3, 0, -1):
        target = (laterals[i - 1].shape[1], laterals[i - 1].shape[2])
        laterals[i - 1] = layers.Add(name=f"fpn_topdown_{i}")(
            [laterals[i - 1], beit_resize(laterals[i], target)]
        )

    fpn_outs = [
        beit_conv_bn(laterals[j], hidden, 3, name=f"fpn_conv_{j}") for j in range(3)
    ]
    fpn_outs.append(laterals[3])
    target = (fpn_outs[0].shape[1], fpn_outs[0].shape[2])
    fpn_outs = [fpn_outs[0]] + [beit_resize(f, target) for f in fpn_outs[1:]]
    fused = ops.concatenate(fpn_outs, axis=-1)
    fused = beit_conv_bn(fused, hidden, 3, name="fpn_bottleneck")
    return layers.Conv2D(
        num_labels,
        1,
        padding="valid",
        use_bias=True,
        data_format="channels_last",
        name="seg_classifier",
    )(fused)


def beit_fpn_neck(feature_maps, hidden):
    f0 = layers.Conv2DTranspose(
        hidden, 2, strides=2, data_format="channels_last", name="fpn1_convtranspose1"
    )(feature_maps[0])
    f0 = layers.BatchNormalization(axis=-1, epsilon=1e-5, name="fpn1_bn")(f0)
    f0 = layers.Activation("gelu", name="fpn1_gelu")(f0)
    f0 = layers.Conv2DTranspose(
        hidden, 2, strides=2, data_format="channels_last", name="fpn1_convtranspose2"
    )(f0)
    f1 = layers.Conv2DTranspose(
        hidden, 2, strides=2, data_format="channels_last", name="fpn2_convtranspose"
    )(feature_maps[1])
    f2 = feature_maps[2]
    f3 = layers.MaxPooling2D(
        2, strides=2, data_format="channels_last", name="fpn4_maxpool"
    )(feature_maps[3])
    return [f0, f1, f2, f3]


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitSemanticSegment(BaseModel):
    """BEiT semantic segmentation: [`BeitModel`] backbone + FPN neck + UPerNet head.

    Four intermediate token sequences (at ``out_indices``) are reshaped into 2D
    feature maps, rescaled by a small FPN neck (x4 up, x2 up, identity, x2 down),
    and decoded by a UPerNet head (pyramid pooling module + FPN fusion) into
    per-pixel class logits at a quarter of the input resolution. The training-only
    auxiliary FCN head is omitted. The segmentation head runs in ``channels_last``.

    Reference:
        - `BEiT: BERT Pre-Training of Image Transformers
          <https://arxiv.org/abs/2106.08254>`_
        - `UPerNet <https://arxiv.org/abs/1807.10221>`_

    Args:
        See [`BeitModel`], plus:
        num_classes: Number of segmentation labels. Defaults to 150 (ADE20K).
        out_indices: 1-based encoder layers feeding the FPN neck.
            Defaults to ``(3, 5, 7, 11)``.
        pool_scales: Pyramid-pooling scales. Defaults to ``(1, 2, 3, 6)``.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = BeitConfig
    HF_MODEL_TYPE = "beit"

    @classmethod
    def config_from_hf(cls, hf_config):
        config = beit_common_config(hf_config)
        config["num_classes"] = len(hf_config.get("id2label", {})) or 150
        # BEiT stores the FPN out_indices under `segmentation_indices` (transformers
        # maps it to out_indices in __post_init__); it is depth-dependent
        # ([3,5,7,11] base / [7,11,15,23] large), so never hardcode the base value.
        out_indices = (
            hf_config.get("out_indices")
            or hf_config.get("segmentation_indices")
            or (3, 5, 7, 11)
        )
        config["out_indices"] = tuple(out_indices)
        config["pool_scales"] = tuple(hf_config.get("pool_scales", (1, 2, 3, 6)))
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_beit_hf_to_keras import (
            transfer_beit_seg_head,
            transfer_beit_weights,
        )

        transfer_beit_weights(keras_model, state_dict)
        transfer_beit_seg_head(keras_model, state_dict)

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=16,
        layer_scale_init_value=0.1,
        layer_norm_eps=1e-12,
        image_size=640,
        input_tensor=None,
        num_classes=150,
        out_indices=(3, 5, 7, 11),
        pool_scales=(1, 2, 3, 6),
        name="BeitSemanticSegment",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)

        backbone = BeitModel(
            as_backbone=True,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            patch_size=patch_size,
            layer_scale_init_value=layer_scale_init_value,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )
        grid = backbone.image_size // patch_size
        intermediates = backbone.output

        feats = []
        for idx in out_indices:
            tok = intermediates[idx - 1]
            g = layers.Lambda(lambda v: v[:, 1:, :])(tok)
            g = layers.Reshape((grid, grid, hidden_size))(g)
            feats.append(g)

        feats = beit_fpn_neck(feats, hidden_size)
        logits = beit_uperhead(feats, hidden_size, pool_scales, num_classes)

        if keras.config.image_data_format() == "channels_first":
            logits = ops.transpose(logits, (0, 3, 1, 2))

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.layer_scale_init_value = layer_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.out_indices = tuple(out_indices)
        self.pool_scales = tuple(pool_scales)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "patch_size": self.patch_size,
                "layer_scale_init_value": self.layer_scale_init_value,
                "layer_norm_eps": self.layer_norm_eps,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "out_indices": self.out_indices,
                "pool_scales": self.pool_scales,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
