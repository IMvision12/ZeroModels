import keras
from keras import layers

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .pvt_v2_config import PvtV2Config
from .pvt_v2_layers import PvtDropPath, PvtV2SelfAttention

PVT_V2_HUB_SIBLINGS = frozenset({"PvtV2Model", "PvtV2ImageClassify"})

PATCH_SIZES = (7, 3, 3, 3)
STRIDES = (4, 2, 2, 2)


def tokens_to_grid(x, H, W, channels, data_format):
    """(B, H*W, C) tokens -> spatial grid in ``data_format`` layout (functional)."""
    x = layers.Reshape((H, W, channels))(x)
    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 3, 1, 2))
    return x


def grid_to_tokens(x, channels, data_format):
    """Spatial grid -> (B, H*W, C) tokens (functional)."""
    if data_format == "channels_first":
        x = keras.ops.transpose(x, (0, 2, 3, 1))
    return layers.Reshape((-1, channels))(x)


def overlap_patch_embed(x, out_channels, patch_size, stride, data_format, stage_idx):
    """Overlapping patch embed: symmetric ZeroPad -> Conv(valid) -> tokens -> LayerNorm.
    Returns ``(tokens, H, W)``."""
    x = layers.ZeroPadding2D(padding=patch_size // 2, data_format=data_format)(x)
    x = layers.Conv2D(
        out_channels,
        patch_size,
        strides=stride,
        padding="valid",
        data_format=data_format,
        name=f"layers_{stage_idx}_patch_embed_proj",
    )(x)
    if data_format == "channels_first":
        H, W = int(x.shape[2]), int(x.shape[3])
    else:
        H, W = int(x.shape[1]), int(x.shape[2])
    x = grid_to_tokens(x, out_channels, data_format)
    x = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"layers_{stage_idx}_patch_embed_layernorm"
    )(x)
    return x, H, W


def conv_mlp(x, H, W, channels, mid_channels, linear, data_format, name_prefix):
    """PVTv2 conv-FFN: Dense -> (ReLU if linear) -> DWConv -> GELU -> Dense."""
    x = layers.Dense(mid_channels, name=f"{name_prefix}_dense1")(x)
    if linear:
        x = layers.Activation("relu")(x)
    grid = tokens_to_grid(x, H, W, mid_channels, data_format)
    grid = layers.DepthwiseConv2D(
        3,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_dwconv",
    )(grid)
    x = grid_to_tokens(grid, mid_channels, data_format)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(channels, name=f"{name_prefix}_dense2")(x)
    return x


def pvt_v2_block(
    x,
    H,
    W,
    dim,
    num_heads,
    sr_ratio,
    mlp_ratio,
    linear,
    drop_prob,
    data_format,
    stage_idx,
    block_idx,
):
    prefix = f"layers_{stage_idx}_blocks_{block_idx}"
    drop_path = PvtDropPath(drop_prob)

    norm1 = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"{prefix}_layernorm_1"
    )(x)
    attn = PvtV2SelfAttention(
        dim,
        num_heads,
        sr_ratio,
        linear_attention=linear,
        qkv_bias=True,
        block_prefix=prefix,
    )(norm1, height=H, width=W)
    x = layers.Add()([x, drop_path(attn)])

    norm2 = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"{prefix}_layernorm_2"
    )(x)
    mlp = conv_mlp(
        norm2,
        H,
        W,
        channels=dim,
        mid_channels=int(dim * mlp_ratio),
        linear=linear,
        data_format=data_format,
        name_prefix=f"{prefix}_mlp",
    )
    return layers.Add()([x, drop_path(mlp)])


def pvt_v2_backbone_feature(
    inputs,
    *,
    hidden_sizes,
    depths,
    num_attention_heads,
    sr_ratios,
    mlp_ratios,
    linear_attention,
    drop_path_rate,
    data_format,
    return_stages=False,
):
    total = sum(depths)
    dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
    x = inputs
    features = []
    cur = 0
    for i in range(4):
        x, H, W = overlap_patch_embed(
            x, hidden_sizes[i], PATCH_SIZES[i], STRIDES[i], data_format, i
        )
        for j in range(depths[i]):
            x = pvt_v2_block(
                x,
                H,
                W,
                hidden_sizes[i],
                num_attention_heads[i],
                sr_ratios[i],
                mlp_ratios[i],
                linear_attention,
                dpr[cur],
                data_format,
                i,
                j,
            )
            cur += 1
        x = layers.LayerNormalization(
            axis=-1, epsilon=1e-6, name=f"layers_{i}_layernorm"
        )(x)
        x = tokens_to_grid(x, H, W, hidden_sizes[i], data_format)
        features.append(x)
    return features if return_stages else features[-1]


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtV2Model(BaseModel):
    """Pyramid Vision Transformer v2 (PVTv2) backbone.

    A hierarchical, convolution-augmented transformer: four stages, each an overlapping
    convolutional patch embedding, spatial-reduction (or linear) attention, and a
    convolutional feed-forward network (a 3x3 depthwise conv between the two Dense
    layers). It uses no learned position embeddings, so variable input resolution works
    out of the box. Output is the final stage's spatial feature map, or, with
    ``as_backbone=True``, the four per-stage feature maps.

    References:
    - [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

    Args:
        See :class:`PvtV2Config`. ``as_backbone`` returns the 4-stage pyramid;
        ``image_size`` sets the input the model is built for. The model takes
        already-normalized input; :class:`PvtV2ImageProcessor` handles the ImageNet
        normalization. Defaults describe PVTv2-B0.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PvtV2Config
    HUB_REPO_SIBLINGS = PVT_V2_HUB_SIBLINGS
    HF_MODEL_TYPE = "pvt_v2"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = PvtV2ImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_sizes": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "sr_ratios": hf_config["sr_ratios"],
            "mlp_ratios": hf_config["mlp_ratios"],
            "linear_attention": hf_config.get("linear_attention", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_pvt_v2_hf_to_keras import transfer_pvt_v2_weights

        transfer_pvt_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        hidden_sizes=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_attention_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        linear_attention=False,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        name="PvtV2Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "hf_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not keras.utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = img_input
        features = pvt_v2_backbone_feature(
            x,
            hidden_sizes=hidden_sizes,
            depths=depths,
            num_attention_heads=num_attention_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            linear_attention=linear_attention,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            return_stages=as_backbone,
        )
        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.hidden_sizes = list(hidden_sizes)
        self.depths = list(depths)
        self.num_attention_heads = list(num_attention_heads)
        self.sr_ratios = list(sr_ratios)
        self.mlp_ratios = list(mlp_ratios)
        self.linear_attention = linear_attention
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "hidden_sizes": self.hidden_sizes,
                "depths": self.depths,
                "num_attention_heads": self.num_attention_heads,
                "sr_ratios": self.sr_ratios,
                "mlp_ratios": self.mlp_ratios,
                "linear_attention": self.linear_attention,
                "drop_path_rate": self.drop_path_rate,
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
class PvtV2ImageClassify(BaseModel):
    """PVTv2 image classifier: :class:`PvtV2Model` backbone + global-average-pool +
    a single Dense head over the last stage's feature map.

    References:
    - [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

    Args:
        See :class:`PvtV2Config`. ``num_classes`` / ``classifier_activation`` are
        head-specific; all other args forward to :class:`PvtV2Model`.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PvtV2Config
    HUB_REPO_SIBLINGS = PVT_V2_HUB_SIBLINGS
    HF_MODEL_TYPE = "pvt_v2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_sizes": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "sr_ratios": hf_config["sr_ratios"],
            "mlp_ratios": hf_config["mlp_ratios"],
            "linear_attention": hf_config.get("linear_attention", False),
            "num_classes": hf_config.get("num_labels", 1000),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_pvt_v2_hf_to_keras import transfer_pvt_v2_weights

        transfer_pvt_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        hidden_sizes=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_attention_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        linear_attention=False,
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PvtV2ImageClassify",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)
        data_format = keras.config.image_data_format()

        backbone = PvtV2Model(
            hidden_sizes=hidden_sizes,
            depths=depths,
            num_attention_heads=num_attention_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            linear_attention=linear_attention,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
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

        self.hidden_sizes = list(hidden_sizes)
        self.depths = list(depths)
        self.num_attention_heads = list(num_attention_heads)
        self.sr_ratios = list(sr_ratios)
        self.mlp_ratios = list(mlp_ratios)
        self.linear_attention = linear_attention
        self.drop_path_rate = drop_path_rate
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_sizes": self.hidden_sizes,
                "depths": self.depths,
                "num_attention_heads": self.num_attention_heads,
                "sr_ratios": self.sr_ratios,
                "mlp_ratios": self.mlp_ratios,
                "linear_attention": self.linear_attention,
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
