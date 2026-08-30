import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape

from .pvt_config import PvtConfig
from .pvt_layers import PvtClsToken, PvtDropPath, PvtSelfAttention, PvtStagePositions

PVT_HUB_SIBLINGS = frozenset({"PvtModel", "PvtImageClassify"})
PATCH_SIZES = (4, 2, 2, 2)
STRIDES = (4, 2, 2, 2)


def grid_to_tokens(x, channels, data_format):
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    return layers.Reshape((-1, channels))(x)


def tokens_to_grid(x, H, W, channels, data_format):
    x = layers.Reshape((H, W, channels))(x)
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 3, 1, 2))
    return x


def pvt_mlp(x, channels, mid, name_prefix):
    x = layers.Dense(mid, name=f"{name_prefix}_dense1")(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(channels, name=f"{name_prefix}_dense2")(x)
    return x


def pvt_block(
    x, H, W, dim, num_heads, sr_ratio, mlp_ratio, drop_prob, stage_idx, block_idx
):
    prefix = f"block_{stage_idx}_{block_idx}"
    drop_path = PvtDropPath(drop_prob)
    norm1 = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"{prefix}_layernorm_1"
    )(x)
    attn = PvtSelfAttention(dim, num_heads, sr_ratio, block_prefix=prefix)(
        norm1, height=H, width=W
    )
    x = layers.Add()([x, drop_path(attn)])
    norm2 = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"{prefix}_layernorm_2"
    )(x)
    mlp = pvt_mlp(norm2, dim, int(dim * mlp_ratio), name_prefix=f"{prefix}_mlp")
    return layers.Add()([x, drop_path(mlp)])


def pvt_backbone_feature(
    inputs,
    *,
    hidden_sizes,
    depths,
    num_attention_heads,
    sr_ratios,
    mlp_ratios,
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
        x = layers.Conv2D(
            hidden_sizes[i],
            PATCH_SIZES[i],
            strides=STRIDES[i],
            padding="valid",
            data_format=data_format,
            name=f"patch_embed_{i}_proj",
        )(x)
        if data_format == "channels_first":
            H, W = int(x.shape[2]), int(x.shape[3])
        else:
            H, W = int(x.shape[1]), int(x.shape[2])
        x = grid_to_tokens(x, hidden_sizes[i], data_format)
        x = layers.LayerNormalization(
            axis=-1, epsilon=1e-6, name=f"patch_embed_{i}_layernorm"
        )(x)
        has_cls = i == 3
        if has_cls:
            x = PvtClsToken(name=f"patch_embed_{i}_cls")(x)
        x = PvtStagePositions(H, W, has_cls=has_cls, name=f"patch_embed_{i}_pos")(x)
        for j in range(depths[i]):
            x = pvt_block(
                x,
                H,
                W,
                hidden_sizes[i],
                num_attention_heads[i],
                sr_ratios[i],
                mlp_ratios[i],
                dpr[cur],
                i,
                j,
            )
            cur += 1
        if i != 3:
            x = tokens_to_grid(x, H, W, hidden_sizes[i], data_format)
            features.append(x)
        else:
            x = layers.LayerNormalization(
                axis=-1, epsilon=1e-6, name="final_layernorm"
            )(x)
            if return_stages:
                patches = layers.Lambda(lambda v: v[:, 1:], name="drop_cls")(x)
                features.append(
                    tokens_to_grid(patches, H, W, hidden_sizes[i], data_format)
                )
    return features if return_stages else x


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtModel(BaseModel):
    """Pyramid Vision Transformer (PVT v1) backbone.

    Four hierarchical stages, each a non-overlapping convolutional patch embedding with a
    learned position embedding, spatial-reduction attention, and a standard feed-forward
    network; the last stage prepends a class token. The default output is the final
    (class-token-carrying) token sequence used by the classifier; ``as_backbone=True``
    returns the four per-stage spatial feature maps. Variable input resolution is supported
    by interpolating each stage's position embedding on weight load.

    References:
    - [Pyramid Vision Transformer](https://arxiv.org/abs/2102.12122)

    Args:
        See :class:`PvtConfig`. ``include_normalization`` bakes ImageNet normalization into
        the graph; ``image_size`` sets the input the model is built for. Defaults describe
        PVT-Tiny.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PvtConfig
    HUB_REPO_SIBLINGS = PVT_HUB_SIBLINGS
    HF_MODEL_TYPE = "pvt"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = PvtImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_sizes": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "sr_ratios": hf_config["sequence_reduction_ratios"],
            "mlp_ratios": hf_config["mlp_ratios"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_pvt_hf_to_keras import transfer_pvt_weights

        transfer_pvt_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        hidden_sizes=(64, 128, 320, 512),
        depths=(2, 2, 2, 2),
        num_attention_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        name="PvtModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
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
        features = pvt_backbone_feature(
            x,
            hidden_sizes=hidden_sizes,
            depths=depths,
            num_attention_heads=num_attention_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
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
class PvtImageClassify(BaseModel):
    """PVT v1 image classifier: :class:`PvtModel` backbone + a Dense head on the class token.

    References:
    - [Pyramid Vision Transformer](https://arxiv.org/abs/2102.12122)

    Args:
        See :class:`PvtConfig`. ``num_classes`` / ``classifier_activation`` are
        head-specific; all other args forward to :class:`PvtModel`.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = PvtConfig
    HUB_REPO_SIBLINGS = PVT_HUB_SIBLINGS
    HF_MODEL_TYPE = "pvt"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_sizes": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "sr_ratios": hf_config["sequence_reduction_ratios"],
            "mlp_ratios": hf_config["mlp_ratios"],
            "num_classes": hf_config.get("num_labels", 1000),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_pvt_hf_to_keras import transfer_pvt_weights

        transfer_pvt_weights(keras_model, state_dict)

    def __init__(
        self,
        hidden_sizes=(64, 128, 320, 512),
        depths=(2, 2, 2, 2),
        num_attention_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        drop_path_rate=0.0,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PvtImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("hf_id", None)

        backbone = PvtModel(
            hidden_sizes=hidden_sizes,
            depths=depths,
            num_attention_heads=num_attention_heads,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )
        tok = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(backbone.output)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(tok)
        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.hidden_sizes = list(hidden_sizes)
        self.depths = list(depths)
        self.num_attention_heads = list(num_attention_heads)
        self.sr_ratios = list(sr_ratios)
        self.mlp_ratios = list(mlp_ratios)
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
