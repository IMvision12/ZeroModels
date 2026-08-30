import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.levit.levit_layers import (
    BN_EPS,
    LevitAttention,
    LevitAttentionSubsample,
    LevitMLP,
)
from zeromodels.utils import standardize_input_shape

from .levit_config import LevitConfig

LEVIT_HUB_SIBLINGS = frozenset({"LevitModel", "LevitImageClassify"})


def levit_patch_embeddings(x, hidden_size, kernel_size, stride, padding, data_format):
    channel_axis = 1 if data_format == "channels_first" else -1
    channels = [hidden_size // 8, hidden_size // 4, hidden_size // 2, hidden_size]
    for i, out_channels in enumerate(channels):
        x = layers.ZeroPadding2D(
            padding=padding, data_format=data_format, name=f"patch_pad_{i + 1}"
        )(x)
        x = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name=f"patch_conv_{i + 1}",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, epsilon=BN_EPS, momentum=0.9, name=f"patch_bn_{i + 1}"
        )(x)
        if i < 3:
            x = layers.Activation("hard_silu", name=f"patch_act_{i + 1}")(x)
    return x


def levit_backbone_feature(
    inputs,
    image_size,
    patch_size,
    kernel_size,
    stride,
    padding,
    hidden_sizes,
    num_attention_heads,
    depths,
    key_dim,
    mlp_ratio,
    attention_ratio,
    data_format,
):
    grid = image_size // patch_size
    x = levit_patch_embeddings(
        inputs, hidden_sizes[0], kernel_size, stride, padding, data_format
    )
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((grid * grid, hidden_sizes[0]))(x)

    resolution = grid
    for s in range(len(depths)):
        for d in range(depths[s]):
            attn = LevitAttention(
                hidden_sizes[s],
                key_dim[s],
                num_attention_heads[s],
                attention_ratio[s],
                resolution,
                name=f"stage{s}_attn{d}",
            )(x)
            x = layers.Add(name=f"stage{s}_attn{d}_res")([x, attn])
            mlp = LevitMLP(
                hidden_sizes[s] * mlp_ratio[s],
                hidden_sizes[s],
                name=f"stage{s}_mlp{d}",
            )(x)
            x = layers.Add(name=f"stage{s}_mlp{d}_res")([x, mlp])

        if s < len(depths) - 1:
            resolution_out = (resolution - 1) // 2 + 1
            x = LevitAttentionSubsample(
                hidden_sizes[s],
                hidden_sizes[s + 1],
                key_dim=key_dim[0],
                num_heads=hidden_sizes[s] // key_dim[0],
                attention_ratio=4,
                stride=2,
                resolution_in=resolution,
                resolution_out=resolution_out,
                name=f"stage{s}_subsample",
            )(x)
            resolution = resolution_out
            sub_mlp = LevitMLP(
                hidden_sizes[s + 1] * 2,
                hidden_sizes[s + 1],
                name=f"stage{s}_sub_mlp",
            )(x)
            x = layers.Add(name=f"stage{s}_sub_mlp_res")([x, sub_mlp])
    return x


def levit_common_config(hf_config):
    image_size = hf_config["image_size"]
    if not isinstance(image_size, int):
        image_size = image_size[0]
    return {
        "image_size": image_size,
        "num_channels": hf_config.get("num_channels", 3),
        "kernel_size": hf_config.get("kernel_size", 3),
        "stride": hf_config.get("stride", 2),
        "padding": hf_config.get("padding", 1),
        "patch_size": hf_config.get("patch_size", 16),
        "hidden_sizes": tuple(hf_config["hidden_sizes"]),
        "num_attention_heads": tuple(hf_config["num_attention_heads"]),
        "depths": tuple(hf_config["depths"]),
        "key_dim": tuple(hf_config["key_dim"]),
        "mlp_ratio": tuple(hf_config["mlp_ratio"]),
        "attention_ratio": tuple(hf_config["attention_ratio"]),
    }


@keras.saving.register_keras_serializable(package="zeromodels")
class LevitModel(BaseModel):
    """LeViT backbone: a convolutional patch stem plus attention stages.

    A four-layer conv stem (each stride 2) downsamples the image 16x and flattens it
    to a token sequence, then three stages of relative-position-bias attention and
    2x-expansion MLPs run over the tokens, with an ``AttentionSubsample`` halving the
    grid between stages. Output is the final token sequence
    ``(B, num_tokens, hidden_sizes[-1])``.

    Reference:
        - `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
          <https://arxiv.org/abs/2104.01136>`_

    Args:
        image_size: Square input resolution the model is built for. Defaults to 224.
        num_channels: Number of input channels. Defaults to 3.
        kernel_size: Patch-stem conv kernel size. Defaults to 3.
        stride: Patch-stem conv stride. Defaults to 2.
        padding: Patch-stem conv zero-padding. Defaults to 1.
        patch_size: Total patch-stem downsampling. Defaults to 16.
        hidden_sizes: Token dimension of each stage. Defaults to (128, 256, 384).
        num_attention_heads: Attention heads per stage. Defaults to (4, 8, 12).
        depths: Attention blocks per stage. Defaults to (4, 4, 4).
        key_dim: Per-head key dimension per stage. Defaults to (16, 16, 16).
        mlp_ratio: MLP expansion per stage. Defaults to (2, 2, 2).
        attention_ratio: Value-to-key dimension ratio per stage. Defaults to
            (2, 2, 2).
        input_tensor: Optional pre-existing Keras input tensor. Defaults to None.
        name: Model name. Defaults to ``"LevitModel"``.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = LevitConfig
    HUB_REPO_SIBLINGS = LEVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "levit"

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = LevitImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return levit_common_config(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_levit_hf_to_keras import transfer_levit_weights

        transfer_levit_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=(128, 256, 384),
        num_attention_heads=(4, 8, 12),
        depths=(4, 4, 4),
        key_dim=(16, 16, 16),
        mlp_ratio=(2, 2, 2),
        attention_ratio=(2, 2, 2),
        input_tensor=None,
        name="LevitModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "use_distillation", "hf_id"):
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
        x = levit_backbone_feature(
            x,
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            hidden_sizes=hidden_sizes,
            num_attention_heads=num_attention_heads,
            depths=depths,
            key_dim=key_dim,
            mlp_ratio=mlp_ratio,
            attention_ratio=attention_ratio,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.patch_size = patch_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_attention_heads = tuple(num_attention_heads)
        self.depths = tuple(depths)
        self.key_dim = tuple(key_dim)
        self.mlp_ratio = tuple(mlp_ratio)
        self.attention_ratio = tuple(attention_ratio)
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "patch_size": self.patch_size,
                "hidden_sizes": self.hidden_sizes,
                "num_attention_heads": self.num_attention_heads,
                "depths": self.depths,
                "key_dim": self.key_dim,
                "mlp_ratio": self.mlp_ratio,
                "attention_ratio": self.attention_ratio,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def levit_head(pooled, num_classes, activation, name):
    x = layers.BatchNormalization(
        axis=-1, epsilon=BN_EPS, momentum=0.9, name=f"{name}_bn"
    )(pooled)
    return layers.Dense(num_classes, activation=activation, name=f"{name}_linear")(x)


@keras.saving.register_keras_serializable(package="zeromodels")
class LevitImageClassify(BaseModel):
    """LeViT image classifier: [`LevitModel`] backbone + mean-pool BatchNorm head.

    The final tokens are mean-pooled and passed through a BatchNorm + Dense head.
    With ``use_distillation=True`` (the released ``facebook/levit-*`` recipe), a
    second identical head reads the same pooled features and the two heads' logits
    are averaged, matching ``LevitForImageClassificationWithTeacher``.

    Reference:
        - `LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference
          <https://arxiv.org/abs/2104.01136>`_

    Args:
        use_distillation: Add a second (distillation) head and average the two
            heads' logits. Defaults to True.
        num_classes: Number of classifier outputs. Defaults to 1000.
        classifier_activation: Head activation. Defaults to ``"linear"``.
        name: Model name. Defaults to ``"LevitImageClassify"``.

    See :class:`LevitModel` for the backbone architecture arguments.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = LevitConfig
    HUB_REPO_SIBLINGS = LEVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = "levit"

    @classmethod
    def config_from_hf(cls, hf_config):
        config = levit_common_config(hf_config)
        config["num_classes"] = len(hf_config.get("id2label", {})) or 1000
        architectures = hf_config.get("architectures") or []
        config["use_distillation"] = (
            any("WithTeacher" in a for a in architectures) if architectures else True
        )
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_levit_hf_to_keras import transfer_levit_weights

        transfer_levit_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=(128, 256, 384),
        num_attention_heads=(4, 8, 12),
        depths=(4, 4, 4),
        key_dim=(16, 16, 16),
        mlp_ratio=(2, 2, 2),
        attention_ratio=(2, 2, 2),
        input_tensor=None,
        num_classes=1000,
        use_distillation=True,
        classifier_activation="linear",
        name="LevitImageClassify",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)

        backbone = LevitModel(
            image_size=image_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            patch_size=patch_size,
            hidden_sizes=hidden_sizes,
            num_attention_heads=num_attention_heads,
            depths=depths,
            key_dim=key_dim,
            mlp_ratio=mlp_ratio,
            attention_ratio=attention_ratio,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        # Pool over the token axis explicitly (the sequence is always
        # (B, seq, hidden) after the patch-embed flatten); GlobalAveragePooling1D
        # would pool the wrong axis under a channels_first global data format.
        pooled = layers.Lambda(
            lambda v: ops.mean(v, axis=1),
            output_shape=lambda s: (s[0], s[2]),
            name="mean_pool",
        )(backbone.output)
        cls_logits = levit_head(
            pooled, num_classes, classifier_activation, "classifier"
        )
        if use_distillation:
            distill_logits = levit_head(
                pooled, num_classes, classifier_activation, "classifier_distill"
            )
            out = layers.Average(name="distillation_average")(
                [cls_logits, distill_logits]
            )
        else:
            out = cls_logits

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.image_size = backbone.image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.patch_size = patch_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_attention_heads = tuple(num_attention_heads)
        self.depths = tuple(depths)
        self.key_dim = tuple(key_dim)
        self.mlp_ratio = tuple(mlp_ratio)
        self.attention_ratio = tuple(attention_ratio)
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.use_distillation = use_distillation
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "patch_size": self.patch_size,
                "hidden_sizes": self.hidden_sizes,
                "num_attention_heads": self.num_attention_heads,
                "depths": self.depths,
                "key_dim": self.key_dim,
                "mlp_ratio": self.mlp_ratio,
                "attention_ratio": self.attention_ratio,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "use_distillation": self.use_distillation,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
