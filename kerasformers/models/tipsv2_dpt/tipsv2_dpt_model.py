import keras
from keras import layers, utils

from kerasformers.base import FunctionalBaseModel
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.models.tipsv2.tipsv2_layers import Tipsv2RegisterTokens
from kerasformers.models.vit.vit_layers import ViTAddPositionEmbs, ViTClassDistToken
from kerasformers.models.vit.vit_model import transformer_block
from kerasformers.utils import standardize_input_shape

from .tipsv2_dpt_config import Tipsv2DptConfig
from .tipsv2_dpt_layers import (
    Tipsv2DptFeaturesToDepth,
    Tipsv2DptReadout,
    Tipsv2DptResize,
    gelu_tanh,
)

DEFAULT_OUT_INDICES = [7, 14, 21, 27]
DEFAULT_NECK_HIDDEN_SIZES = [144, 288, 576, 1152]
DEFAULT_REASSEMBLE_FACTORS = [4, 2, 1, 0.5]

TIPSV2_DPT_HUB_SIBLINGS = frozenset(
    {
        "Tipsv2DptDensePredict",
        "Tipsv2DptDepthEstimation",
        "Tipsv2DptSemanticSegment",
    }
)


def backbone_features(
    images,
    *,
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    patch_size,
    num_register_tokens,
    layerscale_value,
    use_swiglu,
    layer_norm_eps,
    out_indices,
    image_size,
    data_format,
):
    """TIPSv2 vision backbone; returns the LN'd hidden states at ``out_indices``.

    Each returned tensor is ``(B, 1 + num_register_tokens + num_patches, embed_dim)``
    (sequence form). The shared final LayerNorm is applied to every selected stage
    (config ``apply_layernorm=True``, ``reshape_hidden_states=False``).
    """
    grid = image_size // patch_size
    x = layers.Conv2D(
        embed_dim,
        patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="conv1",
    )(images)
    x = layers.Reshape((-1, embed_dim))(x)
    x = ViTClassDistToken(use_distillation=False, name="cls_token")(x)
    x = ViTAddPositionEmbs(
        name="pos_embed",
        no_embed_class=False,
        use_distillation=False,
        grid_h=grid,
        grid_w=grid,
        resize_mode="bilinear",
    )(x)
    x = Tipsv2RegisterTokens(num_tokens=num_register_tokens, name="register_tokens")(x)

    capture_at = {
        oi - 1 for oi in out_indices
    }  # stage i (1-indexed) = block i-1 output
    final_ln = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name="final_layernorm"
    )
    captured = {}
    for i in range(depth):
        x = transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_norm=False,
            block_idx=i,
            layer_scale_init=layerscale_value,
            use_swiglu=use_swiglu,
        )
        if i in capture_at:
            captured[i] = final_ln(x)
    return [captured[oi - 1] for oi in out_indices]


def pre_act_residual(x, channels, name, data_format):
    r = x
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        channels,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv1",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        channels,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    return layers.Add()([x, r])


def reassemble(
    feat_seq,
    i,
    task,
    *,
    hidden,
    grid,
    neck_size,
    factor,
    num_register_tokens,
    data_format,
):
    cls = feat_seq[:, 0]
    patch = feat_seq[:, 1 + num_register_tokens :]
    cat = Tipsv2DptReadout(name=f"{task}_reassemble_readout_cat_{i}")(patch, cls)
    x = layers.Dense(hidden, name=f"{task}_reassemble_readout_{i}")(cat)
    x = layers.Activation(gelu_tanh)(x)
    x = layers.Reshape((grid, grid, hidden))(x)
    x = layers.Conv2D(
        neck_size, 1, data_format=data_format, name=f"{task}_reassemble_proj_{i}"
    )(x)
    if factor > 1:
        f = int(factor)
        x = layers.Conv2DTranspose(
            neck_size,
            f,
            strides=f,
            padding="valid",
            data_format=data_format,
            name=f"{task}_reassemble_resize_{i}",
        )(x)
    elif factor < 1:
        s = int(round(1.0 / factor))
        x = layers.ZeroPadding2D(1, data_format=data_format)(x)
        x = layers.Conv2D(
            neck_size,
            3,
            strides=s,
            padding="valid",
            data_format=data_format,
            name=f"{task}_reassemble_resize_{i}",
        )(x)
    return x


def fusion_layer(hidden, residual, has_residual, channels, target, name, data_format):
    if residual is not None and has_residual:
        residual = pre_act_residual(residual, channels, f"{name}_residual", data_format)
        hidden = layers.Add()([hidden, residual])
    hidden = pre_act_residual(hidden, channels, f"{name}_main", data_format)
    hidden = Tipsv2DptResize(target, target, data_format=data_format)(hidden)
    hidden = layers.Conv2D(
        channels, 1, use_bias=True, data_format=data_format, name=f"{name}_out_conv"
    )(hidden)
    return hidden


def neck(
    features,
    task,
    *,
    hidden,
    grid,
    neck_hidden_sizes,
    reassemble_factors,
    fusion_hidden_size,
    num_register_tokens,
    data_format,
):
    reassembled = [
        reassemble(
            features[i],
            i,
            task,
            hidden=hidden,
            grid=grid,
            neck_size=neck_hidden_sizes[i],
            factor=reassemble_factors[i],
            num_register_tokens=num_register_tokens,
            data_format=data_format,
        )
        for i in range(len(features))
    ]
    convd = [
        layers.Conv2D(
            fusion_hidden_size,
            3,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{task}_conv_{i}",
        )(reassembled[i])
        for i in range(len(reassembled))
    ]
    spat = [int(grid * f) for f in reassemble_factors]
    rev = convd[::-1]
    rev_spat = spat[::-1]
    fused = None
    for i in range(len(rev)):
        target = rev_spat[i] * 2
        if fused is None:
            fused = fusion_layer(
                rev[i],
                None,
                False,
                fusion_hidden_size,
                target,
                f"{task}_fusion_{i}",
                data_format,
            )
        else:
            fused = fusion_layer(
                fused,
                rev[i],
                True,
                fusion_hidden_size,
                target,
                f"{task}_fusion_{i}",
                data_format,
            )
    return fused


def decoder(fused, task, out_channels, activation, fusion_hidden_size, data_format):
    x = layers.Conv2D(
        fusion_hidden_size,
        3,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name=f"{task}_project",
    )(fused)
    if activation is not None:
        x = layers.Activation(activation)(x)
    x = layers.Dense(out_channels, name=f"{task}_head_linear")(x)
    return x


def task_output(
    features,
    task,
    *,
    neck_kw,
    fusion_hidden_size,
    num_depth_bins,
    min_depth,
    max_depth,
    depth_decoder_activation,
    num_labels,
    data_format,
):
    """Build one DPT task head, returning ``(output_key, tensor)``."""
    fused = neck(features, task, **neck_kw)
    if task == "depth":
        logits = decoder(
            fused,
            "depth",
            num_depth_bins,
            depth_decoder_activation,
            fusion_hidden_size,
            data_format,
        )
        depth = Tipsv2DptFeaturesToDepth(
            num_depth_bins=num_depth_bins,
            min_depth=min_depth,
            max_depth=max_depth,
            name="depth_bin_regressor",
        )(logits)
        return "predicted_depth", depth
    if task == "segmentation":
        return "segmentation_logits", decoder(
            fused, "segmentation", num_labels, None, fusion_hidden_size, data_format
        )
    raise ValueError(f"Unknown DPT task: {task!r}")


class Tipsv2DptBase(FunctionalBaseModel):
    """Shared TIPSv2-DPT construction, config, and loading for the task models.

    Subclasses set :attr:`TASKS` (the DPT heads to build). All heads share the same
    per-task layer names, so one weight converter serves every task model, and a
    single-task model warm-starts from :class:`Tipsv2DptDensePredict` weights.
    """

    TASKS = ()
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Tipsv2DptConfig
    HF_MODEL_TYPE = "tipsv2_dpt"
    HUB_REPO_SIBLINGS = TIPSV2_DPT_HUB_SIBLINGS

    @classmethod
    def _release_warm_start_cls(cls):
        return Tipsv2DptDensePredict

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        warm = cls._release_warm_start_cls()
        if cls is warm:
            return super().from_hub_repo(
                repo_id,
                load_weights=load_weights,
                skip_mismatch=skip_mismatch,
                **kwargs,
            )
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = warm.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        bc = hf_config["backbone_config"]
        num_labels = hf_config.get("num_labels")
        if not num_labels:
            id2label = hf_config.get("id2label")
            num_labels = len(id2label) if id2label else 150
        return {
            "image_size": bc.get("image_size", 448),
            "patch_size": bc.get("patch_size", 14),
            "num_register_tokens": bc.get("num_register_tokens", 1),
            "vision_hidden_dim": bc.get("hidden_size", 1152),
            "vision_num_layers": bc.get("num_hidden_layers", 27),
            "vision_num_heads": bc.get("num_attention_heads", 16),
            "vision_mlp_ratio": bc.get("mlp_ratio", 3.7361111111111112),
            "vision_use_swiglu_ffn": bc.get("use_swiglu_ffn", False),
            "vision_layerscale_value": bc.get("layerscale_value", 1.0),
            "vision_layer_norm_eps": bc.get("layer_norm_eps", 1e-6),
            "out_indices": list(bc.get("out_indices", DEFAULT_OUT_INDICES)),
            "neck_hidden_sizes": list(
                hf_config.get("neck_hidden_sizes", DEFAULT_NECK_HIDDEN_SIZES)
            ),
            "reassemble_factors": list(
                hf_config.get("reassemble_factors", DEFAULT_REASSEMBLE_FACTORS)
            ),
            "fusion_hidden_size": hf_config.get("fusion_hidden_size", 256),
            "num_depth_bins": hf_config.get("num_depth_bins", 256),
            "min_depth": hf_config.get("min_depth", 0.001),
            "max_depth": hf_config.get("max_depth", 10.0),
            "depth_decoder_activation": hf_config.get(
                "depth_decoder_activation", "relu"
            ),
            "num_labels": num_labels,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_tipsv2_dpt_hf_to_keras import transfer_tipsv2_dpt_weights

        transfer_tipsv2_dpt_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_size=448,
        patch_size=14,
        num_register_tokens=1,
        vision_hidden_dim=1152,
        vision_num_layers=27,
        vision_num_heads=16,
        vision_mlp_ratio=3.7361111111111112,
        vision_use_swiglu_ffn=False,
        vision_layerscale_value=1.0,
        vision_layer_norm_eps=1e-6,
        out_indices=None,
        neck_hidden_sizes=None,
        reassemble_factors=None,
        fusion_hidden_size=256,
        num_depth_bins=256,
        min_depth=0.001,
        max_depth=10.0,
        depth_decoder_activation="relu",
        num_labels=150,
        input_tensor=None,
        name=None,
        **kwargs,
    ):
        out_indices = list(out_indices or DEFAULT_OUT_INDICES)
        neck_hidden_sizes = list(neck_hidden_sizes or DEFAULT_NECK_HIDDEN_SIZES)
        reassemble_factors = list(reassemble_factors or DEFAULT_REASSEMBLE_FACTORS)
        name = name or type(self).__name__

        data_format = keras.config.image_data_format()
        input_shape = standardize_input_shape(image_size, data_format)
        image_size = (
            input_shape[0] if data_format == "channels_last" else input_shape[1]
        )
        grid = image_size // patch_size

        if input_tensor is None:
            images = layers.Input(shape=input_shape, name="images")
        elif not utils.is_keras_tensor(input_tensor):
            images = layers.Input(tensor=input_tensor, shape=input_shape, name="images")
        else:
            images = input_tensor

        features = backbone_features(
            images,
            embed_dim=vision_hidden_dim,
            depth=vision_num_layers,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            layerscale_value=vision_layerscale_value,
            use_swiglu=vision_use_swiglu_ffn,
            layer_norm_eps=vision_layer_norm_eps,
            out_indices=out_indices,
            image_size=image_size,
            data_format=data_format,
        )
        neck_kw = {
            "hidden": vision_hidden_dim,
            "grid": grid,
            "neck_hidden_sizes": neck_hidden_sizes,
            "reassemble_factors": reassemble_factors,
            "fusion_hidden_size": fusion_hidden_size,
            "num_register_tokens": num_register_tokens,
            "data_format": data_format,
        }
        outputs = {}
        for task in self.TASKS:
            key, tensor = task_output(
                features,
                task,
                neck_kw=neck_kw,
                fusion_hidden_size=fusion_hidden_size,
                num_depth_bins=num_depth_bins,
                min_depth=min_depth,
                max_depth=max_depth,
                depth_decoder_activation=depth_decoder_activation,
                num_labels=num_labels,
                data_format=data_format,
            )
            outputs[key] = tensor

        super().__init__(inputs=images, outputs=outputs, name=name, **kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_use_swiglu_ffn = vision_use_swiglu_ffn
        self.vision_layerscale_value = vision_layerscale_value
        self.vision_layer_norm_eps = vision_layer_norm_eps
        self.out_indices = out_indices
        self.neck_hidden_sizes = neck_hidden_sizes
        self.reassemble_factors = reassemble_factors
        self.fusion_hidden_size = fusion_hidden_size
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_decoder_activation = depth_decoder_activation
        self.num_labels = num_labels
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "vision_use_swiglu_ffn": self.vision_use_swiglu_ffn,
                "vision_layerscale_value": self.vision_layerscale_value,
                "vision_layer_norm_eps": self.vision_layer_norm_eps,
                "out_indices": self.out_indices,
                "neck_hidden_sizes": self.neck_hidden_sizes,
                "reassemble_factors": self.reassemble_factors,
                "fusion_hidden_size": self.fusion_hidden_size,
                "num_depth_bins": self.num_depth_bins,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "depth_decoder_activation": self.depth_decoder_activation,
                "num_labels": self.num_labels,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2DptDensePredict(Tipsv2DptBase):
    """TIPSv2-DPT dense prediction: depth + semantic segmentation.

    One shared TIPSv2 vision backbone feeds two independent DPT necks + decoders in a
    single forward pass. Input pixels in ``[0, 1]``. Output dict: ``predicted_depth``
    ``(B, H', W')`` (meters) and ``segmentation_logits`` ``(B, H', W', num_labels)`` at
    the DPT feature resolution.

    References:
    - [TIPSv2](https://huggingface.co/papers/2604.12012)
    """

    TASKS = ("depth", "segmentation")


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2DptDepthEstimation(Tipsv2DptBase):
    """TIPSv2-DPT monocular depth head. Output dict: ``predicted_depth`` ``(B, H', W')``."""

    TASKS = ("depth",)


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2DptSemanticSegment(Tipsv2DptBase):
    """TIPSv2-DPT semantic-segmentation head. Output dict: ``segmentation_logits``."""

    TASKS = ("segmentation",)
