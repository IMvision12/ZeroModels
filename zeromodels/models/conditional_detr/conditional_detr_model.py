import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.base.base_model import hf_num_classes
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.detr.detr_layers import DETRExpandQueryEmbedding
from zeromodels.models.detr.detr_model import detr_backbone, detr_encoder
from zeromodels.utils import standardize_input_shape

from .conditional_detr_config import ConditionalDetrConfig
from .conditional_detr_layers import (
    ConditionalDetrBoxDecode,
    ConditionalDetrDecoder,
    ConditionalDetrMLP,
)

# The backbone (ConditionalDetrModel) and the detector (ConditionalDETRDetect) share
# one weights repo, whose zm_config.json declares the detector.
CONDITIONAL_DETR_HUB_SIBLINGS = frozenset(
    {"ConditionalDetrModel", "ConditionalDETRDetect"}
)


def conditional_detr_config_from_hf(hf_config):
    """Map a ``conditional_detr`` ``config.json`` to constructor kwargs."""
    if hf_config.get("dilation"):
        raise ValueError(
            "Conditional DETR DC5 checkpoints (dilation=True) are not supported: the "
            "zeromodels ResNet backbone has no dilated last stage."
        )
    backbone = hf_config.get("backbone") or "resnet50"
    return {
        "backbone_variant": "ResNet101" if "101" in backbone else "ResNet50",
        "hidden_dim": hf_config["d_model"],
        "num_heads": hf_config["encoder_attention_heads"],
        "num_encoder_layers": hf_config["encoder_layers"],
        "num_decoder_layers": hf_config["decoder_layers"],
        "dim_feedforward": hf_config["encoder_ffn_dim"],
        "dropout_rate": hf_config.get("dropout", 0.1),
        "num_queries": hf_config["num_queries"],
    }


def conditional_detr_functional(
    inputs,
    backbone_variant,
    hidden_dim,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout_rate,
    num_queries,
):
    """Build the Conditional DETR graph: ResNet -> DETR encoder -> conditional decoder.

    Returns ``(last_hidden_state, reference_points)``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    backbone_features = detr_backbone(
        inputs,
        backbone_variant=backbone_variant,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    # The encoder is DETR's (post-norm self-attention with the sine spatial
    # embedding added to queries and keys), so it is reused as-is.
    encoder_output, spatial_position, _ = detr_encoder(
        backbone_features[-1],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
    )
    query_position = DETRExpandQueryEmbedding(
        num_queries, hidden_dim, name="query_position_embeddings"
    )(encoder_output)
    return ConditionalDetrDecoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        name="decoder",
    )(query_position, encoder_output, spatial_position)


def _image_input(image_size, input_tensor):
    data_format = keras.config.image_data_format()
    # None -> dynamic (None, None, 3): the reference resizes aspect-preserving to
    # variable sizes, which this size-agnostic graph accepts.
    image_size = standardize_input_shape(image_size, data_format)
    if input_tensor is None:
        return layers.Input(shape=image_size), image_size
    if not utils.is_keras_tensor(input_tensor):
        return layers.Input(tensor=input_tensor, shape=image_size), image_size
    return input_tensor, image_size


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDetrModel(BaseModel):
    """Conditional DETR backbone + transformer encoder / decoder (no heads).

    A ResNet-50/101 backbone, the DETR post-norm transformer encoder over the
    stride-32 feature map with a 2D sine position embedding, and the Conditional
    DETR decoder: each object query predicts a 2D reference point, and its
    cross-attention concatenates a spatial query (the sine embedding of that point,
    scaled by a content-dependent transformation) with the content query, so each
    head attends to a spatial band around the object. That decoupling converges
    much faster than DETR. Outputs ``last_hidden_state``
    ``(B, num_queries, hidden_dim)`` and ``reference_points``
    ``(B, num_queries, 2)``; use :class:`ConditionalDETRDetect` for detections.

    Reference:
        - `Conditional DETR for Fast Training Convergence
          <https://arxiv.org/abs/2108.06152>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        hidden_dim: Transformer width. Defaults to ``256``.
        num_heads: Attention heads. Defaults to ``8``.
        num_encoder_layers: Encoder layers. Defaults to ``6``.
        num_decoder_layers: Decoder layers. Defaults to ``6``.
        dim_feedforward: Feed-forward width. Defaults to ``2048``.
        dropout_rate: Dropout rate. Defaults to ``0.1``.
        num_queries: Object queries. Defaults to ``300``.
        image_size: ``None`` (dynamic, the default), an int ``N`` or an
            ``(H, W)`` / ``(H, W, C)`` tuple.
        input_tensor: Optional Keras tensor to use as the input.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "conditional_detr"
    config_class = ConditionalDetrConfig
    HUB_REPO_SIBLINGS = CONDITIONAL_DETR_HUB_SIBLINGS

    def __init__(
        self,
        backbone_variant="ResNet50",
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=300,
        image_size=None,
        input_tensor=None,
        name="ConditionalDetrModel",
        **kwargs,
    ):
        kwargs.pop("num_classes", None)
        img_input, image_size = _image_input(image_size, input_tensor)
        last_hidden_state, reference_points = conditional_detr_functional(
            img_input,
            backbone_variant=backbone_variant,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
        )
        super().__init__(
            inputs=img_input,
            outputs={
                "last_hidden_state": last_hidden_state,
                "reference_points": reference_points,
            },
            name=name,
            **kwargs,
        )
        self.backbone_variant = backbone_variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        return conditional_detr_config_from_hf(hf_config)

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = ConditionalDETRDetect.from_hf(hf_id, skip_mismatch=skip_mismatch)
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not matched "
                    f"from the {type(src).__name__} checkpoint: {unmatched[:5]}"
                )
            del src
        return model

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # The backbone shares the detector's weights repo: build it from the repo's
        # zm_config, then copy the matching weights across.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = ConditionalDETRDetect.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDETRDetect(BaseModel):
    """Conditional DETR object detector.

    :class:`ConditionalDetrModel` plus a linear classifier (sigmoid scores, no
    "no-object" class) and a 3-layer box MLP whose center is predicted as an offset
    from each query's reference point. Returns ``logits``
    ``(B, num_queries, num_classes)`` and ``pred_boxes`` ``(B, num_queries, 4)``
    (normalized ``cx, cy, w, h``). Post-process with
    :meth:`ConditionalDETRImageProcessor.post_process_object_detection`.

    Reference:
        - `Conditional DETR for Fast Training Convergence
          <https://arxiv.org/abs/2108.06152>`_

    Args:
        num_classes: Number of classes. Defaults to ``91`` (COCO).
        Other arguments: see :class:`ConditionalDetrModel`.

    Examples:

    ```python
    model = ConditionalDETRDetect.from_weights("zeromodels/conditional-detr-resnet-50")
    model = ConditionalDETRDetect.from_weights("hf:microsoft/conditional-detr-resnet-50")
    ```
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "conditional_detr"
    config_class = ConditionalDetrConfig
    HUB_REPO_SIBLINGS = CONDITIONAL_DETR_HUB_SIBLINGS

    def __init__(
        self,
        backbone_variant="ResNet50",
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=300,
        num_classes=91,
        image_size=None,
        input_tensor=None,
        name="ConditionalDETRDetect",
        **kwargs,
    ):
        img_input, image_size = _image_input(image_size, input_tensor)
        last_hidden_state, reference_points = conditional_detr_functional(
            img_input,
            backbone_variant=backbone_variant,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
        )
        logits = layers.Dense(num_classes, name="class_labels_classifier")(
            last_hidden_state
        )
        box_delta = ConditionalDetrMLP(hidden_dim, 4, 3, name="bbox_predictor")(
            last_hidden_state
        )
        pred_boxes = ConditionalDetrBoxDecode(name="box_decode")(
            box_delta, reference_points
        )
        super().__init__(
            inputs=img_input,
            outputs={
                "logits": logits,
                "pred_boxes": pred_boxes,
                "last_hidden_state": last_hidden_state,
            },
            name=name,
            **kwargs,
        )
        self.backbone_variant = backbone_variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        kwargs = conditional_detr_config_from_hf(hf_config)
        kwargs["num_classes"] = hf_num_classes(hf_config)
        return kwargs

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_conditional_detr_hf_to_keras import (
            transfer_conditional_detr_weights,
        )

        transfer_conditional_detr_weights(keras_model, hf_state_dict)
