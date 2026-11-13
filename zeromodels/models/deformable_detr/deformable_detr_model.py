import math

import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.base.base_model import hf_num_classes
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.detr.detr_model import detr_backbone
from zeromodels.utils import standardize_input_shape

from .deformable_detr_config import DeformableDetrConfig
from .deformable_detr_layers import (
    DeformableDetrDecoderLayer,
    DeformableDetrEncoderLayer,
    DeformableDetrMLP,
    inverse_sigmoid,
)

# The backbone (DeformableDetrModel) and the detector (DeformableDETRDetect) share
# one weights repo, whose zm_config.json declares the detector.
DEFORMABLE_DETR_HUB_SIBLINGS = frozenset(
    {"DeformableDetrModel", "DeformableDETRDetect"}
)


def level_sine_position_embedding(height, width, d_model, temperature=10000):
    """Unpadded 2D sine embedding of one feature level -> ``(1, H * W, d_model)``.

    Deformable DETR's variant: positions are taken at pixel centers
    (``(i - 0.5) / n``) before scaling by ``2 * pi``; ``[pos_y, pos_x]`` order.
    """
    num_feats = d_model // 2
    scale = 2 * math.pi
    y_embed = (ops.arange(1, height + 1, dtype="float32") - 0.5) / (height + 1e-6)
    x_embed = (ops.arange(1, width + 1, dtype="float32") - 0.5) / (width + 1e-6)
    dim_t = ops.arange(num_feats, dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_feats)

    def encode(values, length):
        e = values[:, None] * scale / dim_t
        e = ops.stack([ops.sin(e[:, 0::2]), ops.cos(e[:, 1::2])], axis=-1)
        return ops.reshape(e, (length, num_feats))

    pos_y = ops.broadcast_to(
        encode(y_embed, height)[:, None], (height, width, num_feats)
    )
    pos_x = ops.broadcast_to(
        encode(x_embed, width)[None, :], (height, width, num_feats)
    )
    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    return ops.reshape(pos, (1, height * width, d_model))


def proposal_position_embedding(proposals, d_model, temperature=10000):
    """Sine embedding of ``(B, Q, 4)`` proposal logits -> ``(B, Q, 2 * d_model)``."""
    num_feats = d_model // 2
    dim_t = ops.arange(num_feats, dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_feats)
    proposals = ops.sigmoid(ops.cast(proposals, "float32")) * (2 * math.pi)
    pos = proposals[..., None] / dim_t  # (B, Q, 4, num_feats)
    pos = ops.stack([ops.sin(pos[..., 0::2]), ops.cos(pos[..., 1::2])], axis=-1)
    b, q = ops.shape(pos)[0], ops.shape(pos)[1]
    return ops.reshape(pos, (b, q, 4 * num_feats))


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrCore(layers.Layer):
    """Imperative Deformable DETR transformer over the projected feature levels.

    Wrapped by the functional :class:`DeformableDetrModel`: the ResNet backbone
    and the per-level input projections are functional, while the transformer
    runs here, eagerly, because the multi-scale sampling and the reference points
    depend on each level's concrete spatial size. ``build`` creates every weight
    up front and ``compute_output_spec`` keeps the dynamic control flow out of the
    functional trace, so the model is fully built at construction and accepts any
    input resolution.

    The flattened levels (plus a level-aware sine position embedding) pass
    through the deformable encoder; the decoder then refines ``num_queries``
    object queries, each attending to sampled points around a reference point
    predicted from its learned position embedding (single-stage) or to the top
    encoder proposals (``two_stage``). With ``with_box_refine`` every decoder
    layer also refines the reference boxes.
    """

    def __init__(
        self,
        d_model=256,
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        num_queries=300,
        with_box_refine=False,
        two_stage=False,
        dropout_rate=0.1,
        num_classes=91,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if two_stage and not with_box_refine:
            raise ValueError("two_stage=True requires with_box_refine=True.")
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.num_queries = num_queries
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.data_format = keras.config.image_data_format()
        # Prediction heads: one per decoder layer (plus one for the encoder
        # proposals when two-stage), or a single shared head otherwise.
        self.num_pred = decoder_layers + 1 if two_stage else decoder_layers

        self.encoder_layers_list = [
            DeformableDetrEncoderLayer(
                d_model,
                encoder_attention_heads,
                encoder_ffn_dim,
                num_feature_levels,
                encoder_n_points,
                dropout_rate,
                name=f"encoder_layer_{i}",
            )
            for i in range(encoder_layers)
        ]
        self.decoder_layers_list = [
            DeformableDetrDecoderLayer(
                d_model,
                decoder_attention_heads,
                decoder_ffn_dim,
                num_feature_levels,
                decoder_n_points,
                dropout_rate,
                name=f"decoder_layer_{i}",
            )
            for i in range(decoder_layers)
        ]
        if two_stage:
            self.enc_output = layers.Dense(d_model, name="enc_output")
            self.enc_output_norm = layers.LayerNormalization(
                epsilon=1e-5, name="enc_output_norm"
            )
            self.pos_trans = layers.Dense(2 * d_model, name="pos_trans")
            self.pos_trans_norm = layers.LayerNormalization(
                epsilon=1e-5, name="pos_trans_norm"
            )
        else:
            self.reference_points_proj = layers.Dense(2, name="reference_points")
        # The decoder itself uses per-layer box heads (box refinement) and the
        # proposal class head (two-stage); the detector reuses them.
        self.bbox_embed = (
            [self._box_head(i) for i in range(self.num_pred)] if with_box_refine else []
        )
        self.class_embed = (
            [self._class_head(i) for i in range(self.num_pred)] if two_stage else []
        )

    def _box_head(self, i):
        return DeformableDetrMLP(self.d_model, 4, 3, name=f"bbox_embed_{i}")

    def _class_head(self, i):
        return layers.Dense(self.num_classes, name=f"class_embed_{i}")

    def build(self, input_shape=None):
        d = self.d_model
        self.level_embed = self.add_weight(
            name="level_embed",
            shape=(self.num_feature_levels, d),
            initializer="zeros",
        )
        if self.two_stage:
            self.enc_output.build((None, d))
            self.enc_output_norm.build((None, d))
            self.pos_trans.build((None, 2 * d))
            self.pos_trans_norm.build((None, 2 * d))
        else:
            self.query_position_embeddings = self.add_weight(
                name="query_position_embeddings",
                shape=(self.num_queries, 2 * d),
                initializer="random_normal",
            )
            self.reference_points_proj.build((None, d))
        for layer in self.encoder_layers_list + self.decoder_layers_list:
            layer.build()
        for head in self.bbox_embed:
            head.build()
        for head in self.class_embed:
            head.build((None, d))
        self.built = True

    def flatten_levels(self, sources):
        if self.data_format == "channels_first":
            sources = [ops.transpose(s, (0, 2, 3, 1)) for s in sources]
        spatial_shapes = [(int(s.shape[1]), int(s.shape[2])) for s in sources]
        batch = ops.shape(sources[0])[0]
        flat, positions = [], []
        for level, (h, w) in enumerate(spatial_shapes):
            flat.append(ops.reshape(sources[level], (batch, h * w, self.d_model)))
            pos = level_sine_position_embedding(h, w, self.d_model)
            positions.append(
                ops.cast(pos, flat[-1].dtype) + self.level_embed[level][None, None]
            )
        memory = ops.concatenate(flat, axis=1)
        position = ops.broadcast_to(
            ops.concatenate(positions, axis=1), ops.shape(memory)
        )
        return memory, position, spatial_shapes

    def encoder_reference_points(self, spatial_shapes, batch):
        refs = []
        for h, w in spatial_shapes:
            ref_y, ref_x = ops.meshgrid(
                ops.linspace(0.5, h - 0.5, h),
                ops.linspace(0.5, w - 0.5, w),
                indexing="ij",
            )
            refs.append(
                ops.stack(
                    [ops.reshape(ref_x, (-1,)) / w, ops.reshape(ref_y, (-1,)) / h],
                    axis=-1,
                )
            )
        ref = ops.cast(ops.concatenate(refs, axis=0), "float32")
        ref = ops.repeat(ref[None, :, None, :], self.num_feature_levels, axis=2)
        return ops.broadcast_to(ref, (batch,) + tuple(ref.shape[1:]))

    def encoder_proposals(self, memory, spatial_shapes):
        proposals = []
        for level, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = ops.meshgrid(
                ops.linspace(0.0, h - 1.0, h),
                ops.linspace(0.0, w - 1.0, w),
                indexing="ij",
            )
            grid = ops.stack(
                [ops.reshape(grid_x, (-1,)), ops.reshape(grid_y, (-1,))], axis=-1
            )
            grid = (ops.cast(grid, "float32") + 0.5) / ops.convert_to_tensor(
                [[w, h]], dtype="float32"
            )
            size = ops.ones_like(grid) * (0.05 * (2.0**level))
            proposals.append(ops.concatenate([grid, size], axis=-1))
        proposals = ops.concatenate(proposals, axis=0)[None]
        valid = ops.all((proposals > 0.01) & (proposals < 0.99), axis=-1, keepdims=True)
        proposals = ops.log(proposals / (1.0 - proposals))
        proposals = ops.where(valid, proposals, float("inf"))
        object_query = ops.where(valid, memory, ops.zeros_like(memory))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, proposals

    def decoder_inputs(self, memory, spatial_shapes):
        batch = ops.shape(memory)[0]
        d = self.d_model
        if self.two_stage:
            object_query, proposals = self.encoder_proposals(memory, spatial_shapes)
            scores = self.class_embed[-1](object_query)[..., 0]
            coord_logits = self.bbox_embed[-1](object_query) + ops.cast(
                proposals, object_query.dtype
            )
            topk = ops.top_k(scores, self.num_queries)[1]
            topk_coords = ops.stop_gradient(
                ops.take_along_axis(
                    coord_logits, ops.repeat(topk[..., None], 4, axis=-1), axis=1
                )
            )
            reference_points = ops.sigmoid(topk_coords)
            pos = proposal_position_embedding(topk_coords, d)
            pos = self.pos_trans_norm(self.pos_trans(ops.cast(pos, memory.dtype)))
            query_position, target = pos[..., :d], pos[..., d:]
        else:
            embed = ops.cast(self.query_position_embeddings, memory.dtype)
            shape = (batch, self.num_queries, d)
            query_position = ops.broadcast_to(embed[None, :, :d], shape)
            target = ops.broadcast_to(embed[None, :, d:], shape)
            reference_points = ops.sigmoid(self.reference_points_proj(query_position))
        return target, query_position, reference_points

    def run_decoder(
        self, target, query_position, reference_points, memory, spatial_shapes, training
    ):
        hidden = target
        intermediate, intermediate_refs = [], []
        for idx, layer in enumerate(self.decoder_layers_list):
            ref_input = ops.repeat(
                reference_points[:, :, None], self.num_feature_levels, axis=2
            )
            hidden = layer(
                hidden,
                query_position,
                ref_input,
                memory,
                spatial_shapes=spatial_shapes,
                training=training,
            )
            if self.with_box_refine:
                delta = self.bbox_embed[idx](hidden)
                reference_points = ops.stop_gradient(
                    ops.sigmoid(self.add_reference(delta, reference_points))
                )
            intermediate.append(hidden)
            intermediate_refs.append(reference_points)
        return hidden, intermediate, intermediate_refs

    @staticmethod
    def add_reference(delta, reference):
        """``delta + logit(reference)``; a 2D reference shifts only the center."""
        reference = ops.cast(inverse_sigmoid(reference), delta.dtype)
        if int(reference.shape[-1]) == 4:
            return delta + reference
        return ops.concatenate([delta[..., :2] + reference, delta[..., 2:]], axis=-1)

    def run_transformer(self, sources, training=None):
        memory, position, spatial_shapes = self.flatten_levels(sources)
        reference = self.encoder_reference_points(spatial_shapes, ops.shape(memory)[0])
        for layer in self.encoder_layers_list:
            memory = layer(
                memory,
                position,
                reference,
                spatial_shapes=spatial_shapes,
                training=training,
            )
        target, query_position, init_reference = self.decoder_inputs(
            memory, spatial_shapes
        )
        hidden, intermediate, intermediate_refs = self.run_decoder(
            target, query_position, init_reference, memory, spatial_shapes, training
        )
        return hidden, intermediate, intermediate_refs, init_reference

    def call(self, sources, training=None):
        hidden, intermediate, _, _ = self.run_transformer(sources, training=training)
        return {
            "last_hidden_state": hidden,
            "intermediate_hidden_states": ops.stack(intermediate, axis=1),
        }

    def compute_output_spec(self, sources, training=None):
        b = sources[0].shape[0]
        return {
            "last_hidden_state": keras.KerasTensor(
                (b, self.num_queries, self.d_model), dtype=self.compute_dtype
            ),
            "intermediate_hidden_states": keras.KerasTensor(
                (b, self.decoder_layers, self.num_queries, self.d_model),
                dtype=self.compute_dtype,
            ),
        }

    def get_config(self):
        config = super().get_config()
        for key in (
            "d_model",
            "encoder_layers",
            "decoder_layers",
            "encoder_ffn_dim",
            "decoder_ffn_dim",
            "encoder_attention_heads",
            "decoder_attention_heads",
            "num_feature_levels",
            "encoder_n_points",
            "decoder_n_points",
            "num_queries",
            "with_box_refine",
            "two_stage",
            "dropout_rate",
            "num_classes",
        ):
            config[key] = getattr(self, key)
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrDetectCore(DeformableDetrCore):
    """:class:`DeformableDetrCore` plus the class and box heads.

    ``logits`` come from the last decoder layer's class head and ``pred_boxes``
    from its box head, added (in logit space) to that layer's input reference.
    Without box refinement one head is shared by every layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        heads = self.num_pred if self.with_box_refine else 1
        if not self.class_embed:
            self.class_embed = [self._class_head(i) for i in range(heads)]
        if not self.bbox_embed:
            self.bbox_embed = [self._box_head(i) for i in range(heads)]

    def build(self, input_shape=None):
        super().build(input_shape)
        for head in self.bbox_embed:
            if not head.built:
                head.build()
        for head in self.class_embed:
            if not head.built:
                head.build((None, self.d_model))

    def call(self, sources, training=None):
        hidden, _, intermediate_refs, init_reference = self.run_transformer(
            sources, training=training
        )
        level = self.decoder_layers - 1
        reference = init_reference if level == 0 else intermediate_refs[level - 1]
        head = level if len(self.class_embed) > 1 else 0
        logits = self.class_embed[head](hidden)
        delta = self.bbox_embed[level if len(self.bbox_embed) > 1 else 0](hidden)
        pred_boxes = ops.sigmoid(self.add_reference(delta, reference))
        return {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "last_hidden_state": hidden,
        }

    def compute_output_spec(self, sources, training=None):
        b = sources[0].shape[0]
        q = self.num_queries
        return {
            "logits": keras.KerasTensor(
                (b, q, self.num_classes), dtype=self.compute_dtype
            ),
            "pred_boxes": keras.KerasTensor((b, q, 4), dtype=self.compute_dtype),
            "last_hidden_state": keras.KerasTensor(
                (b, q, self.d_model), dtype=self.compute_dtype
            ),
        }


def deformable_detr_config_from_hf(hf_config):
    """Map a ``deformable_detr`` ``config.json`` to constructor kwargs."""
    if hf_config.get("dilation"):
        raise ValueError(
            "Deformable DETR DC5 checkpoints (dilation=True) are not supported: the "
            "zeromodels ResNet backbone has no dilated last stage."
        )
    if hf_config.get("position_embedding_type", "sine") != "sine":
        raise ValueError("Only the sine position embedding is supported.")
    backbone = hf_config.get("backbone") or "resnet50"
    two_stage = bool(hf_config.get("two_stage", False))
    return {
        "backbone_variant": "ResNet101" if "101" in backbone else "ResNet50",
        "d_model": hf_config["d_model"],
        "encoder_layers": hf_config["encoder_layers"],
        "decoder_layers": hf_config["decoder_layers"],
        "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
        "decoder_ffn_dim": hf_config["decoder_ffn_dim"],
        "encoder_attention_heads": hf_config["encoder_attention_heads"],
        "decoder_attention_heads": hf_config["decoder_attention_heads"],
        "num_feature_levels": hf_config["num_feature_levels"],
        "encoder_n_points": hf_config["encoder_n_points"],
        "decoder_n_points": hf_config["decoder_n_points"],
        "num_queries": hf_config[
            "two_stage_num_proposals" if two_stage else "num_queries"
        ],
        "with_box_refine": bool(hf_config.get("with_box_refine", False)),
        "two_stage": two_stage,
        "dropout_rate": hf_config.get("dropout", 0.1),
        "num_classes": hf_num_classes(hf_config),
    }


def deformable_detr_sources(inputs, backbone_variant, d_model, num_feature_levels):
    """ResNet backbone + per-level input projections (functional graph).

    Levels come from the stride-8 / 16 / 32 ResNet stages (or only stride 32 for
    a single level), each projected to ``d_model`` by a 1x1 conv + GroupNorm;
    extra levels are stride-2 3x3 convs (+ GroupNorm) on top.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1
    stages = detr_backbone(
        inputs,
        backbone_variant=backbone_variant,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    features = list(stages[1:]) if num_feature_levels > 1 else [stages[-1]]
    features = features[:num_feature_levels]

    def norm(i):
        return layers.GroupNormalization(
            groups=32, axis=channels_axis, epsilon=1e-5, name=f"input_proj_{i}_norm"
        )

    sources = []
    for i, feature in enumerate(features):
        x = layers.Conv2D(
            d_model, 1, data_format=data_format, name=f"input_proj_{i}_conv"
        )(feature)
        sources.append(norm(i)(x))
    for i in range(len(features), num_feature_levels):
        base = features[-1] if i == len(features) else sources[-1]
        # Explicit symmetric padding: torch's padding=1 with stride 2, which
        # Keras "same" padding does not reproduce on even sizes.
        x = layers.ZeroPadding2D(1, data_format=data_format)(base)
        x = layers.Conv2D(
            d_model,
            3,
            strides=2,
            padding="valid",
            data_format=data_format,
            name=f"input_proj_{i}_conv",
        )(x)
        sources.append(norm(i)(x))
    return sources


class _DeformableDetrFunctional(BaseModel):
    """Shared functional wrapper: ``pixel_values -> backbone + projections -> core``."""

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deformable_detr"
    config_class = DeformableDetrConfig
    HUB_REPO_SIBLINGS = DEFORMABLE_DETR_HUB_SIBLINGS
    core_class = DeformableDetrCore

    def __init__(
        self,
        backbone_variant="ResNet50",
        d_model=256,
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        num_queries=300,
        with_box_refine=False,
        two_stage=False,
        dropout_rate=0.1,
        num_classes=91,
        image_size=None,
        input_tensor=None,
        name=None,
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)
        if input_tensor is None:
            img_input = layers.Input(shape=image_size, name="pixel_values")
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        core = self.core_class(
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            num_feature_levels=num_feature_levels,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            num_queries=num_queries,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            name="core",
        )
        sources = deformable_detr_sources(
            img_input, backbone_variant, d_model, num_feature_levels
        )
        outputs = core(sources)
        super().__init__(
            inputs=img_input,
            outputs=outputs,
            name=name or type(self).__name__,
            **kwargs,
        )
        self.core = core
        self.backbone_variant = backbone_variant
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.num_queries = num_queries
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "d_model": self.d_model,
                "encoder_layers": self.encoder_layers,
                "decoder_layers": self.decoder_layers,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "encoder_attention_heads": self.encoder_attention_heads,
                "decoder_attention_heads": self.decoder_attention_heads,
                "num_feature_levels": self.num_feature_levels,
                "encoder_n_points": self.encoder_n_points,
                "decoder_n_points": self.decoder_n_points,
                "num_queries": self.num_queries,
                "with_box_refine": self.with_box_refine,
                "two_stage": self.two_stage,
                "dropout_rate": self.dropout_rate,
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
        return deformable_detr_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_deformable_detr_hf_to_keras import (
            transfer_deformable_detr_weights,
        )

        transfer_deformable_detr_weights(keras_model, hf_state_dict)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrModel(_DeformableDetrFunctional):
    """Deformable DETR backbone + deformable transformer (no detection heads).

    A ResNet-50/101 backbone yields stride-8 / 16 / 32 feature maps (plus an
    extra stride-64 level), each projected to ``d_model``. The encoder replaces
    dense self-attention with multi-scale deformable attention (every token
    attends to a few learned sampling points per level around itself), which
    makes high-resolution, multi-scale features affordable and trains about 10x
    faster than DETR. The decoder refines object queries with self-attention and
    deformable cross-attention around per-query reference points. Returns
    ``last_hidden_state`` ``(B, num_queries, d_model)`` and
    ``intermediate_hidden_states`` ``(B, decoder_layers, num_queries, d_model)``;
    use :class:`DeformableDETRDetect` for detections.

    Reference:
        - `Deformable DETR: Deformable Transformers for End-to-End Object
          Detection <https://arxiv.org/abs/2010.04159>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        d_model: Transformer width. Defaults to ``256``.
        encoder_layers / decoder_layers: Layer counts. Default ``6``.
        encoder_ffn_dim / decoder_ffn_dim: Feed-forward widths. Default ``1024``.
        encoder_attention_heads / decoder_attention_heads: Heads. Default ``8``.
        num_feature_levels: ``4`` (multi-scale) or ``1`` (single scale).
        encoder_n_points / decoder_n_points: Sampling points. Default ``4``.
        num_queries: Object queries / two-stage proposals. Default ``300``.
        with_box_refine: Iterative box refinement. Default ``False``.
        two_stage: Two-stage proposals (needs ``with_box_refine``).
        dropout_rate: Dropout. Default ``0.1``.
        num_classes: Classes of the proposal head (two-stage). Default ``91``.
        image_size: ``None`` (dynamic, the default), an int or a tuple.
        input_tensor: Optional Keras input tensor.
        name: Model name.
    """

    core_class = DeformableDetrCore

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = DeformableDETRDetect.from_hf(hf_id, skip_mismatch=skip_mismatch)
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
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = DeformableDETRDetect.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDETRDetect(_DeformableDetrFunctional):
    """Deformable DETR object detector.

    :class:`DeformableDetrModel` plus sigmoid class heads and box heads. Returns
    ``logits`` ``(B, num_queries, num_classes)``, ``pred_boxes``
    ``(B, num_queries, 4)`` (normalized ``cx, cy, w, h``) and
    ``last_hidden_state``. Post-process with
    :meth:`DeformableDETRImageProcessor.post_process_object_detection`.

    Covers every official checkpoint family: single-scale, multi-scale, iterative
    box refinement and the two-stage variant.

    Reference:
        - `Deformable DETR: Deformable Transformers for End-to-End Object
          Detection <https://arxiv.org/abs/2010.04159>`_

    Examples:

    ```python
    model = DeformableDETRDetect.from_weights("zeromodels/deformable-detr")
    model = DeformableDETRDetect.from_weights("hf:SenseTime/deformable-detr")
    ```
    """

    core_class = DeformableDetrDetectCore
