import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_mixin import inference_scope

from .grounding_dino_config import GroundingDinoConfig
from .grounding_dino_layers import (
    GroundingDinoContrastiveEmbedding,
    GroundingDinoDecoderLayer,
    GroundingDinoEncoderLayer,
    GroundingDinoMLPPredictionHead,
    encode_sine_position,
)
from .grounding_dino_swin import GroundingDinoSwinBackbone
from .grounding_dino_text import GroundingDinoTextModel

MASK_NEG = -1e9
SPECIAL_TOKENS = (101, 102, 1012, 1029)  # [CLS] [SEP] . ?


def special_token_masks_and_positions(input_ids):
    """Host-side block self-attention mask + per-token position ids.

    Tokens between consecutive special tokens form a block that attends only
    within itself (plus self-attention); positions reset per block. Mirrors the
    reference ``generate_masks_with_special_tokens_and_transfer_map``.
    """
    ids = np.asarray(input_ids)
    batch, seq = ids.shape
    special = np.isin(ids, np.asarray(SPECIAL_TOKENS))
    idx = np.broadcast_to(np.arange(seq), (batch, seq))
    prev_special = np.where(special, idx, -1)
    prev_special = np.maximum.accumulate(prev_special, axis=1)
    next_special = np.where(special, idx, seq)
    next_special = np.flip(np.minimum.accumulate(np.flip(next_special, 1), axis=1), 1)
    valid_block = (
        (next_special != 0) & (next_special != seq - 1) & (next_special != seq)
    )
    same = next_special[:, :, None] == next_special[:, None, :]
    attn = same & valid_block[:, None, :]
    identity = np.broadcast_to(np.eye(seq, dtype=bool), (batch, seq, seq))
    attn = identity | attn
    position_ids = idx - prev_special - 1
    position_ids = np.where(valid_block, position_ids, 0)
    position_ids = np.clip(position_ids, 0, None).astype("int64")
    return attn.astype("float32"), position_ids


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoCore(layers.Layer):
    """Imperative Grounding DINO core (backbone + cross-modality encoder-decoder).

    Holds every sublayer and runs the full detection forward eagerly. Wrapped by
    the functional :class:`GroundingDinoModel` (a graph of ``inputs -> this core ->
    outputs``); ``compute_output_spec`` keeps the dynamic spatial control flow and
    host-side text masking out of the functional-graph trace, so the core's
    ``call`` runs only at (eager) runtime with concrete shapes. No head.

    A Swin vision backbone and a BERT text encoder feed a 6-layer deformable
    encoder that fuses vision and text (bi-directional cross-attention + text
    self-attention + image deformable self-attention). Two-stage query selection
    picks the top-``num_queries`` encoder proposals (contrastive vision-text
    scoring) as decoder reference points; the 6-layer decoder runs self-attention,
    text cross-attention and image deformable cross-attention. Returns the decoder
    hidden states; use :class:`GroundingDinoDetect` for logits / boxes.
    """

    def __init__(
        self,
        d_model=256,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        num_queries=900,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        max_text_len=256,
        query_dim=4,
        two_stage=True,
        positional_embedding_temperature=20.0,
        layer_norm_eps=1e-5,
        activation_function="relu",
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_num_heads=(3, 6, 12, 24),
        backbone_window_size=7,
        backbone_out_indices=(2, 3, 4),
        text_vocab_size=30522,
        text_hidden_size=768,
        text_num_layers=12,
        text_num_heads=12,
        text_intermediate_size=3072,
        text_max_position_embeddings=512,
        text_layer_norm_eps=1e-12,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.max_text_len = max_text_len
        self.query_dim = query_dim
        self.two_stage = two_stage
        self.positional_embedding_temperature = positional_embedding_temperature
        self.layer_norm_eps = layer_norm_eps
        self.activation_function = activation_function
        self.backbone_embed_dim = backbone_embed_dim
        self.backbone_depths = tuple(backbone_depths)
        self.backbone_num_heads = tuple(backbone_num_heads)
        self.backbone_window_size = backbone_window_size
        self.backbone_out_indices = tuple(backbone_out_indices)
        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_intermediate_size = text_intermediate_size
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_layer_norm_eps = text_layer_norm_eps
        self.data_format = keras.config.image_data_format()

        self.backbone = GroundingDinoSwinBackbone(
            embed_dim=backbone_embed_dim,
            depths=backbone_depths,
            num_heads=backbone_num_heads,
            window_size=backbone_window_size,
            out_indices=backbone_out_indices,
            data_format=self.data_format,
            name="backbone",
        )
        self.text_backbone = GroundingDinoTextModel(
            text_vocab_size,
            text_hidden_size,
            text_num_layers,
            text_num_heads,
            text_intermediate_size,
            text_max_position_embeddings,
            text_layer_norm_eps,
            name="text_backbone",
        )
        self.text_projection = layers.Dense(d_model, name="text_projection")

        n_backbone = len(backbone_out_indices)
        self.input_proj_conv = []
        self.input_proj_norm = []
        gn_axis = 1 if self.data_format == "channels_first" else -1
        for i in range(num_feature_levels):
            if i < n_backbone:
                conv = layers.Conv2D(
                    d_model,
                    1,
                    data_format=self.data_format,
                    name=f"input_proj_{i}_conv",
                )
            else:
                conv = layers.Conv2D(
                    d_model,
                    3,
                    strides=2,
                    padding="same",
                    data_format=self.data_format,
                    name=f"input_proj_{i}_conv",
                )
            self.input_proj_conv.append(conv)
            self.input_proj_norm.append(
                layers.GroupNormalization(
                    groups=32, axis=gn_axis, epsilon=1e-5, name=f"input_proj_{i}_norm"
                )
            )

        self.encoder_layers_list = [
            GroundingDinoEncoderLayer(
                d_model,
                encoder_attention_heads,
                encoder_ffn_dim,
                num_feature_levels,
                encoder_n_points,
                layer_norm_eps,
                name=f"encoder_layer_{i}",
            )
            for i in range(encoder_layers)
        ]
        self.decoder_layers_list = [
            GroundingDinoDecoderLayer(
                d_model,
                decoder_attention_heads,
                decoder_ffn_dim,
                num_feature_levels,
                decoder_n_points,
                layer_norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(decoder_layers)
        ]
        self.decoder_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="decoder_norm"
        )
        self.reference_points_head = GroundingDinoMLPPredictionHead(
            d_model, d_model, 2, name="reference_points_head"
        )
        self.query_position_embeddings = layers.Embedding(
            num_queries, d_model, name="query_position_embeddings"
        )
        if two_stage:
            self.enc_output = layers.Dense(d_model, name="enc_output")
            self.enc_output_norm = layers.LayerNormalization(
                epsilon=layer_norm_eps, name="enc_output_norm"
            )
            self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                d_model, 4, 3, name="encoder_output_bbox_embed"
            )
            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(
                max_text_len, name="encoder_output_class_embed"
            )

    def build(self, input_shape):
        self.level_embed = self.add_weight(
            name="level_embed",
            shape=(self.num_feature_levels, self.d_model),
            initializer="zeros",
            trainable=True,
        )
        self.query_position_embeddings.build((1, 1))
        self.built = True

    def sine_position_embedding(self, height, width):
        # No-padding sine 2D position embedding -> (1, H, W, d_model).
        dim = self.d_model // 2
        temp = self.positional_embedding_temperature
        scale = 2 * math.pi
        y_embed = ops.cast(ops.arange(1, height + 1), "float32")
        x_embed = ops.cast(ops.arange(1, width + 1), "float32")
        y_embed = y_embed / (height + 1e-6) * scale
        x_embed = x_embed / (width + 1e-6) * scale
        dim_t = ops.arange(dim, dtype="float32")
        dim_t = temp ** (2 * ops.floor(dim_t / 2) / dim)
        pos_x = x_embed[:, None] / dim_t  # (W, dim)
        pos_y = y_embed[:, None] / dim_t  # (H, dim)
        pos_x = ops.reshape(
            ops.stack([ops.sin(pos_x[:, 0::2]), ops.cos(pos_x[:, 1::2])], axis=-1),
            (width, dim),
        )
        pos_y = ops.reshape(
            ops.stack([ops.sin(pos_y[:, 0::2]), ops.cos(pos_y[:, 1::2])], axis=-1),
            (height, dim),
        )
        pos_x = ops.broadcast_to(pos_x[None, :, :], (height, width, dim))
        pos_y = ops.broadcast_to(pos_y[:, None, :], (height, width, dim))
        pos = ops.concatenate([pos_y, pos_x], axis=-1)  # (H, W, d_model)
        return pos[None]

    def get_reference_points(self, spatial_shapes_list):
        # No-padding reference points -> (1, sum(HW), num_levels, 2).
        ref_list = []
        for h, w in spatial_shapes_list:
            ref_y, ref_x = ops.meshgrid(
                ops.linspace(0.5, h - 0.5, h),
                ops.linspace(0.5, w - 0.5, w),
                indexing="ij",
            )
            ref_y = ops.reshape(ref_y, (-1,)) / h
            ref_x = ops.reshape(ref_x, (-1,)) / w
            ref_list.append(ops.stack([ref_x, ref_y], axis=-1))
        ref = ops.concatenate(ref_list, axis=0)  # (sum(HW), 2)
        ref = ops.cast(ref, "float32")[None, :, None, :]
        return ops.broadcast_to(ref, (1, int(ref.shape[1]), self.num_feature_levels, 2))

    def encode_text(self, input_ids, attention_mask):
        ids = np.asarray(ops.convert_to_numpy(ops.convert_to_tensor(input_ids))).astype(
            "int64"
        )
        block_mask, position_ids = special_token_masks_and_positions(ids)
        if block_mask.shape[1] > self.max_text_len:
            block_mask = block_mask[:, : self.max_text_len, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            ids = ids[:, : self.max_text_len]
        additive = (1.0 - ops.convert_to_tensor(block_mask))[:, None] * MASK_NEG
        token_type = ops.zeros_like(ops.convert_to_tensor(ids))
        text = self.text_backbone(
            ops.convert_to_tensor(ids),
            additive,
            token_type,
            ops.convert_to_tensor(position_ids),
        )
        text = self.text_projection(text)
        return text, additive

    def backbone_features(self, pixel_values):
        feats = self.backbone(pixel_values)  # list of (B, H, W, C)
        sources = []
        for i in range(len(feats)):
            x = self.input_proj_norm[i](self.input_proj_conv[i](feats[i]))
            sources.append(x)
        for i in range(len(feats), self.num_feature_levels):
            base = feats[-1] if i == len(feats) else sources[-1]
            x = self.input_proj_norm[i](self.input_proj_conv[i](base))
            sources.append(x)
        return sources

    def run_encoder(self, sources, text_features, text_additive):
        # The deformable encoder works on channels-last (B, H*W, C) sequences;
        # normalize channels-first projections back before flattening.
        if self.data_format == "channels_first":
            sources = [ops.transpose(s, (0, 2, 3, 1)) for s in sources]
        spatial_shapes_list = [(int(s.shape[1]), int(s.shape[2])) for s in sources]
        flat_sources = []
        flat_pos = []
        for level, s in enumerate(sources):
            h, w = spatial_shapes_list[level]
            pos = self.sine_position_embedding(h, w)
            pos = (
                ops.reshape(pos, (1, h * w, self.d_model))
                + self.level_embed[level][None, None]
            )
            flat_sources.append(ops.reshape(s, (ops.shape(s)[0], h * w, self.d_model)))
            flat_pos.append(ops.broadcast_to(pos, ops.shape(flat_sources[-1])))
        vision = ops.concatenate(flat_sources, axis=1)
        vision_pos = ops.concatenate(flat_pos, axis=1)
        reference_points = self.get_reference_points(spatial_shapes_list)
        reference_points = ops.broadcast_to(
            reference_points,
            (ops.shape(vision)[0],) + tuple(reference_points.shape[1:]),
        )
        text_self_mask = text_additive  # additive (B,1,T,T)
        for layer in self.encoder_layers_list:
            vision, text_features = layer(
                vision,
                text_features,
                vision_pos,
                reference_points,
                spatial_shapes_list=spatial_shapes_list,
                text_position_embedding=None,
                text_self_attention_mask=text_self_mask,
            )
        return vision, text_features, spatial_shapes_list

    def generate_proposals(self, enc_output, spatial_shapes_list):
        proposals = []
        for level, (h, w) in enumerate(spatial_shapes_list):
            grid_y, grid_x = ops.meshgrid(
                ops.arange(h, dtype="float32"),
                ops.arange(w, dtype="float32"),
                indexing="ij",
            )
            grid = ops.stack(
                [ops.reshape(grid_x, (-1,)), ops.reshape(grid_y, (-1,))], axis=-1
            )
            scale = ops.convert_to_tensor([[w, h]], dtype="float32")
            grid = (grid + 0.5) / scale
            wh = ops.ones_like(grid) * 0.05 * (2.0**level)
            proposals.append(ops.concatenate([grid, wh], axis=-1))
        output_proposals = ops.concatenate(proposals, axis=0)[None]  # (1, S, 4)
        valid = ops.all(
            (output_proposals > 0.01) & (output_proposals < 0.99),
            axis=-1,
            keepdims=True,
        )
        output_proposals = ops.log(output_proposals / (1 - output_proposals))
        output_proposals = ops.where(valid, output_proposals, float("inf"))
        object_query = ops.where(valid, enc_output, 0.0)
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def select_queries(self, vision, text, text_token_mask, spatial_shapes_list, batch):
        object_query, output_proposals = self.generate_proposals(
            vision, spatial_shapes_list
        )
        enc_class = self.encoder_output_class_embed(object_query, text, text_token_mask)
        enc_coord = self.encoder_output_bbox_embed(object_query) + output_proposals
        topk_logits = ops.max(enc_class, axis=-1)
        topk_idx = ops.top_k(topk_logits, self.num_queries)[1]  # (B, nq)
        gather_idx = ops.repeat(topk_idx[..., None], 4, axis=-1)
        topk_coords = ops.take_along_axis(enc_coord, gather_idx, axis=1)
        reference_points = ops.sigmoid(topk_coords)
        target = ops.broadcast_to(
            self.query_position_embeddings.embeddings[None],
            (batch, self.num_queries, self.d_model),
        )
        return target, reference_points

    def run_decoder(
        self,
        target,
        reference_points,
        vision,
        text,
        text_additive,
        spatial_shapes_list,
        bbox_embed=None,
    ):
        hidden = target
        init_reference = reference_points
        intermediate = []
        intermediate_ref = []
        for idx, layer in enumerate(self.decoder_layers_list):
            ref_input = reference_points[:, :, None] * ops.ones(
                (1, 1, self.num_feature_levels, 1)
            )  # valid_ratios == 1
            query_sine = encode_sine_position(
                ref_input[:, :, 0, :], num_pos_feats=self.d_model // 2
            )
            query_pos = self.reference_points_head(query_sine)
            hidden = layer(
                hidden,
                query_pos,
                ref_input,
                vision,
                text,
                spatial_shapes_list=spatial_shapes_list,
                text_encoder_attention_mask=None,
                vision_encoder_attention_mask=None,
            )
            if bbox_embed is not None:
                delta = bbox_embed[idx](hidden)
                new_ref = ops.sigmoid(
                    delta + ops.log(reference_points / (1 - reference_points))
                )
                reference_points = ops.stop_gradient(new_ref)
            intermediate.append(self.decoder_norm(hidden))
            intermediate_ref.append(reference_points)
        return (
            ops.stack(intermediate, axis=1),
            ops.stack(intermediate_ref, axis=1),
            init_reference,
        )

    def encode(self, inputs):
        pixel_values = ops.cast(
            ops.convert_to_tensor(inputs["pixel_values"]), "float32"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch = int(ops.shape(pixel_values)[0])
        token_mask = (
            ops.cast(ops.convert_to_tensor(attention_mask), "bool")
            if attention_mask is not None
            else ops.cast(ops.ones_like(ops.convert_to_tensor(input_ids)), "bool")
        )
        text_features, text_additive = self.encode_text(input_ids, attention_mask)
        sources = self.backbone_features(pixel_values)
        vision, text, spatial_shapes_list = self.run_encoder(
            sources, text_features, text_additive
        )
        target, reference_points = self.select_queries(
            vision, text, token_mask, spatial_shapes_list, batch
        )
        return {
            "vision": vision,
            "text": text,
            "target": target,
            "reference_points": reference_points,
            "spatial_shapes_list": spatial_shapes_list,
            "text_token_mask": token_mask,
            "text_additive": text_additive,
        }

    def call(self, inputs):
        enc = self.encode(inputs)
        intermediate, intermediate_ref, init_ref = self.run_decoder(
            enc["target"],
            enc["reference_points"],
            enc["vision"],
            enc["text"],
            enc["text_additive"],
            enc["spatial_shapes_list"],
            bbox_embed=None,
        )
        return {
            "last_hidden_state": intermediate[:, -1],
            "intermediate_hidden_states": intermediate,
            "encoder_last_hidden_state_text": enc["text"],
        }

    def compute_output_spec(self, inputs):
        # The whole core runs eagerly at runtime; declare the output shapes so the
        # dynamic spatial / host-side text control flow stays out of the graph trace.
        b = inputs["pixel_values"].shape[0]
        return {
            "last_hidden_state": keras.KerasTensor(
                (b, self.num_queries, self.d_model), dtype=self.compute_dtype
            ),
            "intermediate_hidden_states": keras.KerasTensor(
                (b, self.decoder_layers, self.num_queries, self.d_model),
                dtype=self.compute_dtype,
            ),
            "encoder_last_hidden_state_text": keras.KerasTensor(
                (b, None, self.d_model), dtype=self.compute_dtype
            ),
        }

    def get_config(self):
        config = super().get_config()
        for key in (
            "d_model",
            "encoder_layers",
            "encoder_ffn_dim",
            "encoder_attention_heads",
            "decoder_layers",
            "decoder_ffn_dim",
            "decoder_attention_heads",
            "num_queries",
            "num_feature_levels",
            "encoder_n_points",
            "decoder_n_points",
            "max_text_len",
            "query_dim",
            "two_stage",
            "positional_embedding_temperature",
            "layer_norm_eps",
            "activation_function",
            "backbone_embed_dim",
            "backbone_depths",
            "backbone_num_heads",
            "backbone_window_size",
            "backbone_out_indices",
            "text_vocab_size",
            "text_hidden_size",
            "text_num_layers",
            "text_num_heads",
            "text_intermediate_size",
            "text_max_position_embeddings",
            "text_layer_norm_eps",
        ):
            config[key] = getattr(self, key)
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoDetectCore(GroundingDinoCore):
    """Imperative Grounding DINO detection core: the backbone core plus the
    per-decoder-layer contrastive class heads and bounding-box MLP heads (iterative
    refinement). Wrapped by the functional :class:`GroundingDinoDetect`. ``call``
    returns ``logits`` ``(B, num_queries, max_text_len)`` and ``pred_boxes``
    ``(B, num_queries, 4)`` (cx, cy, w, h in ``[0, 1]``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_embed = [
            GroundingDinoMLPPredictionHead(self.d_model, 4, 3, name=f"bbox_embed_{i}")
            for i in range(self.decoder_layers)
        ]
        self.class_embed = [
            GroundingDinoContrastiveEmbedding(
                self.max_text_len, name=f"class_embed_{i}"
            )
            for i in range(self.decoder_layers)
        ]

    def call(self, inputs):
        enc = self.encode(inputs)
        intermediate, intermediate_ref, init_ref = self.run_decoder(
            enc["target"],
            enc["reference_points"],
            enc["vision"],
            enc["text"],
            enc["text_additive"],
            enc["spatial_shapes_list"],
            bbox_embed=self.bbox_embed,
        )
        text = enc["text"]
        token_mask = enc["text_token_mask"]
        outputs_classes = []
        outputs_coords = []
        for level in range(self.decoder_layers):
            reference = init_ref if level == 0 else intermediate_ref[:, level - 1]
            reference = ops.log(reference / (1 - reference))
            cls = self.class_embed[level](intermediate[:, level], text, token_mask)
            delta = self.bbox_embed[level](intermediate[:, level])
            coord = ops.sigmoid(delta + reference)
            outputs_classes.append(cls)
            outputs_coords.append(coord)
        return {
            "logits": outputs_classes[-1],
            "pred_boxes": outputs_coords[-1],
            "last_hidden_state": intermediate[:, -1],
        }

    def compute_output_spec(self, inputs):
        b = inputs["pixel_values"].shape[0]
        return {
            "logits": keras.KerasTensor(
                (b, self.num_queries, self.max_text_len), dtype=self.compute_dtype
            ),
            "pred_boxes": keras.KerasTensor(
                (b, self.num_queries, 4), dtype=self.compute_dtype
            ),
            "last_hidden_state": keras.KerasTensor(
                (b, self.num_queries, self.d_model), dtype=self.compute_dtype
            ),
        }


def grounding_dino_config_from_hf(hf_config):
    bc = hf_config.get("backbone_config", {})
    tc = hf_config.get("text_config", {})
    return {
        "d_model": hf_config.get("d_model", 256),
        "encoder_layers": hf_config.get("encoder_layers", 6),
        "encoder_ffn_dim": hf_config.get("encoder_ffn_dim", 2048),
        "encoder_attention_heads": hf_config.get("encoder_attention_heads", 8),
        "decoder_layers": hf_config.get("decoder_layers", 6),
        "decoder_ffn_dim": hf_config.get("decoder_ffn_dim", 2048),
        "decoder_attention_heads": hf_config.get("decoder_attention_heads", 8),
        "num_queries": hf_config.get("num_queries", 900),
        "num_feature_levels": hf_config.get("num_feature_levels", 4),
        "encoder_n_points": hf_config.get("encoder_n_points", 4),
        "decoder_n_points": hf_config.get("decoder_n_points", 4),
        "max_text_len": hf_config.get("max_text_len", 256),
        "query_dim": hf_config.get("query_dim", 4),
        "two_stage": hf_config.get("two_stage", True),
        "positional_embedding_temperature": hf_config.get(
            "positional_embedding_temperature", 20.0
        ),
        "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
        "activation_function": hf_config.get("activation_function", "relu"),
        "backbone_embed_dim": bc.get("embed_dim", 96),
        "backbone_depths": tuple(bc.get("depths", (2, 2, 6, 2))),
        "backbone_num_heads": tuple(bc.get("num_heads", (3, 6, 12, 24))),
        "backbone_window_size": bc.get("window_size", 7),
        "backbone_out_indices": tuple(bc.get("out_indices", (2, 3, 4))),
        "text_vocab_size": tc.get("vocab_size", 30522),
        "text_hidden_size": tc.get("hidden_size", 768),
        "text_num_layers": tc.get("num_hidden_layers", 12),
        "text_num_heads": tc.get("num_attention_heads", 12),
        "text_intermediate_size": tc.get("intermediate_size", 3072),
        "text_max_position_embeddings": tc.get("max_position_embeddings", 512),
        "text_layer_norm_eps": tc.get("layer_norm_eps", 1e-12),
    }


_GD_INPUT_SHAPE = {
    "channels_last": (None, None, 3),
    "channels_first": (3, None, None),
}


class _GroundingDinoFunctional(BaseModel):
    """Shared functional wrapper: a graph of ``{pixel_values, input_ids,
    attention_mask} -> core -> outputs``. Subclasses set ``core_class``."""

    HF_MODEL_TYPE = "grounding-dino"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = GroundingDinoConfig
    core_class = GroundingDinoCore

    def __init__(self, name=None, **kwargs):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        core = self.core_class(**kwargs, name="core")
        img_shape = _GD_INPUT_SHAPE[keras.config.image_data_format()]
        pv = layers.Input(shape=img_shape, dtype="float32", name="pixel_values")
        ii = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        am = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        inputs = {"pixel_values": pv, "input_ids": ii, "attention_mask": am}
        outputs = core(inputs)
        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__
        )
        self.core = core

        # The core runs eagerly (compute_output_spec skips its call during graph
        # construction), so its sublayers don't materialize until a real forward.
        with inference_scope():
            self.build_for_transfer()

    def build_for_transfer(self):
        # Always-multimodal build so the vision backbone + cross-modality encoder
        # weights exist before a stream is loaded (used by conversion / caching).
        px = (
            (1, 3, 384, 384)
            if keras.config.image_data_format() == "channels_first"
            else (1, 384, 384, 3)
        )
        self(
            {
                "input_ids": ops.convert_to_tensor(
                    [[101, 102, 1012, 1029, 102, 102]], dtype="int32"
                ),
                "attention_mask": ops.ones((1, 6), dtype="int32"),
                "pixel_values": ops.zeros(px, dtype="float32"),
            }
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        return grounding_dino_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_grounding_dino_hf_to_keras import transfer_grounding_dino_weights

        transfer_grounding_dino_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        for k, v in self.core.get_config().items():
            if k not in ("name", "trainable", "dtype"):
                config[k] = v
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoModel(_GroundingDinoFunctional):
    """Grounding DINO backbone + cross-modality encoder-decoder (no heads).

    A functional model wrapping :class:`GroundingDinoCore`; returns the decoder
    ``last_hidden_state`` ``(B, num_queries, d_model)`` plus the intermediate
    decoder states and the text encoder output. Use :class:`GroundingDinoDetect`
    for logits / boxes.
    """

    core_class = GroundingDinoCore


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoDetect(_GroundingDinoFunctional):
    """Grounding DINO with detection heads (open-set / text-grounded detection).

    A functional model wrapping :class:`GroundingDinoDetectCore`; ``logits`` is
    ``(B, num_queries, max_text_len)`` and ``pred_boxes`` ``(B, num_queries, 4)``
    (cx, cy, w, h in ``[0, 1]``).
    """

    core_class = GroundingDinoDetectCore


# Backward-compatible alias for the previous class name.
GroundingDinoForObjectDetection = GroundingDinoDetect
