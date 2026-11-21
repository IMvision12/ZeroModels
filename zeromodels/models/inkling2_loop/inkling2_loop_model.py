import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    MediaMerge,
    merge_media,
)

from .inkling2_loop_config import INKLING2_LOOP_TOKENS, Inkling2LoopConfig
from .inkling2_loop_layers import (
    MASK_NEG,
    Inkling2LoopBlockAttentionResidual,
    Inkling2LoopDecoderLayer,
    Inkling2LoopMessageAttentionResidual,
    Inkling2LoopRMSNorm,
    Inkling2LoopSoftcap,
    Inkling2VisionEncoderLayer,
    Inkling2VisionMerger,
    vision_axial_rope_cos_sin,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2VisionModel(layers.Layer):
    """Inkling2 Vision Transformer with bilinear pos-emb interpolation, CLS token,
    2D axial RoPE, LayerScale, and 2x2 spatial merger.

    Call args:
        pixel_values: ``(num_image_tokens, merge * patch, merge * patch, C)``
        image_grid_thw: ``(num_images, 3)`` counted in merged LM image tokens.

    Returns:
        Merged image embeddings ``(num_image_tokens, out_dim)``.
    """

    def __init__(
        self,
        embed_dim=1024,
        depth=23,
        num_heads=16,
        mlp_dim=4096,
        in_channels=3,
        patch_size=20,
        spatial_merge_size=2,
        posemb_grid_size=32,
        projection_hidden_dim=4096,
        projector_bias=False,
        out_dim=2304,
        norm_eps=1e-5,
        layer_scale_init_value=0.1,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.posemb_grid_size = posemb_grid_size
        self.projection_hidden_dim = projection_hidden_dim
        self.projector_bias = bool(projector_bias)
        self.out_dim = out_dim
        self.norm_eps = norm_eps
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.rope_theta = float(rope_theta)
        self.head_dim = embed_dim // num_heads

        self.patch_embedding = layers.Dense(
            embed_dim, use_bias=False, name="patch_embedding"
        )
        self.pre_layernorm = layers.LayerNormalization(
            epsilon=norm_eps, name="pre_layernorm"
        )
        self.layers = [
            Inkling2VisionEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                norm_eps=norm_eps,
                layer_scale_init_value=layer_scale_init_value,
                name=f"layers_{i}",
            )
            for i in range(depth)
        ]
        self.merger = Inkling2VisionMerger(
            embed_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
            projection_hidden_dim=projection_hidden_dim,
            out_dim=out_dim,
            projector_bias=projector_bias,
            name="merger",
        )

    def build(self, input_shape):
        patch_feat = self.in_channels * (self.patch_size**2)
        self.patch_embedding.build((None, patch_feat))
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.posemb_grid_size**2, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(self.embed_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.cls_position_embedding = self.add_weight(
            name="cls_position_embedding",
            shape=(self.embed_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.pre_layernorm.build((None, self.embed_dim))
        for layer in self.layers:
            layer.build((None, self.embed_dim))
        self.merger.build((None, (self.spatial_merge_size**2) * self.embed_dim))
        self.built = True

    def _interp_pos_embed(self, patch_grid_rows):
        side = self.posemb_grid_size
        m = self.spatial_merge_size
        pieces = []
        for t, h, w in patch_grid_rows:
            # Spatial-merge-block order: (h//m, w//m, m, m)
            bh, bw = h // m, w // m
            rows_grid, cols_grid = np.meshgrid(
                np.arange(h, dtype="float32"),
                np.arange(w, dtype="float32"),
                indexing="ij",
            )
            rows_ord = (
                rows_grid.reshape(bh, m, bw, m).transpose(0, 2, 1, 3).reshape(-1)
            )
            cols_ord = (
                cols_grid.reshape(bh, m, bw, m).transpose(0, 2, 1, 3).reshape(-1)
            )
            # Bilinear align_corners=False, padding="border"
            src_h = np.clip((rows_ord + 0.5) * (side / float(h)) - 0.5, 0.0, side - 1.0)
            src_w = np.clip((cols_ord + 0.5) * (side / float(w)) - 0.5, 0.0, side - 1.0)
            h0 = np.floor(src_h).astype("int32")
            w0 = np.floor(src_w).astype("int32")
            h1 = np.minimum(h0 + 1, side - 1)
            w1 = np.minimum(w0 + 1, side - 1)
            dh = (src_h - h0).astype("float32")[:, None]
            dw = (src_w - w0).astype("float32")[:, None]

            i00 = ops.convert_to_tensor(h0 * side + w0, dtype="int32")
            i01 = ops.convert_to_tensor(h0 * side + w1, dtype="int32")
            i10 = ops.convert_to_tensor(h1 * side + w0, dtype="int32")
            i11 = ops.convert_to_tensor(h1 * side + w1, dtype="int32")
            w00 = ops.convert_to_tensor((1.0 - dh) * (1.0 - dw), dtype="float32")
            w01 = ops.convert_to_tensor((1.0 - dh) * dw, dtype="float32")
            w10 = ops.convert_to_tensor(dh * (1.0 - dw), dtype="float32")
            w11 = ops.convert_to_tensor(dh * dw, dtype="float32")

            pos_f32 = ops.cast(self.position_embedding, "float32")
            emb = (
                ops.take(pos_f32, i00, axis=0) * w00
                + ops.take(pos_f32, i01, axis=0) * w01
                + ops.take(pos_f32, i10, axis=0) * w10
                + ops.take(pos_f32, i11, axis=0) * w11
            )
            if t > 1:
                emb = ops.concatenate([emb] * t, axis=0)
            pieces.append(emb)
        return ops.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]

    def call(self, pixel_values, image_grid_thw):
        merge = self.spatial_merge_size
        patch = self.patch_size
        if isinstance(image_grid_thw, (list, tuple)):
            grid_rows = [tuple(int(v) for v in row) for row in image_grid_thw]
        else:
            grid_rows = [
                tuple(int(v) for v in row)
                for row in ops.convert_to_numpy(ops.convert_to_tensor(image_grid_thw))
            ]
        patch_grid_rows = [(t, h * merge, w * merge) for t, h, w in grid_rows]

        # pixel_values: (num_image_tokens, merge*patch, merge*patch, C)
        patches = ops.reshape(
            pixel_values, (-1, merge, patch, merge, patch, self.in_channels)
        )
        patches = ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = ops.reshape(patches, (-1, self.in_channels * patch * patch))
        patch_embeds = self.patch_embedding(ops.cast(patches, self.patch_embedding.compute_dtype))
        pos_embeds = ops.cast(self._interp_pos_embed(patch_grid_rows), patch_embeds.dtype)
        patch_embeds = patch_embeds + pos_embeds

        # Prepend one CLS token per image, compute 1-indexed (row, col) position_ids,
        # and build block-diagonal attention mask across images.
        cls_token = ops.cast(
            ops.cast(self.class_embedding, "float32")
            + ops.cast(self.cls_position_embedding, "float32"),
            patch_embeds.dtype,
        )[None, :]  # (1, D)

        seq_chunks = []
        pos_chunks = []
        patch_indices = []
        seg_ids = []
        offset = 0
        seq_cursor = 0
        for img_idx, (t, h, w) in enumerate(patch_grid_rows):
            n_p = t * h * w
            seq_chunks.append(cls_token)
            seq_chunks.append(patch_embeds[offset : offset + n_p])
            offset += n_p

            bh, bw = h // merge, w // merge
            r_grid, c_grid = np.meshgrid(
                np.arange(h, dtype="int32"), np.arange(w, dtype="int32"), indexing="ij"
            )
            r_ord = r_grid.reshape(bh, merge, bw, merge).transpose(0, 2, 1, 3).reshape(-1)
            c_ord = c_grid.reshape(bh, merge, bw, merge).transpose(0, 2, 1, 3).reshape(-1)
            img_pos = np.tile(np.stack([r_ord + 1, c_ord + 1], axis=-1), (t, 1))
            pos_chunks.append(np.zeros((1, 2), dtype="int32"))
            pos_chunks.append(img_pos)

            patch_indices.extend(range(seq_cursor + 1, seq_cursor + 1 + n_p))
            seg_ids.extend([img_idx] * (n_p + 1))
            seq_cursor += n_p + 1

        hidden = ops.concatenate(seq_chunks, axis=0)
        hidden = self.pre_layernorm(hidden)

        pos_ids = ops.convert_to_tensor(np.concatenate(pos_chunks, axis=0), dtype="int32")
        cos, sin = vision_axial_rope_cos_sin(pos_ids, self.head_dim, self.rope_theta)

        if len(patch_grid_rows) > 1:
            seg = ops.convert_to_tensor(seg_ids, dtype="int32")
            mask = ops.cast(
                ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG), "float32"
            )[None, None]
        else:
            mask = None

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask=mask)

        patch_idx_t = ops.convert_to_tensor(patch_indices, dtype="int32")
        patch_states = ops.take(hidden, patch_idx_t, axis=0)
        merged = ops.reshape(patch_states, (-1, (merge**2) * self.embed_dim))
        return self.merger(merged)

    def compute_output_spec(self, pixel_values, image_grid_thw):
        return keras.KerasTensor((None, self.out_dim), dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "in_channels": self.in_channels,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "posemb_grid_size": self.posemb_grid_size,
                "projection_hidden_dim": self.projection_hidden_dim,
                "projector_bias": self.projector_bias,
                "out_dim": self.out_dim,
                "norm_eps": self.norm_eps,
                "layer_scale_init_value": self.layer_scale_init_value,
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopTextModel(layers.Layer):
    """Inkling2Loop text backbone with looped middle layers and 48-slot static cache."""

    def __init__(
        self,
        vocab_size=201024,
        embed_dim=2304,
        num_layers=24,
        num_heads=24,
        num_kv_heads=8,
        head_dim=128,
        swa_num_heads=24,
        swa_num_kv_heads=8,
        swa_head_dim=128,
        sliding_window_size=1024,
        d_rel=16,
        rel_extent=1024,
        log_scaling_n_floor=8192,
        log_scaling_alpha=0.1,
        layer_types=None,
        norm_eps=1e-6,
        conv_kernel_size=4,
        moe_mlp_dim=1152,
        num_experts=128,
        num_experts_per_tok=8,
        route_scale=8.0,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        n_layers_per_attn_res_block=6,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        loop_layer_index_start=6,
        loop_layer_index_end=18,
        loop_num_passes=3,
        loop_gate_rank=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.swa_num_heads = swa_num_heads
        self.swa_num_kv_heads = swa_num_kv_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = float(log_scaling_alpha)
        if layer_types is None:
            layer_types = tuple(
                "hybrid" if (i + 1) % n_layers_per_attn_res_block == 0 else "hybrid_sliding"
                for i in range(num_layers)
            )
        self.layer_types = tuple(layer_types)
        self.norm_eps = norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = float(route_scale)
        self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.loop_layer_index_start = loop_layer_index_start
        self.loop_layer_index_end = loop_layer_index_end
        self.loop_num_passes = loop_num_passes
        self.loop_gate_rank = loop_gate_rank

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.embed_norm = Inkling2LoopRMSNorm(
            embed_dim, eps=norm_eps, name="embed_norm"
        )
        self.decoder_layers = []
        for i in range(num_layers):
            is_sliding = self.layer_types[i] == "hybrid_sliding"
            is_loop = loop_layer_index_start <= i < loop_layer_index_end
            self.decoder_layers.append(
                Inkling2LoopDecoderLayer(
                    layer_idx=i,
                    embed_dim=embed_dim,
                    num_heads=swa_num_heads if is_sliding else num_heads,
                    num_kv_heads=swa_num_kv_heads if is_sliding else num_kv_heads,
                    head_dim=swa_head_dim if is_sliding else head_dim,
                    moe_mlp_dim=moe_mlp_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    route_scale=route_scale,
                    is_sliding=is_sliding,
                    is_loop_layer=is_loop,
                    sliding_window_size=sliding_window_size,
                    d_rel=d_rel,
                    rel_extent=rel_extent,
                    conv_kernel_size=conv_kernel_size,
                    norm_eps=norm_eps,
                    qk_l2_clip_threshold=qk_l2_clip_threshold,
                    rel_l2_clip_threshold=rel_l2_clip_threshold,
                    log_scaling_alpha=log_scaling_alpha,
                    log_scaling_n_floor=log_scaling_n_floor,
                    n_layers_per_attn_res_block=n_layers_per_attn_res_block,
                    mlp_gate_softcap=mlp_gate_softcap,
                    mlp_up_softcap=mlp_up_softcap,
                    loop_gate_rank=loop_gate_rank,
                    name=f"decoder_layer_{i}",
                )
            )
        num_loop_layers = loop_layer_index_end - loop_layer_index_start
        self.message_input = Inkling2LoopMessageAttentionResidual(
            num_loop_layers, embed_dim, name="message_input"
        )
        self.final_input = Inkling2LoopBlockAttentionResidual(
            embed_dim, num_queries=1, name="final_input"
        )
        self.final_norm = Inkling2LoopRMSNorm(
            embed_dim, eps=norm_eps, name="final_norm"
        )

    def build(self, input_shape):
        shape = (None, None, self.embed_dim)
        self.token_embedding.build((None, None))
        self.embed_norm.build(shape)
        for layer in self.decoder_layers:
            layer.build(shape)
        self.message_input.build(shape)
        self.final_input.build(shape)
        self.final_norm.build(shape)
        self.built = True

    def embed_ids(self, input_ids):
        return self.embed_norm(self.token_embedding(input_ids))

    @staticmethod
    def _append_block(block_states, hidden_states):
        entry = hidden_states[:, :, None, :]
        if block_states is None:
            return entry
        return ops.concatenate([block_states, entry], axis=-2)

    def _prefill_masks(self, seq_len, attention_mask=None):
        q_pos = ops.arange(seq_len)[:, None]
        k_pos = ops.arange(seq_len)[None, :]
        rel_dist = q_pos - k_pos

        full_allowed = k_pos <= q_pos
        swa_allowed = ops.logical_and(
            full_allowed, k_pos > q_pos - self.sliding_window_size
        )
        full_mask = ops.cast(ops.where(full_allowed, 0.0, MASK_NEG), "float32")[None, None]
        swa_mask = ops.cast(ops.where(swa_allowed, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(attention_mask, "float32")
            pad_add = (1.0 - am)[:, None, None, :] * MASK_NEG
            full_mask = full_mask + pad_add
            swa_mask = swa_mask + pad_add
        return full_mask, swa_mask, rel_dist

    def call(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        b = ops.shape(inputs_embeds)[0]
        seq_len = ops.shape(inputs_embeds)[1]
        if position_ids is None:
            position_ids = ops.broadcast_to(ops.arange(seq_len, dtype="int32")[None, :], (b, seq_len))

        full_mask, swa_mask, rel_dist = self._prefill_masks(seq_len, attention_mask)
        conv_mask = attention_mask

        loop_start = self.loop_layer_index_start
        loop_end = self.loop_layer_index_end

        hidden_states = inputs_embeds
        block_states = None
        caches = []

        def run_layer(layer, h, blocks, messages=None):
            mask = swa_mask if layer.is_sliding else full_mask
            res = layer(
                h,
                blocks,
                attention_mask=mask,
                relative_distance=rel_dist,
                position_ids=position_ids,
                messages=messages,
                conv_mask=conv_mask,
                use_cache=use_cache,
            )
            if use_cache:
                new_h, st = res
                caches.append(st)
            else:
                new_h = res
            if layer.is_block_boundary:
                blocks = self._append_block(blocks, h)
            return new_h, blocks

        # Pre-loop layers
        for layer in self.decoder_layers[:loop_start]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states)

        pre_loop_hidden = hidden_states
        pre_loop_blocks = block_states
        num_pre_loop_blocks = 0 if pre_loop_blocks is None else pre_loop_blocks.shape[-2]

        # First pass over loop layers
        for layer in self.decoder_layers[loop_start:loop_end]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states)

        # Remaining passes restart from pre-loop state and receive cross-pass messages
        for _ in range(1, self.loop_num_passes):
            prev_blocks = (
                block_states[:, :, num_pre_loop_blocks + 1 :, :]
                if block_states is not None and block_states.shape[-2] > num_pre_loop_blocks + 1
                else None
            )
            prev_hidden = hidden_states
            hidden_states, block_states = pre_loop_hidden, pre_loop_blocks
            for loop_layer_idx, layer in enumerate(self.decoder_layers[loop_start:loop_end]):
                messages = self.message_input(
                    prev_hidden, block_states=prev_blocks, loop_layer_idx=loop_layer_idx
                )
                hidden_states, block_states = run_layer(
                    layer, hidden_states, block_states, messages=messages
                )

        # Post-loop layers
        for layer in self.decoder_layers[loop_end:]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states)

        hidden_states = self.final_input(hidden_states, block_states=block_states)[0]
        hidden_states = ops.cast(self.final_norm(hidden_states), inputs_embeds.dtype)
        if use_cache:
            return hidden_states, tuple(caches)
        return hidden_states

    def decode_step(self, inputs_embeds, cache, cache_update_index, max_len):
        b = ops.shape(inputs_embeds)[0]
        position_ids = ops.broadcast_to(
            ops.reshape(ops.cast(cache_update_index, "int32"), (1, 1)), (b, 1)
        )
        kv_pos = ops.arange(max_len, dtype="int32")
        rel_dist = ops.reshape(ops.cast(cache_update_index, "int32") - kv_pos, (1, max_len))

        full_allowed = kv_pos <= cache_update_index
        swa_allowed = ops.logical_and(
            full_allowed, kv_pos > cache_update_index - self.sliding_window_size
        )
        full_mask = ops.cast(ops.where(full_allowed, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        swa_mask = ops.cast(ops.where(swa_allowed, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]

        loop_start = self.loop_layer_index_start
        loop_end = self.loop_layer_index_end
        hidden_states = inputs_embeds
        block_states = None
        new_cache = []
        slot = 0

        def run_decode_layer(layer, h, blocks, slot_idx, messages=None):
            mask = swa_mask if layer.is_sliding else full_mask
            new_h, st = layer.decode_step(
                h,
                blocks,
                cache[slot_idx],
                cache_update_index,
                attention_mask=mask,
                relative_distance=rel_dist,
                position_ids=position_ids,
                messages=messages,
            )
            new_cache.append(st)
            if layer.is_block_boundary:
                blocks = self._append_block(blocks, h)
            return new_h, blocks

        for layer in self.decoder_layers[:loop_start]:
            hidden_states, block_states = run_decode_layer(
                layer, hidden_states, block_states, slot
            )
            slot += 1

        pre_loop_hidden = hidden_states
        pre_loop_blocks = block_states
        num_pre_loop_blocks = 0 if pre_loop_blocks is None else pre_loop_blocks.shape[-2]

        for layer in self.decoder_layers[loop_start:loop_end]:
            hidden_states, block_states = run_decode_layer(
                layer, hidden_states, block_states, slot
            )
            slot += 1

        for _ in range(1, self.loop_num_passes):
            prev_blocks = (
                block_states[:, :, num_pre_loop_blocks + 1 :, :]
                if block_states is not None and block_states.shape[-2] > num_pre_loop_blocks + 1
                else None
            )
            prev_hidden = hidden_states
            hidden_states, block_states = pre_loop_hidden, pre_loop_blocks
            for loop_layer_idx, layer in enumerate(self.decoder_layers[loop_start:loop_end]):
                messages = self.message_input(
                    prev_hidden, block_states=prev_blocks, loop_layer_idx=loop_layer_idx
                )
                hidden_states, block_states = run_decode_layer(
                    layer, hidden_states, block_states, slot, messages=messages
                )
                slot += 1

        for layer in self.decoder_layers[loop_end:]:
            hidden_states, block_states = run_decode_layer(
                layer, hidden_states, block_states, slot
            )
            slot += 1

        hidden_states = self.final_input(hidden_states, block_states=block_states)[0]
        hidden_states = ops.cast(self.final_norm(hidden_states), inputs_embeds.dtype)
        return hidden_states, tuple(new_cache)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "swa_num_heads": self.swa_num_heads,
                "swa_num_kv_heads": self.swa_num_kv_heads,
                "swa_head_dim": self.swa_head_dim,
                "sliding_window_size": self.sliding_window_size,
                "d_rel": self.d_rel,
                "rel_extent": self.rel_extent,
                "log_scaling_n_floor": self.log_scaling_n_floor,
                "log_scaling_alpha": self.log_scaling_alpha,
                "layer_types": list(self.layer_types),
                "norm_eps": self.norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "route_scale": self.route_scale,
                "qk_l2_clip_threshold": self.qk_l2_clip_threshold,
                "rel_l2_clip_threshold": self.rel_l2_clip_threshold,
                "n_layers_per_attn_res_block": self.n_layers_per_attn_res_block,
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
                "loop_layer_index_start": self.loop_layer_index_start,
                "loop_layer_index_end": self.loop_layer_index_end,
                "loop_num_passes": self.loop_num_passes,
                "loop_gate_rank": self.loop_gate_rank,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopModel(BaseModel):
    """Multimodal Inkling-2-Mini backbone (``Inkling2VisionModel`` + ``Inkling2LoopTextModel``).

    Accepts a dict with ``input_ids`` and optional ``attention_mask``, ``position_ids``,
    ``pixel_values``, and ``image_grid_thw``. Returns ``{"last_hidden_state": ...}``.
    Use :class:`Inkling2LoopConditionalGenerate` for logits and ``.generate()``.
    """

    HF_MODEL_TYPE = ("inkling2_loop", "inkling2_loop_text")
    default_load_dtype = "bfloat16"
    config_class = Inkling2LoopConfig
    output_logits = False

    def __init__(
        self,
        vocab_size=201024,
        unpadded_vocab_size=200058,
        embed_dim=2304,
        num_layers=24,
        num_heads=24,
        num_kv_heads=8,
        head_dim=128,
        swa_num_heads=24,
        swa_num_kv_heads=8,
        swa_head_dim=128,
        sliding_window_size=1024,
        d_rel=16,
        rel_extent=1024,
        log_scaling_n_floor=8192,
        log_scaling_alpha=0.1,
        layer_types=None,
        max_position_embeddings=1048576,
        norm_eps=1e-6,
        conv_kernel_size=4,
        moe_mlp_dim=1152,
        num_experts=128,
        num_experts_per_tok=8,
        route_scale=8.0,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        n_layers_per_attn_res_block=6,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        loop_layer_index_start=6,
        loop_layer_index_end=18,
        loop_num_passes=3,
        loop_gate_rank=64,
        vision_embed_dim=1024,
        vision_depth=23,
        vision_num_heads=16,
        vision_mlp_ratio=4.0,
        vision_mlp_dim=None,
        in_channels=3,
        patch_size=20,
        spatial_merge_size=2,
        posemb_grid_size=32,
        projection_hidden_dim=4096,
        projector_bias=False,
        vision_out_dim=None,
        vision_norm_eps=1e-5,
        vision_layer_scale_init_value=0.1,
        vision_rope_theta=10000.0,
        image_token_id=INKLING2_LOOP_TOKENS["image_token_id"],
        eos_token_id=INKLING2_LOOP_TOKENS["eos_token_id"],
        **kwargs,
    ):
        nm = kwargs.pop("name", None)
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        vision_out_dim = vision_out_dim or embed_dim
        vision_mlp_dim = vision_mlp_dim or int(vision_embed_dim * vision_mlp_ratio)
        if layer_types is None:
            layer_types = tuple(
                "hybrid" if (i + 1) % n_layers_per_attn_res_block == 0 else "hybrid_sliding"
                for i in range(num_layers)
            )
        layer_types = tuple(layer_types)

        super().__init__(name=nm or type(self).__name__, **kwargs)

        self.vision_tower = Inkling2VisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_dim=vision_mlp_dim,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            posemb_grid_size=posemb_grid_size,
            projection_hidden_dim=projection_hidden_dim,
            projector_bias=projector_bias,
            out_dim=vision_out_dim,
            norm_eps=vision_norm_eps,
            layer_scale_init_value=vision_layer_scale_init_value,
            rope_theta=vision_rope_theta,
            name="vision_tower",
        )
        self.language_model = Inkling2LoopTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            swa_num_heads=swa_num_heads,
            swa_num_kv_heads=swa_num_kv_heads,
            swa_head_dim=swa_head_dim,
            sliding_window_size=sliding_window_size,
            d_rel=d_rel,
            rel_extent=rel_extent,
            log_scaling_n_floor=log_scaling_n_floor,
            log_scaling_alpha=log_scaling_alpha,
            layer_types=layer_types,
            norm_eps=norm_eps,
            conv_kernel_size=conv_kernel_size,
            moe_mlp_dim=moe_mlp_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            route_scale=route_scale,
            qk_l2_clip_threshold=qk_l2_clip_threshold,
            rel_l2_clip_threshold=rel_l2_clip_threshold,
            n_layers_per_attn_res_block=n_layers_per_attn_res_block,
            mlp_gate_softcap=mlp_gate_softcap,
            mlp_up_softcap=mlp_up_softcap,
            loop_layer_index_start=loop_layer_index_start,
            loop_layer_index_end=loop_layer_index_end,
            loop_num_passes=loop_num_passes,
            loop_gate_rank=loop_gate_rank,
            name="language_model",
        )
        self.image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        self.lm_head = None
        self.logit_softcap = None
        if self.output_logits:
            self.lm_head = layers.Dense(
                unpadded_vocab_size, use_bias=False, name="lm_head"
            )
            self.logit_softcap = Inkling2LoopSoftcap(
                final_logit_softcapping, name="logit_softcap"
            )

        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.swa_num_heads = swa_num_heads
        self.swa_num_kv_heads = swa_num_kv_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.layer_types = layer_types
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = float(route_scale)
        self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.final_logit_softcapping = float(final_logit_softcapping)
        self.loop_layer_index_start = loop_layer_index_start
        self.loop_layer_index_end = loop_layer_index_end
        self.loop_num_passes = loop_num_passes
        self.loop_gate_rank = loop_gate_rank
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = float(vision_mlp_ratio)
        self.vision_mlp_dim = vision_mlp_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.posemb_grid_size = posemb_grid_size
        self.projection_hidden_dim = projection_hidden_dim
        self.projector_bias = bool(projector_bias)
        self.vision_out_dim = vision_out_dim
        self.vision_norm_eps = vision_norm_eps
        self.vision_layer_scale_init_value = float(vision_layer_scale_init_value)
        self.vision_rope_theta = float(vision_rope_theta)
        self.image_token_id = image_token_id
        self.eos_token_id_val = eos_token_id

    def build(self, input_shape=None):
        token_patch = self.patch_size * self.spatial_merge_size
        self.vision_tower.build((None, token_patch, token_patch, self.in_channels))
        self.language_model.build((None, None, self.embed_dim))
        if self.lm_head is not None:
            self.lm_head.build((None, None, self.embed_dim))
        self.built = True

    def get_image_features(self, pixel_values, image_grid_thw):
        return self.vision_tower(pixel_values, image_grid_thw)

    def _prepare_embeds(self, input_ids, pixel_values=None, image_grid_thw=None):
        embeds = self.language_model.embed_ids(input_ids)
        if pixel_values is not None and image_grid_thw is not None:
            image_features = self.get_image_features(
                ops.convert_to_tensor(pixel_values), image_grid_thw
            )
            embeds = merge_media(
                embeds, input_ids, image_features, self.image_token_id, self.embed_dim
            )
        return embeds

    def project(self, hidden):
        logits = ops.cast(self.lm_head(hidden), "float32")
        return self.logit_softcap(logits)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = inputs.get("position_ids")
            if position_ids is not None:
                position_ids = ops.cast(ops.convert_to_tensor(position_ids), "int32")
            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")
        else:
            input_ids = ops.cast(ops.convert_to_tensor(inputs), "int32")
            attention_mask = None
            position_ids = None
            pixel_values = None
            image_grid_thw = None

        embeds = self._prepare_embeds(input_ids, pixel_values, image_grid_thw)
        hidden = self.language_model(
            embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        out = {"last_hidden_state": hidden}
        if self.output_logits:
            out["logits"] = self.project(hidden)
        return out

    @classmethod
    def config_from_hf(cls, hf_config):
        tc = hf_config.get("text_config", hf_config)
        vc = hf_config.get("vision_config", {})
        lc = tc.get("loop_config") or {}
        hidden = tc.get("hidden_size", 2304)
        n_layers = tc.get("num_hidden_layers", 24)
        n_block = tc.get("n_layers_per_attn_res_block", 6)

        local_ids = tc.get("local_layer_ids")
        layer_types = tc.get("layer_types")
        if layer_types is None:
            if local_ids is not None:
                local_set = set(local_ids)
                layer_types = tuple(
                    "hybrid_sliding" if i in local_set else "hybrid"
                    for i in range(n_layers)
                )
            else:
                layer_types = tuple(
                    "hybrid" if (i + 1) % n_block == 0 else "hybrid_sliding"
                    for i in range(n_layers)
                )
        else:
            layer_types = tuple(layer_types)

        v_hidden = vc.get("hidden_size", 1024)
        v_mlp_ratio = vc.get("mlp_ratio", 4.0)
        v_rope = vc.get("rope_parameters") or {}
        return {
            "vocab_size": tc.get("vocab_size", 201024),
            "unpadded_vocab_size": tc.get("unpadded_vocab_size", 200058),
            "embed_dim": hidden,
            "num_layers": n_layers,
            "num_heads": tc.get("num_attention_heads", 24),
            "num_kv_heads": tc.get("num_key_value_heads", 8),
            "head_dim": tc.get("head_dim", 128),
            "swa_num_heads": tc.get("swa_num_attention_heads", tc.get("num_attention_heads", 24)),
            "swa_num_kv_heads": tc.get("swa_num_key_value_heads", tc.get("num_key_value_heads", 8)),
            "swa_head_dim": tc.get("swa_head_dim", tc.get("head_dim", 128)),
            "sliding_window_size": tc.get("sliding_window_size", tc.get("sliding_window", 1024)),
            "d_rel": tc.get("d_rel", 16),
            "rel_extent": tc.get("rel_extent", 1024),
            "log_scaling_n_floor": tc.get("log_scaling_n_floor", 8192),
            "log_scaling_alpha": tc.get("log_scaling_alpha", 0.1),
            "layer_types": layer_types,
            "max_position_embeddings": tc.get(
                "model_max_length", tc.get("max_position_embeddings", 1048576)
            ),
            "norm_eps": tc.get("rms_norm_eps", 1e-6),
            "conv_kernel_size": tc.get("sconv_kernel_size", tc.get("conv_kernel_size", 4)),
            "moe_mlp_dim": tc.get("intermediate_size", tc.get("moe_intermediate_size", 1152)),
            "num_experts": tc.get("n_routed_experts", tc.get("num_local_experts", 128)),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "route_scale": tc.get("route_scale", tc.get("routed_scaling_factor", 8.0)),
            "qk_l2_clip_threshold": tc.get("qk_l2_clip_threshold", 22.627416997969522),
            "rel_l2_clip_threshold": tc.get("rel_l2_clip_threshold", 8.0),
            "n_layers_per_attn_res_block": n_block,
            "mlp_gate_softcap": tc.get("mlp_gate_softcap", 4.0),
            "mlp_up_softcap": tc.get("mlp_up_softcap", 25.0),
            "final_logit_softcapping": tc.get("final_logit_softcapping", 30.0),
            "loop_layer_index_start": lc.get(
                "core_start", tc.get("loop_layer_index_start", 6)
            ),
            "loop_layer_index_end": lc.get(
                "core_end", tc.get("loop_layer_index_end", 18)
            ),
            "loop_num_passes": lc.get("num_passes", tc.get("loop_num_passes", 3)),
            "loop_gate_rank": lc.get("gate_rank", tc.get("loop_gate_rank", 64)),
            "vision_embed_dim": v_hidden,
            "vision_depth": vc.get("num_hidden_layers", 23),
            "vision_num_heads": vc.get("num_attention_heads", 16),
            "vision_mlp_ratio": v_mlp_ratio,
            "vision_mlp_dim": vc.get("intermediate_size", int(v_hidden * v_mlp_ratio)),
            "in_channels": vc.get("num_channels", 3),
            "patch_size": vc.get("native_patch_size", vc.get("patch_size", 20)),
            "spatial_merge_size": vc.get(
                "adapter_merge_size", vc.get("spatial_merge_size", 2)
            ),
            "posemb_grid_size": vc.get("posemb_grid_size", 32),
            "projection_hidden_dim": vc.get("projection_hidden_dim", 4096),
            "projector_bias": bool(vc.get("projector_bias", False)),
            "vision_out_dim": vc.get("text_hidden_size", hidden),
            "vision_norm_eps": vc.get("layer_norm_eps", 1e-5),
            "vision_layer_scale_init_value": vc.get("layer_scale_init_value", 0.1),
            "vision_rope_theta": v_rope.get("rope_theta", 10000.0),
            "image_token_id": hf_config.get("image_token_id", 200054),
            "eos_token_id": hf_config.get("eos_token_id", tc.get("eos_token_id", 200006)),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_inkling2_loop_hf_to_keras import transfer_inkling2_loop_weights

        transfer_inkling2_loop_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "unpadded_vocab_size": self.unpadded_vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "swa_num_heads": self.swa_num_heads,
                "swa_num_kv_heads": self.swa_num_kv_heads,
                "swa_head_dim": self.swa_head_dim,
                "sliding_window_size": self.sliding_window_size,
                "d_rel": self.d_rel,
                "rel_extent": self.rel_extent,
                "log_scaling_n_floor": self.log_scaling_n_floor,
                "log_scaling_alpha": self.log_scaling_alpha,
                "layer_types": list(self.layer_types),
                "max_position_embeddings": self.max_position_embeddings,
                "norm_eps": self.norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "route_scale": self.route_scale,
                "qk_l2_clip_threshold": self.qk_l2_clip_threshold,
                "rel_l2_clip_threshold": self.rel_l2_clip_threshold,
                "n_layers_per_attn_res_block": self.n_layers_per_attn_res_block,
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
                "final_logit_softcapping": self.final_logit_softcapping,
                "loop_layer_index_start": self.loop_layer_index_start,
                "loop_layer_index_end": self.loop_layer_index_end,
                "loop_num_passes": self.loop_num_passes,
                "loop_gate_rank": self.loop_gate_rank,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_depth": self.vision_depth,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "vision_mlp_dim": self.vision_mlp_dim,
                "in_channels": self.in_channels,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "posemb_grid_size": self.posemb_grid_size,
                "projection_hidden_dim": self.projection_hidden_dim,
                "projector_bias": self.projector_bias,
                "vision_out_dim": self.vision_out_dim,
                "vision_norm_eps": self.vision_norm_eps,
                "vision_layer_scale_init_value": self.vision_layer_scale_init_value,
                "vision_rope_theta": self.vision_rope_theta,
                "image_token_id": self.image_token_id,
                "eos_token_id": self.eos_token_id_val,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopConditionalGenerate(Inkling2LoopModel, BaseGeneration):
    """Inkling-2-Mini with unpadded LM head, logit softcapping, and JAX-JIT ``.generate()``."""

    eos_token_id = (INKLING2_LOOP_TOKENS["eos_token_id"],)
    output_logits = True

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        image_grid_thw=None,
    ):
        batch = int(token_ids.shape[0])
        embeds = self._prepare_embeds(token_ids, pixel_values, image_grid_thw)
        hidden, raw_caches = self.language_model(
            embeds, attention_mask=padding_mask, use_cache=True
        )
        padded_cache = []
        for k, v, k_conv, v_conv, attn_conv, mlp_conv in raw_caches:
            nkv = int(k.shape[1])
            hd = int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            padded_cache.append((ck, cv, k_conv, v_conv, attn_conv, mlp_conv))
        logits = self.project(hidden[:, -1, :])
        return tuple(padded_cache), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        max_len = int(cache[0][0].shape[2])
        embeds = self.language_model.embed_ids(token_ids)
        hidden, new_cache = self.language_model.decode_step(
            embeds, cache, cache_update_index, max_len
        )
        logits = self.project(hidden)[:, 0, :]
        return logits, new_cache
