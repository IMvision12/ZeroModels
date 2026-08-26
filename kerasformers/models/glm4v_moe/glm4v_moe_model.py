import itertools

import keras
from keras import layers, ops

from kerasformers.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
)
from kerasformers.base.base_mixin import inference_scope
from kerasformers.models.glm4_moe.glm4_moe_layers import (
    Glm4MoeDecoderLayer,
    Glm4MoeRMSNorm,
)
from kerasformers.models.glm4v.glm4v_vision_layers import Glm4vVisionModel

from .glm4v_moe_config import Glm4vMoeConfig

MASK_NEG = -1e9


def glm_moe_mrope_cos_sin(position_ids, rotary_dim, theta, mrope_section):
    """Merged M-RoPE tables for GLM-4.5V's partial NeoX rope.

    Per-axis frequencies over ``rotary_dim // 2`` channels are collapsed by the
    ``mrope_section`` split (chunk ``i`` keeps axis ``i % 3``), then duplicated
    (``cat(freqs, freqs)``) to ``rotary_dim``. Returns ``(cos, sin)`` each
    ``(batch, seq, rotary_dim)``.
    """
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq  # (3, b, s, half)
    pts = list(itertools.accumulate(mrope_section))[:-1]
    splits = ops.split(freqs, pts, axis=-1)
    merged = ops.concatenate(
        [splits[i][i % 3] for i in range(len(splits))], axis=-1
    )  # (b, s, half)
    emb = ops.concatenate([merged, merged], axis=-1)
    return ops.cos(emb), ops.sin(emb)


def glm4v_moe_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
    position_ids,
    *,
    language_model,
    visual,
    image_merge,
    causal_mask,
    rotary_dim,
    rope_theta,
    mrope_section,
):
    inputs_embeds = language_model.token_embedding(input_ids)
    image_embeds = visual(pixel_values, image_grid_thw)
    hidden = image_merge(inputs_embeds, input_ids, image_embeds)
    pos = ops.transpose(position_ids, (1, 0, 2))  # (batch, 3, seq) -> (3, batch, seq)
    cos, sin = glm_moe_mrope_cos_sin(pos, rotary_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(hidden, cos, sin, attention_mask=mask)


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vMoeTextModel(layers.Layer):
    """GLM-4.5V text decoder: ``embed -> num_layers x Glm4MoeDecoderLayer -> RMSNorm``.

    Reuses the GLM-4.5 MoE decoder block (grouped-topk sigmoid router with a
    shared expert, NeoX partial rope, biased q/k/v); the first ``first_k_dense``
    layers are dense. ``call`` consumes the fused ``inputs_embeds`` and merged
    M-RoPE tables.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mlp_dim,
        moe_mlp_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        num_experts,
        num_experts_per_tok,
        n_shared_experts,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        first_k_dense,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.norm_eps = norm_eps

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Glm4MoeDecoderLayer(
                embed_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                use_moe=i >= first_k_dense,
                mlp_dim=mlp_dim,
                moe_mlp_dim=moe_mlp_dim,
                shared_mlp_dim=moe_mlp_dim * n_shared_experts,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                use_qk_norm=False,
                attention_bias=True,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Glm4MoeRMSNorm(eps=norm_eps, name="final_norm")

    def call(self, inputs_embeds, cos, sin, attention_mask=None, use_cache=False):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        for layer in self.decoder_layers:
            out = layer(
                hidden, cos, sin, attention_mask=attention_mask, use_cache=use_cache
            )
            if use_cache:
                hidden, kv = out
                new_cache.append(kv)
            else:
                hidden = out
        hidden = self.final_norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def compute_output_spec(
        self, inputs_embeds, cos, sin, attention_mask=None, use_cache=False
    ):
        return keras.KerasTensor(inputs_embeds.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "rotary_dim": self.rotary_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense": self.first_k_dense,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vMoeModel(BaseModel):
    """GLM-4.5V multimodal backbone: GLM-4V vision tower + GLM-4.5 MoE decoder.

    A functional model: the vision tower and image -> text merge run in the graph
    over ``{input_ids, attention_mask, pixel_values, image_grid_thw,
    position_ids}`` (image always a graph input; a no-op merge when no
    ``image_token_id`` is present). The 3D M-RoPE ``position_ids`` are precomputed
    imperatively by :meth:`get_rope_index` and passed in. Returns
    ``last_hidden_state``; use :class:`Glm4vMoeConditionalGenerate` for logits / text.
    """

    HF_MODEL_TYPE = "glm4v_moe"
    config_class = Glm4vMoeConfig
    output_logits = False

    def __init__(
        self,
        vocab_size=151424,
        embed_dim=4096,
        mlp_dim=10944,
        moe_mlp_dim=1408,
        num_layers=46,
        num_heads=96,
        num_kv_heads=8,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        first_k_dense=1,
        partial_rotary_factor=0.5,
        norm_eps=1e-5,
        rope_theta=10000.0,
        mrope_section=(8, 12, 12),
        tie_embeddings=False,
        vision_depth=24,
        vision_embed_dim=1536,
        vision_num_heads=12,
        vision_mlp_dim=13696,
        vision_out_dim=4096,
        image_size=336,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        vision_norm_eps=1e-5,
        image_token_id=151363,
        video_token_id=151364,
        image_start_token_id=151339,
        image_end_token_id=151340,
        video_start_token_id=151341,
        video_end_token_id=151342,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads
        rotary_dim = int(head_dim * partial_rotary_factor)
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        visual = Glm4vVisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            out_hidden_size=vision_out_dim,
            intermediate_size=vision_mlp_dim,
            image_size=image_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            norm_eps=vision_norm_eps,
            name="visual",
        )
        language_model = Glm4vMoeTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            moe_mlp_dim=moe_mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_shared_experts=n_shared_experts,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense=first_k_dense,
            norm_eps=norm_eps,
            name="language_model",
        )
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        causal_mask = CausalMask(name="causal_mask")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=(patch_dim,), dtype="float32", name="pixel_values"
            ),
            "image_grid_thw": layers.Input(
                shape=(3,), dtype="int32", name="image_grid_thw"
            ),
            "position_ids": layers.Input(
                shape=(3, None), dtype="int32", name="position_ids"
            ),
        }
        hidden = glm4v_moe_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["position_ids"],
            language_model=language_model,
            visual=visual,
            image_merge=image_merge,
            causal_mask=causal_mask,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            mrope_section=tuple(mrope_section),
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(language_model.token_embedding, name="lm_head")(hidden)
            )

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.visual = visual
        self.language_model = language_model
        self.image_merge = image_merge
        self.causal_mask_layer = causal_mask
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.tie_embeddings = tie_embeddings
        self.vision_depth = vision_depth
        self.vision_embed_dim = vision_embed_dim
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_out_dim = vision_out_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.vision_norm_eps = vision_norm_eps
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        orig = image_size // patch_size
        n = (orig // spatial_merge_size) ** 2
        seq = n + 2
        with inference_scope():
            self(
                {
                    "input_ids": ops.concatenate(
                        [
                            ops.zeros((1, 1), dtype="int32"),
                            ops.full((1, n), image_token_id, dtype="int32"),
                            ops.ones((1, 1), dtype="int32"),
                        ],
                        axis=1,
                    ),
                    "attention_mask": ops.ones((1, seq), dtype="int32"),
                    "pixel_values": ops.zeros(
                        (orig * orig, patch_dim), dtype="float32"
                    ),
                    "image_grid_thw": ops.convert_to_tensor(
                        [[1, orig, orig]], dtype="int32"
                    ),
                    "position_ids": ops.broadcast_to(
                        ops.arange(seq, dtype="int32"), (1, 3, seq)
                    ),
                }
            )

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        m = self.spatial_merge_size
        ids_host = ops.convert_to_numpy(ops.convert_to_tensor(input_ids)).tolist()
        batch, seq = len(ids_host), len(ids_host[0])

        def _rows(grid):
            return [
                tuple(int(v) for v in row)
                for row in ops.convert_to_numpy(ops.convert_to_tensor(grid))
            ]

        grid_iter = iter(_rows(image_grid_thw)) if image_grid_thw is not None else None
        mask_host = (
            ops.convert_to_numpy(ops.convert_to_tensor(attention_mask)).tolist()
            if attention_mask is not None
            else None
        )
        pos_rows = []
        rope_deltas = []
        for bi in range(batch):
            ids = ids_host[bi]
            keep = (
                [bool(v) for v in mask_host[bi]]
                if mask_host is not None
                else [True] * seq
            )
            kept_idx = [j for j in range(seq) if keep[j]]
            ids_kept = [ids[j] for j in kept_idx]
            ttype = [1 if v == self.image_token_id else 0 for v in ids_kept]
            cur = 0
            pieces = []
            for key, group in itertools.groupby(enumerate(ttype), lambda x: x[1]):
                g = list(group)
                start, end = g[0][0], g[-1][0] + 1
                if key == 0:
                    length = end - start
                    pieces.append(
                        ops.broadcast_to(ops.arange(cur, cur + length), (3, length))
                    )
                    cur += length
                else:
                    t, h, w = (int(v) for v in next(grid_iter))
                    lt, lh, lw = t, h // m, w // m
                    wpos = ops.tile(ops.arange(cur, cur + lw), [lh * lt])
                    hpos = ops.repeat(ops.tile(ops.arange(cur, cur + lh), [lt]), lw)
                    tpos = ops.repeat(ops.arange(lt), lh * lw) + cur
                    pieces.append(ops.stack([tpos, hpos, wpos], axis=0))
                    cur += max(h, w) // m
            llm_positions = ops.concatenate(pieces, axis=1)
            if len(kept_idx) == seq:
                pos_b = llm_positions
            else:
                scatter_idx = ops.reshape(
                    ops.convert_to_tensor(kept_idx, dtype="int32"), (-1, 1)
                )
                pos_b = ops.stack(
                    [
                        ops.scatter(scatter_idx, llm_positions[r], (seq,))
                        for r in range(3)
                    ],
                    axis=0,
                )
            rope_deltas.append(int(ops.max(llm_positions)) + 1 - len(ids_kept))
            pos_rows.append(ops.cast(pos_b, "int32"))
        position_ids = ops.stack(pos_rows, axis=1)
        return position_ids, ops.convert_to_tensor(rope_deltas, dtype="int32")

    def _causal_mask(self, q_len, kv_len, offset, attention_mask=None):
        qi = ops.arange(q_len)[:, None] + offset
        ki = ops.arange(kv_len)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def _prepare_inputs(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.language_model.token_embedding(input_ids)
        rope_deltas = ops.zeros((batch,), dtype="int32")

        has_image = pixel_values is not None and image_grid_thw is not None
        if has_image:
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            embeds_flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
            image_grid = ops.cast(ops.convert_to_tensor(image_grid_thw), "int32")
            image_embeds = self.visual(pixel_values, image_grid)
            idx = [j for j, v in enumerate(ids_flat) if v == self.image_token_id]
            embeds_flat = ops.scatter_update(
                embeds_flat,
                ops.reshape(ops.convert_to_tensor(idx, dtype="int32"), (-1, 1)),
                ops.cast(image_embeds, embeds_flat.dtype),
            )
            inputs_embeds = ops.reshape(embeds_flat, (batch, seq, self.embed_dim))
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid, attention_mask=attention_mask
            )
        else:
            if attention_mask is not None:
                am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
                pos = ops.where(am == 0, 0, ops.cumsum(am, axis=-1) - 1)
            else:
                pos = ops.broadcast_to(ops.arange(seq), (batch, seq))
            position_ids = ops.broadcast_to(pos, (3, batch, seq))
        return inputs_embeds, position_ids, rope_deltas

    def _merged_cos_sin(self, position_ids):
        return glm_moe_mrope_cos_sin(
            position_ids, self.rotary_dim, self.rope_theta, self.mrope_section
        )

    def build_for_transfer(self):
        # Multimodal lazy build: mirror the conversion feed so the vision tower +
        # merger weights exist before a weight stream (the base text-only build
        # would leave them uncreated). from_hub_repo / converted cache.
        orig = self.image_size // self.patch_size
        n = (orig // self.spatial_merge_size) ** 2
        patch_dim = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        self(
            {
                "input_ids": ops.concatenate(
                    [
                        ops.zeros((1, 1), dtype="int32"),
                        ops.full((1, n), self.image_token_id, dtype="int32"),
                        ops.ones((1, 1), dtype="int32"),
                    ],
                    axis=1,
                ),
                "pixel_values": ops.zeros((orig * orig, patch_dim), dtype="float32"),
                "image_grid_thw": ops.convert_to_tensor(
                    [[1, orig, orig]], dtype="int32"
                ),
            }
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        tc = hf_config.get("text_config", hf_config)
        vc = hf_config.get("vision_config", {})
        rope = tc.get("rope_parameters") or {}
        mrope = rope.get("mrope_section", [8, 12, 12])
        prf = rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.5))
        return {
            "vocab_size": tc["vocab_size"],
            "embed_dim": tc["hidden_size"],
            "mlp_dim": tc["intermediate_size"],
            "moe_mlp_dim": tc.get("moe_intermediate_size", 1408),
            "num_layers": tc["num_hidden_layers"],
            "num_heads": tc["num_attention_heads"],
            "num_kv_heads": tc.get("num_key_value_heads", tc["num_attention_heads"]),
            "head_dim": tc.get("head_dim"),
            "num_experts": tc.get("n_routed_experts", 128),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "n_shared_experts": tc.get("n_shared_experts", 1),
            "n_group": tc.get("n_group") or 1,
            "topk_group": tc.get("topk_group") or 1,
            "norm_topk_prob": bool(tc.get("norm_topk_prob", True)),
            "routed_scaling_factor": tc.get("routed_scaling_factor", 1.0),
            "first_k_dense": tc.get("first_k_dense_replace", 1),
            "partial_rotary_factor": prf,
            "norm_eps": tc.get("rms_norm_eps", 1e-5),
            "rope_theta": rope.get("rope_theta", tc.get("rope_theta", 10000.0)),
            "mrope_section": tuple(mrope),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
            "vision_depth": vc.get("depth", 24),
            "vision_embed_dim": vc.get("hidden_size", 1536),
            "vision_num_heads": vc.get("num_heads", 12),
            "vision_mlp_dim": vc.get("intermediate_size", 13696),
            "vision_out_dim": vc.get("out_hidden_size", tc["hidden_size"]),
            "image_size": vc.get("image_size", 336),
            "patch_size": vc.get("patch_size", 14),
            "spatial_merge_size": vc.get("spatial_merge_size", 2),
            "temporal_patch_size": vc.get("temporal_patch_size", 2),
            "in_channels": vc.get("in_channels", 3),
            "vision_norm_eps": vc.get("rms_norm_eps", 1e-5),
            "image_token_id": hf_config.get("image_token_id", 151363),
            "video_token_id": hf_config.get("video_token_id", 151364),
            "image_start_token_id": hf_config.get("image_start_token_id", 151339),
            "image_end_token_id": hf_config.get("image_end_token_id", 151340),
            "video_start_token_id": hf_config.get("video_start_token_id", 151341),
            "video_end_token_id": hf_config.get("video_end_token_id", 151342),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_glm4v_moe_hf_to_keras import transfer_glm4v_moe_weights

        transfer_glm4v_moe_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense": self.first_k_dense,
                "partial_rotary_factor": self.partial_rotary_factor,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "mrope_section": self.mrope_section,
                "tie_embeddings": self.tie_embeddings,
                "vision_depth": self.vision_depth,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_out_dim": self.vision_out_dim,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "vision_norm_eps": self.vision_norm_eps,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "image_start_token_id": self.image_start_token_id,
                "image_end_token_id": self.image_end_token_id,
                "video_start_token_id": self.video_start_token_id,
                "video_end_token_id": self.video_end_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vMoeConditionalGenerate(Glm4vMoeModel, BaseGeneration):
    """GLM-4.5V with an LM head + fast ``.generate()`` (image+text -> text)."""

    eos_token_id = (151329,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(
            hidden, ops.transpose(self.language_model.token_embedding.embeddings)
        )

    def build_cache(
        self, token_ids, padding_mask, max_len, pixel_values=None, image_grid_thw=None
    ):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv, hd = self.num_kv_heads, self.head_dim
        inputs_embeds, position_ids, rope_deltas = self._prepare_inputs(
            token_ids, pixel_values, image_grid_thw, padding_mask
        )
        cos, sin = self._merged_cos_sin(position_ids)
        causal = self._causal_mask(
            prompt_len, prompt_len, offset=0, attention_mask=padding_mask
        )
        hidden, kv = self.language_model(
            inputs_embeds, cos, sin, attention_mask=causal, use_cache=True
        )
        layer_caches = []
        for k, v in kv:
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        kv_cache = ops.stack(layer_caches, axis=1)
        logits = self.project(hidden[:, -1, :])
        return (kv_cache, rope_deltas), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        kv_cache, rope_deltas = cache
        batch = int(token_ids.shape[0])
        max_len = int(kv_cache.shape[4])
        pos = ops.broadcast_to(
            ops.reshape(cache_update_index + rope_deltas, (1, batch, 1)), (3, batch, 1)
        )
        cos, sin = self._merged_cos_sin(pos)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= cache_update_index, 0.0, MASK_NEG),
            "float32",
        )[None, None, None, :]
        h = self.language_model.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.language_model.decoder_layers):
            h, ck, cv = layer.decode_step(
                h,
                cos,
                sin,
                kv_cache[:, i, 0],
                kv_cache[:, i, 1],
                cache_update_index,
                key_mask,
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        kv_cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.language_model.final_norm(h))[:, 0, :]
        return logits, (kv_cache, rope_deltas)
