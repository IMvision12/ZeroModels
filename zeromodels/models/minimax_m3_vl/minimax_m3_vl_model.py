import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
    merge_media,
)
from zeromodels.base.base_mixin import inference_scope

from .minimax_m3_vl_config import MINIMAX_M3_VL_CONFIG, MINIMAX_M3_VL_WEIGHTS_URLS
from .minimax_m3_vl_layers import (
    MASK_NEG,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLVisionLayer,
    apply_partial_rope,
    vision_rope_3d,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLVisionModel(layers.Layer):
    """MiniMax-M3 vision tower: Conv3d-equivalent patch embed + 3D RoPE +
    CLIP-style pre-LN blocks over the packed patch sequence (no final norm).

    Args:
        embed_dim / mlp_dim / num_layers / num_heads: Tower dims.
        patch_size / temporal_patch_size / spatial_merge_size: Patch geometry.
        rope_theta: 3D-rope base frequency (1e4).
        norm_eps: LayerNorm epsilon (1e-5).

    Call args:
        pixel_values: packed patches ``(num_patches, C * t * p * p)``.
        grid_thw: host-side list of per-image ``(t, h, w)`` grids.

    Returns:
        ``(num_patches, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim=1280,
        mlp_dim=5120,
        num_layers=32,
        num_heads=16,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        rope_theta=10000.0,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.head_dim = embed_dim // num_heads

        self.patch_embed = layers.Dense(embed_dim, use_bias=False, name="patch_embed")
        self.pre_norm = layers.LayerNormalization(epsilon=norm_eps, name="pre_norm")
        self.blocks = [
            MiniMaxM3VLVisionLayer(
                embed_dim, mlp_dim, num_heads, norm_eps, name=f"blocks_{i}"
            )
            for i in range(num_layers)
        ]

    def build(self, input_shape):
        # input_shape = (num_patches, patch_dim) packed patches.
        self.patch_embed.build(input_shape)
        self.pre_norm.build((None, self.embed_dim))
        for block in self.blocks:
            block.build((None, None, self.embed_dim))
        self.built = True

    def call(self, pixel_values, grid_thw=None):
        # grid_thw drives host-side rope construction (``for t,h,w in grid_thw``),
        # so it must be a concrete iterable. In the functional graph this call runs
        # eagerly (see compute_output_spec), so a tensor grid is materialized here.
        if not isinstance(grid_thw, (list, tuple)):
            import numpy as np

            grid_thw = (
                np.asarray(ops.convert_to_numpy(grid_thw)).astype("int64").tolist()
            )
        cos, sin = vision_rope_3d(
            grid_thw, self.head_dim, self.rope_theta, self.spatial_merge_size
        )
        x = self.patch_embed(pixel_values)
        x = self.pre_norm(x)[None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return x[0]

    def compute_output_spec(self, pixel_values, grid_thw=None):
        # Merged-patch count is data-dependent (grid), so the leading axis is
        # dynamic; keeping an explicit spec lets this tower live in the functional
        # graph while its grid-iterating call runs eagerly at run time.
        return keras.KerasTensor((None, self.embed_dim), dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "rope_theta": self.rope_theta,
                "norm_eps": self.norm_eps,
            }
        )
        return config


def minimax_m3_vl_rope_tables(position_ids, rotary_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def minimax_m3_vl_image_features(
    pixel_values,
    grid_thw,
    *,
    vision_tower,
    projector_linear_1,
    projector_linear_2,
    projector_merge_linear_1,
    projector_merge_linear_2,
    spatial_merge_size,
    embed_dim,
):
    features = vision_tower(pixel_values, grid_thw)
    h = projector_linear_2(ops.gelu(projector_linear_1(features), approximate=False))
    h = ops.reshape(h, (-1, spatial_merge_size**2 * embed_dim))
    return projector_merge_linear_2(
        ops.gelu(projector_merge_linear_1(h), approximate=False)
    )


def minimax_m3_vl_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
    pixel_values_videos,
    video_grid_thw,
    *,
    token_embedding,
    vision_tower,
    projector_linear_1,
    projector_linear_2,
    projector_merge_linear_1,
    projector_merge_linear_2,
    image_merge,
    video_merge,
    decoder_layers,
    final_norm,
    causal_mask,
    rotary_dim,
    rope_theta,
    spatial_merge_size,
    embed_dim,
    image_token_id,
    video_token_id,
    compute_dtype,
):
    media = ops.logical_or(
        ops.equal(input_ids, image_token_id),
        ops.equal(input_ids, video_token_id),
    )
    hidden = token_embedding(ops.where(media, 0, input_ids))
    proj = {
        "vision_tower": vision_tower,
        "projector_linear_1": projector_linear_1,
        "projector_linear_2": projector_linear_2,
        "projector_merge_linear_1": projector_merge_linear_1,
        "projector_merge_linear_2": projector_merge_linear_2,
        "spatial_merge_size": spatial_merge_size,
        "embed_dim": embed_dim,
    }
    image_feats = minimax_m3_vl_image_features(pixel_values, image_grid_thw, **proj)
    hidden = image_merge(hidden, input_ids, image_feats)
    video_feats = minimax_m3_vl_image_features(
        pixel_values_videos, video_grid_thw, **proj
    )
    hidden = video_merge(hidden, input_ids, video_feats)

    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = minimax_m3_vl_rope_tables(
        position_ids, rotary_dim, rope_theta, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask, position_ids=position_ids)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLModel(BaseModel):
    """MiniMax-M3 vision-language backbone (MiniMaxAI/MiniMax-M3).

    The packed vision features are projected per patch (GELU MLP to the text
    width), grouped ``spatial_merge_size**2`` patches at a time, fused by a
    second GELU MLP, and scattered into the ``image_token_id`` /
    ``video_token_id`` slots of the M3 text decoder: a 60-layer GQA model
    with per-head Gemma QK-norms, partial RoPE (64 of 128 channels),
    Lightning-Indexer block-sparse attention on the sparse layers, and
    per-layer dense SwiGLU-OAI MLPs or sigmoid-routed MoE (top-4 of 128, x2
    routed scaling, shared expert). Returns raw features; use
    :class:`MiniMaxM3VLConditionalGenerate` for logits / text.

    Args:
        vocab_size / embed_dim / mlp_dim / dense_mlp_dim / shared_mlp_dim /
        num_layers / num_heads / num_kv_heads / head_dim: Text dims.
        num_experts / num_experts_per_tok / routed_scaling_factor: MoE shape.
        layer_types: Per-layer ``"full_attention"`` / ``"minimax_m3_sparse"``.
        mlp_layer_types: Per-layer ``"dense"`` / ``"sparse"``.
        index_n_heads / index_head_dim / index_block_size /
        index_topk_blocks / index_local_blocks: Indexer geometry.
        swiglu_alpha / swiglu_limit: SwiGLU-OAI constants.
        partial_rotary_factor / rope_theta: Text rope (0.5 / 5e6).
        norm_eps: Text RMSNorm epsilon.
        vision_embed_dim / vision_mlp_dim / vision_num_layers /
        vision_num_heads / patch_size / temporal_patch_size /
        spatial_merge_size / vision_rope_theta / vision_norm_eps: Tower dims.
        projector_dim: Projector MLP hidden width.
        image_token_id / video_token_id: Placeholder ids (200025 / 200026).
        tie_embeddings: Whether :class:`MiniMaxM3VLConditionalGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "minimax_m3_vl"
    BASE_MODEL_CONFIG = MINIMAX_M3_VL_CONFIG
    BASE_WEIGHT_CONFIG = MINIMAX_M3_VL_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=200064,
        embed_dim=6144,
        mlp_dim=3072,
        dense_mlp_dim=12288,
        shared_mlp_dim=3072,
        num_layers=60,
        num_heads=64,
        num_kv_heads=4,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=4,
        routed_scaling_factor=2.0,
        layer_types=None,
        mlp_layer_types=None,
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        partial_rotary_factor=0.5,
        rope_theta=5000000.0,
        norm_eps=1e-6,
        vision_embed_dim=1280,
        vision_mlp_dim=5120,
        vision_num_layers=32,
        vision_num_heads=16,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        vision_rope_theta=10000.0,
        vision_norm_eps=1e-5,
        projector_dim=6144,
        image_token_id=200025,
        video_token_id=200026,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        if layer_types is None:
            layer_types = tuple("full_attention" for _ in range(num_layers))
        if mlp_layer_types is None:
            mlp_layer_types = tuple("sparse" for _ in range(num_layers))
        layer_types = tuple(layer_types)
        mlp_layer_types = tuple(mlp_layer_types)
        head_dim = head_dim or embed_dim // num_heads
        rotary_dim = int(head_dim * partial_rotary_factor)
        patch_dim = 3 * temporal_patch_size * patch_size**2

        vision_tower = MiniMaxM3VLVisionModel(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            vision_rope_theta,
            vision_norm_eps,
            name="vision_tower",
        )
        projector_linear_1 = layers.Dense(projector_dim, name="projector_linear_1")
        projector_linear_2 = layers.Dense(embed_dim, name="projector_linear_2")
        projector_merge_linear_1 = layers.Dense(
            projector_dim, name="projector_merge_linear_1"
        )
        projector_merge_linear_2 = layers.Dense(
            embed_dim, name="projector_merge_linear_2"
        )
        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            MiniMaxM3VLDecoderLayer(
                embed_dim,
                mlp_dim,
                dense_mlp_dim,
                shared_mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_experts,
                num_experts_per_tok,
                mlp_type=mlp_layer_types[i],
                routed_scaling_factor=routed_scaling_factor,
                use_indexer=layer_types[i] == "minimax_m3_sparse",
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_block_size=index_block_size,
                index_topk_blocks=index_topk_blocks,
                index_local_blocks=index_local_blocks,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = MiniMaxM3VLRMSNorm(eps=norm_eps, name="final_norm")
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        video_merge = MediaMerge(video_token_id, embed_dim, name="video_merge")
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
            "pixel_values_videos": layers.Input(
                shape=(patch_dim,), dtype="float32", name="pixel_values_videos"
            ),
            "video_grid_thw": layers.Input(
                shape=(3,), dtype="int32", name="video_grid_thw"
            ),
        }
        hidden = minimax_m3_vl_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["pixel_values_videos"],
            inputs["video_grid_thw"],
            token_embedding=token_embedding,
            vision_tower=vision_tower,
            projector_linear_1=projector_linear_1,
            projector_linear_2=projector_linear_2,
            projector_merge_linear_1=projector_merge_linear_1,
            projector_merge_linear_2=projector_merge_linear_2,
            image_merge=image_merge,
            video_merge=video_merge,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            spatial_merge_size=spatial_merge_size,
            embed_dim=embed_dim,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            compute_dtype=token_embedding.compute_dtype,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(token_embedding, name="lm_head")(hidden)
            )

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.vision_tower = vision_tower
        self.projector_linear_1 = projector_linear_1
        self.projector_linear_2 = projector_linear_2
        self.projector_merge_linear_1 = projector_merge_linear_1
        self.projector_merge_linear_2 = projector_merge_linear_2
        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.image_merge = image_merge
        self.video_merge = video_merge
        self.causal_mask_layer = causal_mask
        self.lm_head = lm_head
        self.rotary_dim = rotary_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dense_mlp_dim = dense_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.layer_types = tuple(layer_types)
        self.mlp_layer_types = tuple(mlp_layer_types)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.index_local_blocks = index_local_blocks
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.vision_rope_theta = vision_rope_theta
        self.vision_norm_eps = vision_norm_eps
        self.projector_dim = projector_dim
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.tie_embeddings = tie_embeddings

        # Vision-tower / sparse-indexer sublayers do not all auto-build during the
        # symbolic functional build; a concrete dummy forward materializes them so
        # from_weights (which loads before any forward) has a complete model.
        with inference_scope():
            self(self.dummy_media_inputs())

    def dummy_media_inputs(self):
        sms = self.spatial_merge_size
        patch_dim = 3 * self.temporal_patch_size * self.patch_size**2
        ids = ops.convert_to_tensor(
            [[0, self.image_token_id, self.video_token_id, 1]], dtype="int32"
        )
        media = ops.zeros((sms * sms, patch_dim), dtype="float32")
        grid = ops.convert_to_tensor([[1, sms, sms]], dtype="int32")
        return {
            "input_ids": ids,
            "attention_mask": ops.ones_like(ids),
            "pixel_values": media,
            "image_grid_thw": grid,
            "pixel_values_videos": media,
            "video_grid_thw": grid,
        }

    def get_image_features(self, pixel_values, grid_thw):
        features = self.vision_tower(pixel_values, grid_thw=grid_thw)
        h = self.projector_linear_2(
            ops.gelu(self.projector_linear_1(features), approximate=False)
        )
        merge = self.spatial_merge_size**2
        h = ops.reshape(h, (-1, merge * self.embed_dim))
        return self.projector_merge_linear_2(
            ops.gelu(self.projector_merge_linear_1(h), approximate=False)
        )

    def rope_tables(self, position_ids):
        rd = self.rotary_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, rd, 2, dtype="float32") / rd
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def scatter_features(self, inputs_embeds, input_ids, features, token_id):
        # Graph-safe cumsum-gather-where scatter (shared with the functional graph).
        return merge_media(inputs_embeds, input_ids, features, token_id, self.embed_dim)

    def host_grid(self, grid_thw):
        if isinstance(grid_thw, (list, tuple)):
            return [list(map(int, g)) for g in grid_thw]
        import numpy as np

        return np.asarray(ops.convert_to_numpy(grid_thw)).astype("int64").tolist()

    def prepare_inputs(self, inputs):
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        media = ops.logical_or(
            ops.equal(input_ids, self.image_token_id),
            ops.equal(input_ids, self.video_token_id),
        )
        hidden = self.token_embedding(ops.where(media, 0, input_ids))
        if inputs.get("pixel_values") is not None:
            features = self.get_image_features(
                ops.convert_to_tensor(inputs["pixel_values"]),
                self.host_grid(inputs["image_grid_thw"]),
            )
            hidden = self.scatter_features(
                hidden, input_ids, features, self.image_token_id
            )
        if inputs.get("pixel_values_videos") is not None:
            features = self.get_image_features(
                ops.convert_to_tensor(inputs["pixel_values_videos"]),
                self.host_grid(inputs["video_grid_thw"]),
            )
            hidden = self.scatter_features(
                hidden, input_ids, features, self.video_token_id
            )
        return hidden

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        rope = text.get("rope_parameters") or {}
        sparse = text.get("sparse_attention_config") or {}
        layer_types = text.get("layer_types")
        if layer_types is None and "sparse_attention_freq" in sparse:
            layer_types = [
                "minimax_m3_sparse" if f else "full_attention"
                for f in sparse["sparse_attention_freq"]
            ]
        mlp_layer_types = text.get("mlp_layer_types")
        if mlp_layer_types is None and text.get("moe_layer_freq") is not None:
            mlp_layer_types = [
                "sparse" if f else "dense" for f in text["moe_layer_freq"]
            ]
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "dense_mlp_dim": text.get("dense_intermediate_size", 12288),
            "shared_mlp_dim": text.get("shared_intermediate_size", 3072),
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim"),
            "num_experts": text.get("num_local_experts", 128),
            "num_experts_per_tok": text.get("num_experts_per_tok", 4),
            "routed_scaling_factor": text.get("routed_scaling_factor", 2.0),
            "layer_types": tuple(layer_types) if layer_types else None,
            "mlp_layer_types": tuple(mlp_layer_types) if mlp_layer_types else None,
            "index_n_heads": sparse.get(
                "sparse_num_index_heads", text.get("index_n_heads", 4)
            ),
            "index_head_dim": sparse.get(
                "sparse_index_dim", text.get("index_head_dim", 128)
            ),
            "index_block_size": sparse.get(
                "sparse_block_size", text.get("index_block_size", 128)
            ),
            "index_topk_blocks": sparse.get(
                "sparse_topk_blocks", text.get("index_topk_blocks", 16)
            ),
            "index_local_blocks": sparse.get(
                "sparse_local_block", text.get("index_local_blocks", 1)
            ),
            "swiglu_alpha": text.get("swiglu_alpha", 1.702),
            "swiglu_limit": text.get("swiglu_limit", 7.0),
            "partial_rotary_factor": rope.get(
                "partial_rotary_factor", text.get("partial_rotary_factor", 0.5)
            ),
            "rope_theta": rope.get("rope_theta", text.get("rope_theta", 5000000.0)),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "patch_size": vision.get("patch_size", 14),
            "temporal_patch_size": vision.get("temporal_patch_size", 2),
            "spatial_merge_size": vision.get("spatial_merge_size", 2),
            "vision_rope_theta": (vision.get("rope_parameters") or {}).get(
                "rope_theta", vision.get("rope_theta", 10000.0)
            ),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-5),
            "projector_dim": hf_config.get("projector_hidden_size", 6144),
            "image_token_id": hf_config.get(
                "image_token_id", hf_config.get("image_token_index", 200025)
            ),
            "video_token_id": hf_config.get(
                "video_token_id", hf_config.get("video_token_index", 200026)
            ),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_minimax_m3_vl_hf_to_keras import transfer_minimax_m3_vl_weights

        transfer_minimax_m3_vl_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "dense_mlp_dim": self.dense_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "routed_scaling_factor": self.routed_scaling_factor,
                "layer_types": self.layer_types,
                "mlp_layer_types": self.mlp_layer_types,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_block_size": self.index_block_size,
                "index_topk_blocks": self.index_topk_blocks,
                "index_local_blocks": self.index_local_blocks,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_theta": self.rope_theta,
                "norm_eps": self.norm_eps,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "vision_rope_theta": self.vision_rope_theta,
                "vision_norm_eps": self.vision_norm_eps,
                "projector_dim": self.projector_dim,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLConditionalGenerate(MiniMaxM3VLModel, BaseGeneration):
    """MiniMax-M3 VL with an LM head + fast ``.generate()`` (image+text -> text).

    ``build_cache`` runs the vision tower + projector + prefill once
    (consuming pixel inputs); the per-layer cache is ``(kv,)`` for dense
    layers and ``(kv, indexer_keys)`` for sparse layers, so the decode loop
    re-runs the Lightning-Indexer block selection against the cached indexer
    keys at every step:

        gen.generate(input_ids, pixel_values=..., image_grid_thw=...)
    """

    # MiniMax-M3 eos `[e~[`. Explicit generate() args override.
    eos_token_id = (200020,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden = self.prepare_inputs(
            {
                "input_ids": token_ids,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }
        )
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            hidden, piece = layer(
                hidden,
                cos,
                sin,
                attention_mask=causal,
                position_ids=position_ids,
                use_cache=True,
            )
            k, v = piece[0], piece[1]
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            kv = ops.stack([ck, cv], axis=1)
            if len(piece) == 3:
                idx_k = piece[2]
                cidx = ops.slice_update(
                    ops.zeros(
                        (batch, 1, max_len, self.index_head_dim), dtype=idx_k.dtype
                    ),
                    (0, 0, 0, 0),
                    idx_k,
                )
                caches.append((kv, cidx))
            else:
                caches.append((kv,))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0][0].shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_ok = ops.arange(max_len) <= pos
        key_mask = ops.cast(ops.where(key_ok, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            residual = h
            x = layer.attention_norm(h)
            attention = layer.attention
            piece = cache[i]
            if attention.use_indexer:
                idx_cache = piece[1]
                idx_k_new = attention.project_index_k(x, cos, sin)
                # rope runs in float32; match the cache dtype before writing
                idx_k_new = ops.cast(idx_k_new, idx_cache.dtype)
                idx_cache = ops.slice_update(idx_cache, (0, 0, pos, 0), idx_k_new)
                idx_q = attention.project_index_q(x, cos, sin)
                keep = attention.block_keep_mask(
                    idx_q, idx_cache, positions, ops.arange(max_len)
                )
                mask = ops.where(
                    ops.logical_and(keep, key_ok[None, None, None, :]), 0.0, MASK_NEG
                )
            else:
                mask = key_mask
            q, k, v = attention.project_qkv(x)
            cos_e = ops.expand_dims(cos, axis=1)
            sin_e = ops.expand_dims(sin, axis=1)
            q = apply_partial_rope(q, cos_e, sin_e)
            k = apply_partial_rope(k, cos_e, sin_e)
            kv = piece[0]
            ck = ops.slice_update(kv[:, 0], (0, 0, pos, 0), k)
            cv = ops.slice_update(kv[:, 1], (0, 0, pos, 0), v)
            kk, vv = ck, cv
            if attention.num_kv_groups > 1:
                kk = ops.repeat(kk, attention.num_kv_groups, axis=1)
                vv = ops.repeat(vv, attention.num_kv_groups, axis=1)
            attn = ops.matmul(q, ops.transpose(kk, (0, 1, 3, 2))) * attention.scaling
            attn = attn + ops.cast(mask, attn.dtype)
            attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
            out = ops.matmul(attn, vv)
            out = ops.reshape(
                ops.transpose(out, (0, 2, 1, 3)),
                (batch, 1, attention.num_heads * attention.head_dim),
            )
            attn_out = attention.output_proj(out)
            h = residual + attn_out
            h = h + layer.mlp(layer.mlp_norm(h))
            new_kv = ops.stack([ck, cv], axis=1)
            if attention.use_indexer:
                new_cache.append((new_kv, idx_cache))
            else:
                new_cache.append((new_kv,))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
