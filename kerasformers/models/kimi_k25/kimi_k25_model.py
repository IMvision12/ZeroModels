import keras
from keras import layers, ops

from kerasformers.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
    merge_media,
)
from kerasformers.base.base_mixin import inference_scope
from kerasformers.models.deepseek_v3.deepseek_v3_layers import (
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm,
    yarn_get_mscale,
)
from kerasformers.models.deepseek_v3.deepseek_v3_model import (
    DeepseekV3Model,
    deepseek_v3_rope_tables,
)

from .kimi_k25_config import KIMI_K25_CONFIG, KIMI_K25_WEIGHTS_URLS
from .kimi_k25_layers import KimiK25MultimodalProjection
from .kimi_k25_vision import KimiK25VisionModel

MASK_NEG = -1e9


def kimi_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
    pixel_values_videos,
    video_grid_thw,
    *,
    token_embedding,
    vision_tower,
    mm_projector,
    image_merge,
    video_merge,
    decoder_layers,
    final_norm,
    causal_mask,
    inv_freq,
    attention_scaling,
    image_token_id,
    video_token_id,
    compute_dtype,
):
    # Media placeholders (image/video token ids) are out of / masked from the
    # embedding lookup, then overwritten by the projected vision features. Both
    # media streams are always present as graph inputs (KerasHub style); a stream
    # whose token does not appear in input_ids merges as a no-op.
    media = ops.logical_or(
        ops.equal(input_ids, image_token_id),
        ops.equal(input_ids, video_token_id),
    )
    hidden = token_embedding(ops.where(media, 0, input_ids))
    image_embeds = mm_projector(vision_tower(pixel_values, image_grid_thw))
    hidden = image_merge(hidden, input_ids, image_embeds)
    video_embeds = mm_projector(vision_tower(pixel_values_videos, video_grid_thw))
    hidden = video_merge(hidden, input_ids, video_embeds)

    # Kimi (like DeepSeek-V3) uses plain arange positions.
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = deepseek_v3_rope_tables(
        position_ids, inv_freq, attention_scaling, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25Model(BaseModel):
    """Kimi K2.5 multimodal backbone (MoonViT vision tower + DeepSeek-V3 text).

    A functional model: the vision tower and the image/video -> text token merge
    run inside the graph over ``{input_ids, attention_mask, pixel_values,
    image_grid_thw, pixel_values_videos, video_grid_thw}`` (media inputs are
    always present; an absent media token merges as a no-op). The text tower reuses
    DeepSeek-V3's MLA + DeepSeekMoE decoder layers. Returns ``last_hidden_state``;
    use :class:`KimiK25ConditionalGenerate` for logits / text.
    """

    HF_MODEL_TYPE = "kimi_k25"
    BASE_MODEL_CONFIG = KIMI_K25_CONFIG
    BASE_WEIGHT_CONFIG = KIMI_K25_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=163840,
        embed_dim=7168,
        num_layers=61,
        num_heads=64,
        mlp_dim=18432,
        moe_mlp_dim=2048,
        num_experts=384,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.827,
        first_k_dense=1,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=50000.0,
        rope_scaling=None,
        norm_eps=1e-5,
        max_position_embeddings=262144,
        tie_embeddings=False,
        vision_embed_dim=1152,
        vision_depth=27,
        vision_num_heads=16,
        vision_mlp_dim=4304,
        vision_patch_size=14,
        pos_emb_height=64,
        pos_emb_width=64,
        pos_emb_time=4,
        merge_kernel=(2, 2),
        vision_rope_theta=10000.0,
        projection_hidden_size=1152,
        projection_norm_eps=1e-5,
        image_token_id=163605,
        video_token_id=163840,
        vision_start_token_id=163602,
        vision_end_token_id=163604,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        rope_scaling = dict(rope_scaling) if rope_scaling else None
        merge_kernel = tuple(merge_kernel)

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        softmax_scale = qk_head_dim**-0.5
        # Like DeepSeek-V3, the yarn mscale^2 correction is folded into the scale.
        scaling_cfg = rope_scaling or {}
        rope_type = scaling_cfg.get("rope_type", scaling_cfg.get("type", "default"))
        if rope_type != "default":
            mscale_all_dim = scaling_cfg.get("mscale_all_dim", 0)
            factor = scaling_cfg.get("factor")
            if factor is None and scaling_cfg.get("original_max_position_embeddings"):
                factor = (
                    max_position_embeddings
                    / scaling_cfg["original_max_position_embeddings"]
                )
            if mscale_all_dim and factor:
                mscale = yarn_get_mscale(factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale

        inv_freq, attention_scaling = DeepseekV3Model.build_rope(
            qk_rope_head_dim, rope_theta, rope_scaling, max_position_embeddings
        )

        vision_tower = KimiK25VisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_dim=vision_mlp_dim,
            patch_size=vision_patch_size,
            pos_emb_height=pos_emb_height,
            pos_emb_width=pos_emb_width,
            pos_emb_time=pos_emb_time,
            merge_kernel=merge_kernel,
            rope_theta=vision_rope_theta,
            name="vision_tower",
        )
        mm_projector = KimiK25MultimodalProjection(
            vision_embed_dim * merge_kernel[0] * merge_kernel[1],
            embed_dim,
            norm_eps=projection_norm_eps,
            name="mm_projector",
        )
        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            DeepseekV3DecoderLayer(
                embed_dim,
                num_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                softmax_scale,
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
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = DeepseekV3RMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        video_merge = MediaMerge(video_token_id, embed_dim, name="video_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        patch_shape = (3, vision_patch_size, vision_patch_size)
        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=patch_shape, dtype="float32", name="pixel_values"
            ),
            "image_grid_thw": layers.Input(
                shape=(3,), dtype="int32", name="image_grid_thw"
            ),
            "pixel_values_videos": layers.Input(
                shape=patch_shape, dtype="float32", name="pixel_values_videos"
            ),
            "video_grid_thw": layers.Input(
                shape=(3,), dtype="int32", name="video_grid_thw"
            ),
        }
        hidden = kimi_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["pixel_values_videos"],
            inputs["video_grid_thw"],
            token_embedding=token_embedding,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_merge=image_merge,
            video_merge=video_merge,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
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
        self.mm_projector = mm_projector
        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.causal_mask_layer = causal_mask
        self.image_merge = image_merge
        self.video_merge = video_merge
        self.lm_head = lm_head
        self.inv_freq = inv_freq
        self.attention_scaling = attention_scaling
        self.qk_head_dim = qk_head_dim
        self.softmax_scale = softmax_scale
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_patch_size = vision_patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.pos_emb_time = pos_emb_time
        self.merge_kernel = merge_kernel
        self.vision_rope_theta = vision_rope_theta
        self.projection_hidden_size = projection_hidden_size
        self.projection_norm_eps = projection_norm_eps
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # The vision tower's grid-iterating call is skipped by the symbolic
        # auto-build, so its blocks stay unbuilt; a concrete dummy forward
        # materializes every weight for from_weights (which loads before any call).
        with inference_scope():
            self(self.dummy_media_inputs())

    def dummy_media_inputs(self):
        # One 2x2 image + one 2x2 video patch clip (grid t=1,h=2,w=2 -> 4 patches
        # -> a single merged token each), matched by one image + one video token.
        p = ops.zeros((4, 3, self.vision_patch_size, self.vision_patch_size), "float32")
        grid = ops.convert_to_tensor([[1, 2, 2]], dtype="int32")
        ids = ops.convert_to_tensor(
            [[self.image_token_id, self.video_token_id, 0, 0]], dtype="int32"
        )
        return {
            "input_ids": ids,
            "attention_mask": ops.ones((1, 4), dtype="int32"),
            "pixel_values": p,
            "image_grid_thw": grid,
            "pixel_values_videos": p,
            "video_grid_thw": grid,
        }

    def build_for_transfer(self):
        with inference_scope():
            self(self.dummy_media_inputs())

    def rope_tables(self, position_ids):
        return deepseek_v3_rope_tables(
            position_ids, self.inv_freq, self.attention_scaling, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def get_image_features(self, pixel_values, grid_thw):
        return self.mm_projector(self.vision_tower(pixel_values, grid_thw))

    def merge_media(self, hidden, input_ids, features, token_id):
        return merge_media(hidden, input_ids, features, token_id, self.embed_dim)

    def embed_inputs(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
    ):
        # Imperative merge for the KV-cache prefill: handles absent media (None).
        media = ops.logical_or(
            ops.equal(input_ids, self.image_token_id),
            ops.equal(input_ids, self.video_token_id),
        )
        hidden = self.token_embedding(ops.where(media, 0, input_ids))
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            hidden = self.merge_media(
                hidden, input_ids, image_embeds, self.image_token_id
            )
        if pixel_values_videos is not None:
            video_embeds = self.get_image_features(pixel_values_videos, video_grid_thw)
            hidden = self.merge_media(
                hidden, input_ids, video_embeds, self.video_token_id
            )
        return hidden

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config") or {}
        vision = hf_config.get("vision_config") or {}

        def vision_get(default, *names):
            for name in names:
                if name in vision:
                    return vision[name]
            return default

        config = DeepseekV3Model.config_from_hf(text)
        merge = tuple(vision_get((2, 2), "merge_kernel_size"))
        vision_rope = vision.get("rope_parameters") or {}
        config.update(
            {
                "tie_embeddings": bool(
                    hf_config.get(
                        "tie_word_embeddings", text.get("tie_word_embeddings")
                    )
                    or False
                ),
                "vision_embed_dim": vision_get(1152, "hidden_size", "vt_hidden_size"),
                "vision_depth": vision_get(
                    27, "num_hidden_layers", "vt_num_hidden_layers"
                ),
                "vision_num_heads": vision_get(
                    16, "num_attention_heads", "vt_num_attention_heads"
                ),
                "vision_mlp_dim": vision_get(
                    4304, "intermediate_size", "vt_intermediate_size"
                ),
                "vision_patch_size": vision_get(14, "patch_size"),
                "pos_emb_height": vision_get(
                    64, "pos_emb_height", "init_pos_emb_height"
                ),
                "pos_emb_width": vision_get(64, "pos_emb_width", "init_pos_emb_width"),
                "pos_emb_time": vision_get(4, "pos_emb_time", "init_pos_emb_time"),
                "merge_kernel": merge,
                "vision_rope_theta": vision_rope.get("rope_theta", 10000.0),
                "projection_hidden_size": hf_config.get("projection_hidden_size", 1152),
                "projection_norm_eps": hf_config.get("projection_layer_norm_eps")
                or vision.get("projector_ln_eps")
                or 1e-5,
                "image_token_id": hf_config.get(
                    "image_token_id",
                    hf_config.get("media_placeholder_token_id", 163605),
                ),
                "video_token_id": hf_config.get("video_token_id", 163840),
                "vision_start_token_id": hf_config.get("vision_start_token_id", 163602),
                "vision_end_token_id": hf_config.get("vision_end_token_id", 163604),
            }
        )
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_kimi_k25_hf_to_keras import transfer_kimi_k25_weights

        transfer_kimi_k25_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense": self.first_k_dense,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "norm_eps": self.norm_eps,
                "max_position_embeddings": self.max_position_embeddings,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_depth": self.vision_depth,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_patch_size": self.vision_patch_size,
                "pos_emb_height": self.pos_emb_height,
                "pos_emb_width": self.pos_emb_width,
                "pos_emb_time": self.pos_emb_time,
                "merge_kernel": self.merge_kernel,
                "vision_rope_theta": self.vision_rope_theta,
                "projection_hidden_size": self.projection_hidden_size,
                "projection_norm_eps": self.projection_norm_eps,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25ConditionalGenerate(KimiK25Model, BaseGeneration):
    """Kimi K2.5 with an LM head + fast ``.generate()`` (image/video+text -> text).

    Media only enters through the prefill, so ``pixel_values`` / ``image_grid_thw``
    (and the video pair) are passed to ``generate`` as prefill kwargs; decode
    steps run text-only against the MLA cache. As in DeepSeek-V3 the cache stores
    expanded per-head keys and values as a per-layer ``(k, v)`` tuple, since their
    head dims differ (k: nope+rope = 192, v: ``v_head_dim`` = 128).
    """

    # <|im_end|> (163586), per the published generation_config.
    eos_token_id = (163586,)
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
        hidden = self.embed_inputs(
            ops.cast(token_ids, "int32"),
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
            )
            ck = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.qk_head_dim), dtype=k.dtype
                ),
                (0, 0, 0, 0),
                k,
            )
            cv = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.v_head_dim), dtype=v.dtype
                ),
                (0, 0, 0, 0),
                v,
            )
            caches.append((ck, cv))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0][0].shape[2])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[i][0], cache[i][1], pos, key_mask
            )
            new_cache.append((ck, cv))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
