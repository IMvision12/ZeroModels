import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    CausalMask,
    MediaMerge,
    TiedHead,
)
from zeromodels.base.base_mixin import inference_scope
from zeromodels.models.qwen2_vl.qwen2_vl_model import (
    Qwen2VLModel,
    vision_rotary_cos_sin,
)
from zeromodels.models.qwen3_next.qwen3_next_layers import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
)
from zeromodels.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)
from zeromodels.models.qwen3_vl.qwen3_vl_model import qwen3_text_cos_sin

from .qwen3_5_moe_config import QWEN3_5_MOE_TOKENS, Qwen3_5MoeConfig

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeVisionModel(layers.Layer):
    """Qwen3.5-MoE vision tower: learned pos-embeds -> GELU blocks -> 2x2 merger.

    The Qwen3-VL ViT **without DeepStack**: a per-patch embedding, learned position
    embeddings bilinearly interpolated to each image's grid and added in, ``depth``
    full-attention GELU blocks with 2D rotary positions and a block-diagonal
    (per-image) mask, then a single 2x2 merger to ``out_hidden_size``.

    Args:
        embed_dim: Vision hidden width.
        depth: Number of vision blocks.
        num_heads: Vision attention heads.
        intermediate_size: Vision MLP hidden width.
        out_hidden_size: Output width of the merger (the LLM's hidden size).
        num_position_embeddings: Size of the learned position-embedding grid.
        hidden_act: Vision MLP activation (e.g. ``"gelu_pytorch_tanh"``).
        patch_size: Vision patch size, in pixels.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).

    Call args:
        pixel_values: Flattened patches ``(num_patches, patch_dim)``.
        grid_thw: Per-image ``(t, h, w)`` patch-grid sizes.

    Returns:
        Merged image embeddings ``(num_merged_tokens, out_hidden_size)``.
    """

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        intermediate_size,
        out_hidden_size,
        num_position_embeddings,
        hidden_act="gelu_pytorch_tanh",
        patch_size=16,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads
        self.num_grid_per_side = int(round(num_position_embeddings**0.5))

        self.patch_embed = Qwen3VLVisionPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen3VLVisionBlock(
                embed_dim, num_heads, intermediate_size, name=f"blocks_{i}"
            )
            for i in range(depth)
        ]
        self.merger = Qwen3VLVisionPatchMerger(
            out_hidden_size,
            embed_dim,
            spatial_merge_size,
            use_postshuffle_norm=False,
            name="merger",
        )

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.num_position_embeddings, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def _interp_pos_embed(self, grid_rows):
        npos = self.num_grid_per_side
        m = self.spatial_merge_size
        pieces = []
        for t, h, w in grid_rows:
            hi = ops.linspace(0.0, float(npos - 1), h)
            wi = ops.linspace(0.0, float(npos - 1), w)
            hf = ops.cast(hi, "int32")
            wf = ops.cast(wi, "int32")
            hc = ops.minimum(hf + 1, npos - 1)
            wc = ops.minimum(wf + 1, npos - 1)
            dh = (hi - ops.cast(hf, "float32"))[:, None]
            dw = (wi - ops.cast(wf, "float32"))[None, :]
            i00 = ops.reshape(hf[:, None] * npos + wf[None, :], (-1,))
            i01 = ops.reshape(hf[:, None] * npos + wc[None, :], (-1,))
            i10 = ops.reshape(hc[:, None] * npos + wf[None, :], (-1,))
            i11 = ops.reshape(hc[:, None] * npos + wc[None, :], (-1,))
            w00 = ops.reshape((1 - dh) * (1 - dw), (-1, 1))
            w01 = ops.reshape((1 - dh) * dw, (-1, 1))
            w10 = ops.reshape(dh * (1 - dw), (-1, 1))
            w11 = ops.reshape(dh * dw, (-1, 1))
            emb = (
                ops.take(self.pos_embed, i00, axis=0) * w00
                + ops.take(self.pos_embed, i01, axis=0) * w01
                + ops.take(self.pos_embed, i10, axis=0) * w10
                + ops.take(self.pos_embed, i11, axis=0) * w11
            )
            emb = ops.reshape(emb, (1, h // m, m, w // m, m, self.embed_dim))
            emb = ops.transpose(emb, (0, 1, 3, 2, 4, 5))
            emb = ops.reshape(emb, (h * w, self.embed_dim))
            if t > 1:
                emb = ops.concatenate([emb] * t, axis=0)
            pieces.append(emb)
        return ops.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]

    def _full_mask(self, grid_rows, seq):
        cu = [0]
        for t, h, w in grid_rows:
            for _ in range(t):
                cu.append(cu[-1] + h * w)
        if len(cu) <= 2:
            return None
        seg = [0] * seq
        for i in range(len(cu) - 1):
            for j in range(cu[i], cu[i + 1]):
                seg[j] = i
        seg = ops.convert_to_tensor(seg, dtype="int32")
        mask = ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG)
        return ops.cast(mask, "float32")[None, None]

    def call(self, pixel_values, grid_thw):
        grid_rows = [
            tuple(int(v) for v in row)
            for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
        ]
        seq = sum(t * h * w for t, h, w in grid_rows)
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self._interp_pos_embed(grid_rows)

        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        mask = self._full_mask(grid_rows, seq)

        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_mask=mask)
        return self.merger(hidden)

    def compute_output_spec(self, pixel_values, grid_thw):
        # Merged-token count is grid-dependent (dynamic); the grid-iterating call
        # runs eagerly at runtime.
        return keras.KerasTensor((None, self.out_hidden_size), dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "out_hidden_size": self.out_hidden_size,
                "num_position_embeddings": self.num_position_embeddings,
                "hidden_act": self.hidden_act,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeTextModel(layers.Layer):
    """Qwen3-Next hybrid MoE decoder (no LM head), driven by external M-RoPE tables.

    ``token_embedding -> num_layers x Qwen3NextDecoderLayer -> final RMSNorm``. The
    layers alternate Gated-DeltaNet linear attention and gated full attention (every
    ``full_attention_interval``-th), each followed by a sparse MoE block (routed +
    shared expert). Unlike the standalone Qwen3-Next backbone, the interleaved M-RoPE
    ``cos`` / ``sin`` tables are computed by the multimodal model and passed in (only
    the full-attention layers use them); the DeltaNet ``pad_mask`` is threaded so
    padded tokens do not leak through the recurrence.

    Args mirror the flat Qwen3.5-MoE text hyperparameters (see [`Qwen3_5MoeModel`]).

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeddings.
        cos, sin: interleaved-M-RoPE tables ``(batch, seq, rotary_dim)``.
        attention_mask: additive mask broadcastable to ``(batch, 1, q_len, kv_len)``.
        pad_mask: ``(batch, seq, 1)`` 1/0 mask for the DeltaNet layers, or ``None``.
        use_cache: when ``True``, also return the per-layer hybrid cache.

    Returns:
        ``(batch, seq, embed_dim)``, or ``(hidden, cache)`` when ``use_cache``.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        norm_eps=1e-6,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_experts=256,
        num_experts_per_tok=8,
        moe_mlp_dim=512,
        shared_mlp_dim=512,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.mlp_only_layers = tuple(mlp_only_layers)
        self.layer_types = [
            "full_attention"
            if (i + 1) % full_attention_interval == 0
            else "linear_attention"
            for i in range(num_layers)
        ]
        layer_cfg = {
            "embed_dim": embed_dim,
            "mlp_dim": mlp_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "norm_eps": norm_eps,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_mlp_dim": moe_mlp_dim,
            "shared_mlp_dim": shared_mlp_dim,
            "norm_topk_prob": norm_topk_prob,
        }
        self._layer_cfg = layer_cfg
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3NextDecoderLayer(
                layer_cfg,
                self.layer_types[i],
                use_moe=(
                    i not in self.mlp_only_layers
                    and num_experts > 0
                    and (i + 1) % decoder_sparse_step == 0
                ),
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3NextRMSNorm(eps=norm_eps, name="final_norm")

    def build(self, input_shape):
        self.token_embedding.build((input_shape[0], input_shape[1]))
        for layer in self.decoder_layers:
            layer.build(input_shape)
        self.final_norm.build(input_shape)
        self.built = True

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        pad_mask=None,
        use_cache=False,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        for layer in self.decoder_layers:
            out = layer(
                hidden,
                cos,
                sin,
                attention_mask=attention_mask,
                use_cache=use_cache,
                pad_mask=pad_mask,
            )
            if use_cache:
                hidden, state = out
                new_cache.append(state)
            else:
                hidden = out
        hidden = self.final_norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def compute_output_spec(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        pad_mask=None,
        use_cache=False,
    ):
        return keras.KerasTensor(inputs_embeds.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self._layer_cfg["mlp_dim"],
                "num_layers": self.num_layers,
                "num_heads": self._layer_cfg["num_heads"],
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "rotary_dim": self._layer_cfg["rotary_dim"],
                "norm_eps": self.norm_eps,
                "full_attention_interval": next(
                    (
                        i + 1
                        for i, t in enumerate(self.layer_types)
                        if t == "full_attention"
                    ),
                    4,
                ),
                "linear_conv_kernel_dim": self._layer_cfg["linear_conv_kernel_dim"],
                "linear_key_head_dim": self._layer_cfg["linear_key_head_dim"],
                "linear_value_head_dim": self._layer_cfg["linear_value_head_dim"],
                "linear_num_key_heads": self._layer_cfg["linear_num_key_heads"],
                "linear_num_value_heads": self._layer_cfg["linear_num_value_heads"],
                "num_experts": self._layer_cfg["num_experts"],
                "num_experts_per_tok": self._layer_cfg["num_experts_per_tok"],
                "moe_mlp_dim": self._layer_cfg["moe_mlp_dim"],
                "shared_mlp_dim": self._layer_cfg["shared_mlp_dim"],
                "norm_topk_prob": self._layer_cfg["norm_topk_prob"],
                "mlp_only_layers": self.mlp_only_layers,
            }
        )
        return config


def qwen3_5_moe_multimodal_features(
    input_ids,
    attention_mask,
    position_ids,
    pixel_values,
    image_grid_thw,
    pixel_values_videos,
    video_grid_thw,
    *,
    token_embedding,
    visual,
    language_model,
    image_merge,
    video_merge,
    causal_mask,
    rotary_dim,
    rope_theta,
    mrope_section,
    image_token_id,
    video_token_id,
):
    # Always-media multimodal graph (no DeepStack). Vision runs unconditionally
    # (no-op merge when a stream's token is absent); partial-rotary interleaved
    # M-RoPE; the DeltaNet pad_mask is threaded to the hybrid text decoder.
    media = ops.logical_or(
        ops.equal(input_ids, image_token_id), ops.equal(input_ids, video_token_id)
    )
    hidden = token_embedding(ops.where(media, 0, input_ids))
    hidden = image_merge(hidden, input_ids, visual(pixel_values, image_grid_thw))
    hidden = video_merge(hidden, input_ids, visual(pixel_values_videos, video_grid_thw))
    pos = ops.transpose(position_ids, (1, 0, 2))  # (batch, 3, seq) -> (3, batch, seq)
    cos, sin = qwen3_text_cos_sin(pos, rotary_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    pad_mask = ops.cast(attention_mask, "float32")[:, :, None]
    return language_model(hidden, cos, sin, attention_mask=mask, pad_mask=pad_mask)


def qwen3_5_moe_text_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    language_model,
    causal_mask,
    rotary_dim,
    rope_theta,
    mrope_section,
):
    hidden = token_embedding(input_ids)
    pos1 = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    pos = ops.stack([pos1, pos1, pos1], axis=0)  # (3, batch, seq)
    cos, sin = qwen3_text_cos_sin(pos, rotary_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    pad_mask = ops.cast(attention_mask, "float32")[:, :, None]
    return language_model(hidden, cos, sin, attention_mask=mask, pad_mask=pad_mask)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeModel(Qwen2VLModel):
    """Qwen3.5-MoE multimodal backbone: ViT + Qwen3-Next MoE decoder (no LM head).

    Reuses :class:`Qwen2VLModel`'s multimodal fusion (image/video placeholder scatter,
    M-RoPE position indexing via ``get_rope_index``, causal masking) but with a
    Qwen3.5-MoE vision tower (no DeepStack) and a Qwen3-Next **MoE hybrid** text
    decoder (Gated-DeltaNet + gated full attention + routed/shared experts), using
    **interleaved** partial-rotary M-RoPE. Returns raw features; use
    :class:`Qwen3_5MoeConditionalGenerate` for logits / text.

    Output dict:

    .. code-block:: python

        out = model({
            "input_ids": ...,            # (B, L) int, image/video placeholders
            "pixel_values": ...,         # (num_patches, patch_dim) image patches
            "image_grid_thw": ...,       # (num_images, 3) per-image (t, h, w)
            "pixel_values_videos": ...,  # (num_patches, patch_dim) video patches
            "video_grid_thw": ...,       # (num_videos, 3) per-video (t, h, w)
        })
        out["last_hidden_state"]   # (B, L, embed_dim)

    The vision keys are optional: pass images, video, both, or neither (text-only).

    Reference:
        - `Qwen3 Technical Report <https://arxiv.org/abs/2505.09388>`_
    """

    HF_MODEL_TYPE = ("qwen3_5_moe", "qwen3_5_moe_text")
    default_load_dtype = "bfloat16"
    config_class = Qwen3_5MoeConfig

    def __init__(
        self,
        vocab_size=248320,
        embed_dim=2048,
        mlp_dim=512,
        num_layers=40,
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        tie_embeddings=False,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_experts=256,
        num_experts_per_tok=8,
        moe_mlp_dim=512,
        shared_mlp_dim=512,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        vision_depth=27,
        vision_embed_dim=1152,
        vision_mlp_dim=4304,
        vision_num_heads=16,
        vision_out_dim=None,
        vision_act="gelu_pytorch_tanh",
        num_position_embeddings=2304,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN3_5_MOE_TOKENS["image_token_id"],
        video_token_id=QWEN3_5_MOE_TOKENS["video_token_id"],
        vision_start_token_id=QWEN3_5_MOE_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN3_5_MOE_TOKENS["vision_end_token_id"],
        **kwargs,
    ):
        nm = kwargs.pop("name", None)
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        vision_out_dim = vision_out_dim or embed_dim
        rotary_dim = int(head_dim * partial_rotary_factor)
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        visual = Qwen3_5MoeVisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            intermediate_size=vision_mlp_dim,
            out_hidden_size=vision_out_dim,
            num_position_embeddings=num_position_embeddings,
            hidden_act=vision_act,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            name="visual",
        )
        language_model = Qwen3_5MoeTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            norm_eps=norm_eps,
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_mlp_dim=moe_mlp_dim,
            shared_mlp_dim=shared_mlp_dim,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            name="language_model",
        )
        causal_mask = CausalMask(name="causal_mask")
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        video_merge = MediaMerge(video_token_id, embed_dim, name="video_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "position_ids": layers.Input(
                shape=(3, None), dtype="int32", name="position_ids"
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
        hidden = qwen3_5_moe_multimodal_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["position_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs["pixel_values_videos"],
            inputs["video_grid_thw"],
            token_embedding=language_model.token_embedding,
            visual=visual,
            language_model=language_model,
            image_merge=image_merge,
            video_merge=video_merge,
            causal_mask=causal_mask,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            mrope_section=tuple(mrope_section),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(language_model.token_embedding, name="lm_head")(hidden)
            )

        # Skip Qwen2VLModel.__init__ (it builds the 2-VL graph); go straight to the
        # functional keras init after Qwen2VLModel in the MRO.
        super(Qwen2VLModel, self).__init__(
            inputs=inputs, outputs=outputs, name=nm or type(self).__name__, **kwargs
        )

        self.visual = visual
        self.language_model = language_model
        self.causal_mask_layer = causal_mask
        self.image_merge = image_merge
        self.video_merge = video_merge
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.mrope_section = tuple(mrope_section)
        self.tie_embeddings = tie_embeddings
        self.full_attention_interval = full_attention_interval
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = tuple(mlp_only_layers)
        self.vision_depth = vision_depth
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_heads = vision_num_heads
        self.vision_out_dim = vision_out_dim
        self.vision_act = vision_act
        self.num_position_embeddings = num_position_embeddings
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.rotary_dim = rotary_dim
        self.patch_dim = patch_dim
        self.tokens_per_second = 1
        self.build_vision = True

        with inference_scope():
            self.materialize_build()

    def get_image_features(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw)

    def _merged_cos_sin(self, position_ids):
        return qwen3_text_cos_sin(
            position_ids, self.rotary_dim, self.rope_theta, self.mrope_section
        )

    @staticmethod
    def _pad_mask(attention_mask):
        if attention_mask is None:
            return None
        return ops.cast(ops.convert_to_tensor(attention_mask), "float32")[:, :, None]

    @classmethod
    def config_from_hf(cls, hf_config):
        tc = hf_config.get("text_config", hf_config)
        vc = hf_config.get("vision_config", {})
        rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        mrope = rope.get("mrope_section") or [11, 11, 10]
        hidden = tc["hidden_size"]
        return {
            "vocab_size": tc["vocab_size"],
            "embed_dim": hidden,
            "mlp_dim": tc.get(
                "intermediate_size", tc.get("moe_intermediate_size", 512)
            ),
            "num_layers": tc["num_hidden_layers"],
            "num_heads": tc["num_attention_heads"],
            "num_kv_heads": tc["num_key_value_heads"],
            "head_dim": tc.get("head_dim", 256),
            "norm_eps": tc.get("rms_norm_eps", 1e-6),
            "rope_theta": rope.get("rope_theta", tc.get("rope_theta", 10000000.0)),
            "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
            "mrope_section": tuple(mrope),
            "tie_embeddings": bool(
                hf_config.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
                or False
            ),
            "full_attention_interval": tc.get("full_attention_interval", 4),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "num_experts": tc.get("num_experts", 256),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "moe_mlp_dim": tc.get("moe_intermediate_size", 512),
            "shared_mlp_dim": tc.get("shared_expert_intermediate_size", 512),
            "norm_topk_prob": bool(tc.get("norm_topk_prob", True)),
            "decoder_sparse_step": tc.get("decoder_sparse_step", 1),
            "mlp_only_layers": tuple(tc.get("mlp_only_layers") or ()),
            "vision_depth": vc.get("depth", 27),
            "vision_embed_dim": vc.get("hidden_size", 1152),
            "vision_mlp_dim": vc.get("intermediate_size", 4304),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_out_dim": vc.get("out_hidden_size", hidden),
            "vision_act": vc.get("hidden_act", "gelu_pytorch_tanh"),
            "num_position_embeddings": vc.get("num_position_embeddings", 2304),
            "patch_size": vc.get("patch_size", 16),
            "spatial_merge_size": vc.get("spatial_merge_size", 2),
            "temporal_patch_size": vc.get("temporal_patch_size", 2),
            "in_channels": vc.get("in_chans", vc.get("in_channels", 3)),
            "image_token_id": hf_config.get("image_token_id", 248056),
            "video_token_id": hf_config.get("video_token_id", 248057),
            "vision_start_token_id": hf_config.get("vision_start_token_id", 248053),
            "vision_end_token_id": hf_config.get("vision_end_token_id", 248054),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen3_5_moe_hf_to_keras import transfer_qwen3_5_moe_weights

        transfer_qwen3_5_moe_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super(Qwen2VLModel, self).get_config()
        for k in [
            "vocab_size",
            "embed_dim",
            "mlp_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "norm_eps",
            "rope_theta",
            "partial_rotary_factor",
            "mrope_section",
            "tie_embeddings",
            "full_attention_interval",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "num_experts",
            "num_experts_per_tok",
            "moe_mlp_dim",
            "shared_mlp_dim",
            "norm_topk_prob",
            "decoder_sparse_step",
            "mlp_only_layers",
            "vision_depth",
            "vision_embed_dim",
            "vision_mlp_dim",
            "vision_num_heads",
            "vision_out_dim",
            "vision_act",
            "num_position_embeddings",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            config[k] = getattr(self, k)
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeConditionalGenerate(Qwen3_5MoeModel, BaseGeneration):
    """Qwen3.5-MoE with an LM head + fast ``.generate()`` (image+text -> text).

    ``build_cache`` runs the vision encoder + interleaved-M-RoPE prefill into a
    **hybrid per-layer cache** (fixed-slot ``(key, value)`` for the full-attention
    layers, ``(conv_state, recurrent_state)`` for the Gated-DeltaNet layers), with
    ``rope_deltas`` carried alongside; ``call_with_cache`` then decodes one text token
    at M-RoPE position ``cache_idx + rope_delta``. Vision pixels are passed as for
    :class:`Qwen3_5MoeModel`.
    """

    # Qwen's <|im_end|> stop id in the Qwen3.5 vocab. Explicit generate() args (or the
    # tokenizer's eos_token_id) override this; confirm against the real tokenizer.
    eos_token_id = (247356,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(
            hidden, ops.transpose(self.language_model.token_embedding.embeddings)
        )

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
        nkv, hd = self.num_kv_heads, self.head_dim
        inputs_embeds, position_ids, rope_deltas, _ = self._prepare_inputs(
            token_ids,
            pixel_values,
            image_grid_thw,
            padding_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        prompt_len = int(token_ids.shape[1])
        cos, sin = self._merged_cos_sin(position_ids)
        causal = self._causal_mask(
            prompt_len, prompt_len, offset=0, attention_mask=padding_mask
        )
        hidden, states = self.language_model(
            inputs_embeds,
            cos,
            sin,
            attention_mask=causal,
            pad_mask=self._pad_mask(padding_mask),
            use_cache=True,
        )
        cache = []
        for i, state in enumerate(states):
            if self.language_model.layer_types[i] == "full_attention":
                k, v = state
                ck = ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
                )
                cv = ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
                )
                cache.append((ck, cv))
            else:
                cache.append(state)
        logits = self.project(hidden[:, -1, :])
        return (tuple(cache), rope_deltas), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        hybrid, rope_deltas = cache
        batch = int(token_ids.shape[0])
        pos = ops.broadcast_to(
            ops.reshape(cache_update_index + rope_deltas, (1, batch, 1)), (3, batch, 1)
        )
        cos, sin = self._merged_cos_sin(pos)
        full_idx = next(
            (
                i
                for i, t in enumerate(self.language_model.layer_types)
                if t == "full_attention"
            ),
            None,
        )
        key_mask = None
        if full_idx is not None:
            max_len = int(hybrid[full_idx][0].shape[2])
            key_mask = ops.cast(
                ops.where(ops.arange(max_len) <= cache_update_index, 0.0, MASK_NEG),
                "float32",
            )[None, None, None, :]
        h = self.language_model.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.language_model.decoder_layers):
            h, state = layer.decode_step(
                h, cos, sin, hybrid[i], cache_update_index, key_mask
            )
            new_cache.append(state)
        logits = self.project(self.language_model.final_norm(h))[:, 0, :]
        return logits, (tuple(new_cache), rope_deltas)
