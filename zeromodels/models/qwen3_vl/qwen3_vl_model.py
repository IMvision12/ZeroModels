import keras
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    CausalMask,
    CheckpointSource,
    MediaMerge,
    TiedHead,
)
from zeromodels.base.base_mixin import inference_scope
from zeromodels.models.qwen2_vl.qwen2_vl_model import (
    Qwen2VLModel,
    vision_rotary_cos_sin,
)

from .qwen3_vl_config import QWEN3_VL_TOKENS, Qwen3VLConfig, Qwen3VLTextConfig
from .qwen3_vl_layers import (
    Qwen3VLRMSNorm,
    Qwen3VLTextDecoderLayer,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)

MASK_NEG = -1e9


def qwen3_text_cos_sin(position_ids, head_dim, theta, mrope_section):
    """Interleaved M-RoPE cos/sin (Qwen3-VL).

    Builds per-axis frequencies then interleaves them channel-wise: T on
    channels ``0,3,6,...``, H on ``1,4,...`` (up to ``mrope_section[1]*3``),
    W on ``2,5,...`` (up to ``mrope_section[2]*3``), the tail staying T: rather
    than the contiguous T/H/W sections of Qwen2.x. Returns merged
    ``(batch, seq, head_dim)`` cos/sin tensors.
    """
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    half = head_dim // 2
    sel = [0] * half
    for dim, offset in ((1, 1), (2, 2)):
        for c in range(offset, min(mrope_section[dim] * 3, half), 3):
            sel[c] = dim
    sel = ops.convert_to_tensor(sel, dtype="int32")
    freqs_t = (
        ops.where(sel == 0, freqs[0], 0.0)
        + ops.where(sel == 1, freqs[1], 0.0)
        + ops.where(sel == 2, freqs[2], 0.0)
    )
    emb = ops.concatenate([freqs_t, freqs_t], axis=-1)
    return ops.cos(emb), ops.sin(emb)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLVisionModel(layers.Layer):
    """Qwen3-VL vision tower: learned pos-embeds -> GELU blocks -> merger + DeepStack.

    A ViT with **learned** position embeddings (bilinearly interpolated to each
    image's grid and added to the patch embeddings), ``depth`` full-attention GELU
    blocks with 2D rotary positions, and a final 2x2 merger. At each index in
    ``deepstack_visual_indexes`` an extra "DeepStack" merger taps the block output;
    those features are later injected into the text decoder's early layers.

    Args:
        embed_dim: Vision hidden width.
        depth: Number of vision blocks.
        num_heads: Vision attention heads.
        intermediate_size: Vision MLP hidden width.
        out_hidden_size: Output width of the mergers (the LLM's hidden size).
        num_position_embeddings: Size of the learned position-embedding grid.
        deepstack_visual_indexes: Block indices that feed a DeepStack merger.
        hidden_act: Vision MLP activation (e.g. ``"gelu_pytorch_tanh"``).
        patch_size: Vision patch size, in pixels.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).

    Call args:
        pixel_values: Flattened patches ``(num_patches, patch_dim)``.
        grid_thw: Per-image ``(t, h, w)`` patch-grid sizes.

    Returns:
        ``(merged, deepstack)``: merged image embeddings
        ``(num_merged_tokens, out_hidden_size)`` plus one DeepStack tensor of the
        same shape per entry in ``deepstack_visual_indexes``.
    """

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        intermediate_size,
        out_hidden_size,
        num_position_embeddings,
        deepstack_visual_indexes,
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
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads
        self.merge_unit = spatial_merge_size * spatial_merge_size
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
        self.deepstack_mergers = [
            Qwen3VLVisionPatchMerger(
                out_hidden_size,
                embed_dim,
                spatial_merge_size,
                use_postshuffle_norm=True,
                name=f"deepstack_merger_{i}",
            )
            for i in range(len(self.deepstack_visual_indexes))
        ]

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

        deepstack = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, attention_mask=mask)
            if i in self.deepstack_visual_indexes:
                j = self.deepstack_visual_indexes.index(i)
                deepstack.append(self.deepstack_mergers[j](hidden))
        merged = self.merger(hidden)
        return merged, deepstack

    def compute_output_spec(self, pixel_values, grid_thw):
        # Merged-token count is grid-dependent (dynamic); the grid-iterating call
        # runs eagerly at runtime. Returns (merged, [deepstack maps]).
        spec = keras.KerasTensor((None, self.out_hidden_size), dtype=self.compute_dtype)
        ds = [
            keras.KerasTensor((None, self.out_hidden_size), dtype=self.compute_dtype)
            for _ in self.deepstack_visual_indexes
        ]
        return spec, ds

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
                "deepstack_visual_indexes": self.deepstack_visual_indexes,
                "hidden_act": self.hidden_act,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLTextModel(layers.Layer):
    """Qwen3 causal decoder with DeepStack visual-feature injection.

    A Qwen3 decoder (per-head QK-norm, no qkv bias, SwiGLU) whose token embedding is
    reused (tied) as the LM head. Identical to the plain Qwen3 decoder except that,
    during prefill, the i-th DeepStack feature map (scattered to a full
    ``(batch, seq, embed_dim)`` tensor by the model) is added to the output of
    decoder layer ``i``. ``call`` takes the merged interleaved-M-RoPE tables and
    threads an optional KV cache.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (grouped-query attention).
        head_dim: Per-head dim of the attention.
        norm_eps: RMSNorm epsilon (shared by the per-head QK-norms too).

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeddings.
        cos, sin: merged interleaved-M-RoPE tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to ``(batch, 1, q_len, kv_len)``,
            or ``None``.
        past_key_values: optional list of per-layer ``(key, value)`` cache entries.
        use_cache: when ``True``, also return the updated per-layer cache.
        deepstack_full: optional list of ``(batch, seq, embed_dim)`` DeepStack maps
            added to the first ``len(deepstack_full)`` decoder layers (prefill only).

    Returns:
        ``(batch, seq, embed_dim)``, or ``(hidden, new_cache)`` when ``use_cache``.
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
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3VLTextDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3VLRMSNorm(eps=norm_eps, name="final_norm")

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        deepstack_full=None,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        n_ds = 0 if deepstack_full is None else len(deepstack_full)
        for i, layer in enumerate(self.decoder_layers):
            past = past_key_values[i] if past_key_values is not None else None
            out = layer(
                hidden,
                cos,
                sin,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden, kv = out
                new_cache.append(kv)
            else:
                hidden = out
            if i < n_ds:
                hidden = hidden + deepstack_full[i]
        hidden = self.final_norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def compute_output_spec(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        deepstack_full=None,
    ):
        return keras.KerasTensor(inputs_embeds.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


def qwen3_vl_multimodal_features(
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
    head_dim,
    rope_theta,
    mrope_section,
    image_token_id,
    video_token_id,
    n_deepstack,
):
    # KerasHub-style always-media multimodal graph with DeepStack. Vision runs
    # unconditionally (no-op merge when a stream's token is absent); each DeepStack
    # map is scattered into a zeros tensor at the image/video slots (reusing the
    # weightless MediaMerge) and summed, matching the imperative _deepstack_full.
    media = ops.logical_or(
        ops.equal(input_ids, image_token_id), ops.equal(input_ids, video_token_id)
    )
    hidden = token_embedding(ops.where(media, 0, input_ids))
    image_embeds, image_ds = visual(pixel_values, image_grid_thw)
    hidden = image_merge(hidden, input_ids, image_embeds)
    video_embeds, video_ds = visual(pixel_values_videos, video_grid_thw)
    hidden = video_merge(hidden, input_ids, video_embeds)
    zeros = ops.zeros_like(hidden)
    deepstack_full = [
        image_merge(zeros, input_ids, image_ds[i])
        + video_merge(zeros, input_ids, video_ds[i])
        for i in range(n_deepstack)
    ]
    pos = ops.transpose(position_ids, (1, 0, 2))  # (batch, 3, seq) -> (3, batch, seq)
    cos, sin = qwen3_text_cos_sin(pos, head_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(
        hidden, cos, sin, attention_mask=mask, deepstack_full=deepstack_full
    )


def qwen3_vl_text_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    language_model,
    causal_mask,
    head_dim,
    rope_theta,
    mrope_section,
):
    hidden = token_embedding(input_ids)
    pos1 = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    pos = ops.stack([pos1, pos1, pos1], axis=0)  # (3, batch, seq)
    cos, sin = qwen3_text_cos_sin(pos, head_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(hidden, cos, sin, attention_mask=mask)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLModel(Qwen2VLModel):
    """Qwen3-VL multimodal backbone: vision tower + Qwen3 decoder + DeepStack.

    Subclasses :class:`Qwen2VLModel`, reusing its multimodal fusion and M-RoPE
    indexing, but uses a Qwen3 decoder (per-head QK-norm, no qkv bias),
    **interleaved** M-RoPE (:func:`qwen3_text_cos_sin`), a vision tower with learned
    (interpolated) position embeddings and GELU blocks, and **DeepStack**: features
    from several vision layers are scattered into the text decoder's early layers
    during prefill. This base model returns raw features (no LM head); use
    :class:`Qwen3VLConditionalGenerate` for logits / text.

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

    Construction:

    >>> Qwen3VLModel.from_weights("zeromodels/qwen3-vl-2b-instruct")
    >>> Qwen3VLModel.from_weights("hf:Qwen/Qwen3-VL-4B-Instruct")

    Reference:
        - `Qwen3 Technical Report <https://arxiv.org/abs/2505.09388>`_

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text decoder / residual-stream width.
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer (grouped-query attention).
        head_dim: Per-head dim of the text attention.
        norm_eps: RMSNorm epsilon (shared by the per-head QK-norms too).
        rope_theta: Rotary base frequency.
        mrope_section: Per-axis (temporal, height, width) channel split of the
            interleaved M-RoPE.
        tie_embeddings: Whether :class:`Qwen3VLConditionalGenerate` ties the LM head to the
            token embedding instead of a separate projection.
        vision_depth: Number of vision-transformer blocks.
        vision_embed_dim: Vision hidden width.
        vision_mlp_dim: Vision MLP hidden width.
        vision_num_heads: Vision attention heads.
        vision_out_dim: Output width of the vision merger; defaults to
            ``embed_dim`` (the LLM hidden size).
        vision_act: Vision MLP activation (e.g. ``"gelu_pytorch_tanh"``).
        num_position_embeddings: Size of the learned vision position-embedding grid
            (bilinearly interpolated per image).
        deepstack_visual_indexes: Vision block indices whose features are injected
            into the text decoder's early layers (DeepStack).
        patch_size: Vision patch size, in pixels.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).
        temporal_patch_size: Number of frames grouped into one temporal patch.
        in_channels: Image channels (``3`` for RGB).
        image_token_id: Placeholder token id replaced by image patch embeddings.
        video_token_id: Placeholder token id replaced by video patch embeddings.
        vision_start_token_id: Token id marking the start of a vision span.
        vision_end_token_id: Token id marking the end of a vision span.
    """

    HF_MODEL_TYPE = "qwen3_vl"
    default_load_dtype = "bfloat16"
    config_class = Qwen3VLConfig

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=6144,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=5000000.0,
        mrope_section=(24, 20, 20),
        tie_embeddings=True,
        vision_depth=24,
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_heads=16,
        vision_out_dim=None,
        vision_act="gelu_pytorch_tanh",
        num_position_embeddings=2304,
        deepstack_visual_indexes=(5, 11, 17),
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN3_VL_TOKENS["image_token_id"],
        video_token_id=QWEN3_VL_TOKENS["video_token_id"],
        vision_start_token_id=QWEN3_VL_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN3_VL_TOKENS["vision_end_token_id"],
        build_vision=True,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        vision_out_dim = vision_out_dim or embed_dim
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        n_deepstack = len(deepstack_visual_indexes)

        visual = (
            Qwen3VLVisionModel(
                embed_dim=vision_embed_dim,
                depth=vision_depth,
                num_heads=vision_num_heads,
                intermediate_size=vision_mlp_dim,
                out_hidden_size=vision_out_dim,
                num_position_embeddings=num_position_embeddings,
                deepstack_visual_indexes=deepstack_visual_indexes,
                hidden_act=vision_act,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
                name="visual",
            )
            if build_vision
            else None
        )
        language_model = Qwen3VLTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            name="language_model",
        )
        causal_mask = CausalMask(name="causal_mask")
        image_merge = video_merge = None
        if build_vision:
            image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
            video_merge = MediaMerge(video_token_id, embed_dim, name="video_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        def text_input():
            return {
                "input_ids": layers.Input(
                    shape=(None,), dtype="int32", name="input_ids"
                ),
                "attention_mask": layers.Input(
                    shape=(None,), dtype="int32", name="attention_mask"
                ),
            }

        if build_vision:
            inputs = text_input()
            inputs["position_ids"] = layers.Input(
                shape=(3, None), dtype="int32", name="position_ids"
            )
            inputs["pixel_values"] = layers.Input(
                shape=(patch_dim,), dtype="float32", name="pixel_values"
            )
            inputs["image_grid_thw"] = layers.Input(
                shape=(3,), dtype="int32", name="image_grid_thw"
            )
            inputs["pixel_values_videos"] = layers.Input(
                shape=(patch_dim,), dtype="float32", name="pixel_values_videos"
            )
            inputs["video_grid_thw"] = layers.Input(
                shape=(3,), dtype="int32", name="video_grid_thw"
            )
            hidden = qwen3_vl_multimodal_features(
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
                head_dim=head_dim,
                rope_theta=rope_theta,
                mrope_section=tuple(mrope_section),
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                n_deepstack=n_deepstack,
            )
        else:
            inputs = text_input()
            hidden = qwen3_vl_text_features(
                inputs["input_ids"],
                inputs["attention_mask"],
                token_embedding=language_model.token_embedding,
                language_model=language_model,
                causal_mask=causal_mask,
                head_dim=head_dim,
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

        # Skip Qwen2VLModel.__init__ (it builds the 2-VL graph); go straight to the
        # functional keras init after Qwen2VLModel in the MRO (BaseModel).
        super(Qwen2VLModel, self).__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
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
        self.mrope_section = tuple(mrope_section)
        self.tie_embeddings = tie_embeddings
        self.vision_depth = vision_depth
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_heads = vision_num_heads
        self.vision_out_dim = vision_out_dim
        self.vision_act = vision_act
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_dim = patch_dim
        self.tokens_per_second = 1
        self.build_vision = build_vision

        with inference_scope():
            self.materialize_build()

    def _merged_cos_sin(self, position_ids):
        return qwen3_text_cos_sin(
            position_ids, self.head_dim, self.rope_theta, self.mrope_section
        )

    def _prepare_inputs(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        attention_mask,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.language_model.token_embedding(input_ids)
        rope_deltas = ops.zeros((batch,), dtype="int32")
        extra = {}
        has_image = pixel_values is not None and image_grid_thw is not None
        has_video = pixel_values_videos is not None and video_grid_thw is not None
        image_grid = video_grid = None
        if has_image or has_video:
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
            deepstack_full = None
            if has_image:
                image_grid = ops.cast(ops.convert_to_tensor(image_grid_thw), "int32")
                image_embeds, ds = self.visual(pixel_values, image_grid)
                idx_t = ops.reshape(
                    ops.convert_to_tensor(
                        [j for j, v in enumerate(ids_flat) if v == self.image_token_id],
                        dtype="int32",
                    ),
                    (-1, 1),
                )
                flat = ops.scatter_update(
                    flat, idx_t, ops.cast(image_embeds, flat.dtype)
                )
                deepstack_full = self._deepstack_full(ds, idx_t, batch, seq)
            if has_video:
                video_grid = ops.cast(ops.convert_to_tensor(video_grid_thw), "int32")
                video_embeds, vds = self.visual(pixel_values_videos, video_grid)
                vidx_t = ops.reshape(
                    ops.convert_to_tensor(
                        [j for j, v in enumerate(ids_flat) if v == self.video_token_id],
                        dtype="int32",
                    ),
                    (-1, 1),
                )
                flat = ops.scatter_update(
                    flat, vidx_t, ops.cast(video_embeds, flat.dtype)
                )
                vds_full = self._deepstack_full(vds, vidx_t, batch, seq)
                deepstack_full = (
                    vds_full
                    if deepstack_full is None
                    else [a + b for a, b in zip(deepstack_full, vds_full)]
                )
            inputs_embeds = ops.reshape(flat, (batch, seq, self.embed_dim))
            if deepstack_full is not None:
                extra = {"deepstack_full": deepstack_full}
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid, video_grid, attention_mask=attention_mask
            )
        else:
            pos = ops.broadcast_to(ops.arange(seq), (batch, seq))
            position_ids = ops.broadcast_to(pos, (3, batch, seq))
        return inputs_embeds, position_ids, rope_deltas, extra

    def _deepstack_full(self, deepstack, idx_t, batch, seq):
        out = []
        for emb in deepstack:
            z = ops.zeros((batch * seq, self.embed_dim), dtype=emb.dtype)
            z = ops.scatter_update(z, idx_t, ops.cast(emb, z.dtype))
            out.append(ops.reshape(z, (batch, seq, self.embed_dim)))
        return out

    @classmethod
    def config_from_hf(cls, hf_config):
        tc = hf_config.get("text_config", hf_config)
        vc = hf_config.get("vision_config", {})
        rope_scaling = tc.get("rope_scaling") or hf_config.get("rope_scaling") or {}
        mrope = rope_scaling.get("mrope_section", [24, 20, 20])
        hidden = tc["hidden_size"]
        heads = tc["num_attention_heads"]
        return {
            "vocab_size": tc["vocab_size"],
            "embed_dim": hidden,
            "mlp_dim": tc["intermediate_size"],
            "num_layers": tc["num_hidden_layers"],
            "num_heads": heads,
            "num_kv_heads": tc["num_key_value_heads"],
            "head_dim": tc.get("head_dim", hidden // heads),
            "norm_eps": tc.get("rms_norm_eps", 1e-6),
            "rope_theta": tc.get("rope_theta", 5000000.0),
            "mrope_section": tuple(mrope),
            "tie_embeddings": hf_config.get(
                "tie_word_embeddings", tc.get("tie_word_embeddings", False)
            ),
            "vision_depth": vc.get("depth", 24),
            "vision_embed_dim": vc.get("hidden_size", 1024),
            "vision_mlp_dim": vc.get("intermediate_size", 4096),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_out_dim": vc.get("out_hidden_size", hidden),
            "vision_act": vc.get("hidden_act", "gelu_pytorch_tanh"),
            "num_position_embeddings": vc.get("num_position_embeddings", 2304),
            "deepstack_visual_indexes": tuple(
                vc.get("deepstack_visual_indexes", (5, 11, 17))
            ),
            "patch_size": vc.get("patch_size", 16),
            "spatial_merge_size": vc.get("spatial_merge_size", 2),
            "temporal_patch_size": vc.get("temporal_patch_size", 2),
            "in_channels": vc.get("in_chans", vc.get("in_channels", 3)),
            "image_token_id": hf_config.get("image_token_id", 151655),
            "video_token_id": hf_config.get("video_token_id", 151656),
            "vision_start_token_id": hf_config.get("vision_start_token_id", 151652),
            "vision_end_token_id": hf_config.get("vision_end_token_id", 151653),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen3_vl_hf_to_keras import transfer_qwen3_vl_weights

        transfer_qwen3_vl_weights(keras_model, hf_state_dict)

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
            "mrope_section",
            "tie_embeddings",
            "vision_depth",
            "vision_embed_dim",
            "vision_mlp_dim",
            "vision_num_heads",
            "vision_out_dim",
            "vision_act",
            "num_position_embeddings",
            "deepstack_visual_indexes",
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
class Qwen3VLConditionalGenerate(Qwen3VLModel, BaseGeneration):
    """Qwen3-VL with an LM head + fast ``.generate()`` (image+text -> text).

    Same fast multimodal generation as
    :class:`~zeromodels.models.qwen2_vl.qwen2_vl_model.Qwen2VLConditionalGenerate`: ``build_cache``
    runs the vision encoder + M-RoPE prefill into a fixed KV cache (DeepStack vision
    features threaded through the prefill via ``extra``; ``rope_deltas`` carried in the
    cache), then ``call_with_cache`` does text-only decode at M-RoPE position
    ``cache_idx + rope_delta``. The Qwen3-VL backbone (vision encoder, interleaved
    M-RoPE, ``lm_head``) resolves through :class:`Qwen3VLModel`. Image / video pixels
    are passed as for that class.
    """

    # Qwen's <|im_end|> stop id. Explicit generate() args override this.
    eos_token_id = (151645,)
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
        prompt_len = int(token_ids.shape[1])
        nkv = self.language_model.num_kv_heads
        hd = self.language_model.head_dim
        inputs_embeds, position_ids, rope_deltas, extra = self._prepare_inputs(
            token_ids,
            pixel_values,
            image_grid_thw,
            padding_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        cos, sin = self._merged_cos_sin(position_ids)
        causal = self._causal_mask(
            prompt_len, prompt_len, offset=0, attention_mask=padding_mask
        )
        hidden, kv = self.language_model(
            inputs_embeds, cos, sin, attention_mask=causal, use_cache=True, **extra
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


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLTextGenerate(Qwen3VLConditionalGenerate):
    """Text-only counterpart of :class:`Qwen3VLConditionalGenerate` (no vision tower).

    Built with ``build_vision=False`` so the DeepStack ViT is never constructed;
    ``.generate()`` takes just token ids. Loads just the text backbone of a Qwen3-VL
    checkpoint: the target-driven ``hf:`` transfer copies only the language-model weights,
    and a zeromodels checkpoint declaring ``Qwen3VLConditionalGenerate`` is read via
    :attr:`CHECKPOINT_SOURCE`.

        gen = Qwen3VLTextGenerate.from_weights("hf:Qwen/Qwen3-VL-...")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    config_class = Qwen3VLTextConfig
    CHECKPOINT_SOURCE = CheckpointSource(
        "Qwen3VLConditionalGenerate",
        module="zeromodels.models.qwen3_vl.qwen3_vl_model",
        match="path",
    )

    def __init__(self, *args, **kwargs):
        kwargs["build_vision"] = False
        super().__init__(*args, **kwargs)
