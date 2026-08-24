import keras
from keras import layers, ops

from kerasformers.base import (
    BaseGeneration,
    CausalMask,
    CheckpointSource,
    MediaMerge,
    TiedHead,
)
from kerasformers.base.base_mixin import inference_scope
from kerasformers.models.qwen2_vl.qwen2_vl_model import (
    Qwen2VLModel,
    qwen2_vl_multimodal_features,
    qwen2_vl_text_features,
    vision_rotary_cos_sin,
)

from .qwen2_5_vl_config import Qwen2_5VLConfig, Qwen2_5VLTextConfig
from .qwen2_5_vl_layers import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5VLDecoderLayer,
    Qwen2_5VLPatchMerger,
    Qwen2_5VLRMSNorm,
    Qwen2_5VLVisionBlock,
)

MASK_NEG = -1e9


def get_window_index(grid_thw, window_size, spatial_merge_size, patch_size):
    """Window-partition the merged-patch sequence (mirrors HF ``get_window_index``).

    Qwen2.5-VL runs *windowed* attention on most vision blocks: patches are
    reordered so each spatial window is contiguous, attention is masked
    block-diagonally per window, then the output is scattered back. This computes
    that permutation. Per image, the merged-patch grid (``h // merge`` x
    ``w // merge`` per temporal slice) is padded up to whole
    ``vit_window`` x ``vit_window`` windows
    (``vit_window = window_size // merge // patch_size``), laid out window by
    window, and the padding (sentinel ``-100``) is dropped.

    Args:
        grid_thw: Per-image ``(t, h, w)`` patch-grid sizes (tensor / array / list).
        window_size: Attention window size, in pixels.
        spatial_merge_size: Spatial patch-merge factor.
        patch_size: Vision patch size, in pixels.

    Returns:
        ``(window_index, cu_window_seqlens)``:

        - ``window_index``: an ``int32`` permutation tensor of the
          ``seq // merge_unit`` merge-unit groups into window-contiguous order
          (apply with ``ops.take``; invert with ``ops.argsort``).
        - ``cu_window_seqlens``: a Python list of cumulative per-window sequence
          lengths in patch units (consecutive duplicates removed), used to build
          the block-diagonal window attention mask.
    """
    m = spatial_merge_size
    merge_unit = m * m
    vit_window = window_size // m // patch_size
    grid_rows = [
        tuple(int(v) for v in row)
        for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
    ]
    window_index = []
    cu_window_seqlens = [0]
    offset = 0
    for t, h, w in grid_rows:
        lh, lw = h // m, w // m
        pad_h = (vit_window - lh % vit_window) % vit_window
        pad_w = (vit_window - lw % vit_window) % vit_window
        nwh = (lh + pad_h) // vit_window
        nww = (lw + pad_w) // vit_window
        index = ops.reshape(ops.arange(t * lh * lw), (t, lh, lw))
        index = ops.pad(index, [(0, 0), (0, pad_h), (0, pad_w)], constant_values=-100)
        index = ops.reshape(index, (t, nwh, vit_window, nww, vit_window))
        index = ops.transpose(index, (0, 1, 3, 2, 4))
        index = ops.reshape(index, (t * nwh * nww, vit_window * vit_window))
        for win in ops.convert_to_numpy(index).tolist():
            vals = [v for v in win if v != -100]
            window_index.extend(x + offset for x in vals)
            cu_window_seqlens.append(cu_window_seqlens[-1] + len(vals) * merge_unit)
        offset += t * lh * lw
    cu = [cu_window_seqlens[0]]
    for v in cu_window_seqlens[1:]:
        if v != cu[-1]:
            cu.append(v)
    return ops.convert_to_tensor(window_index, dtype="int32"), cu


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5VLVisionModel(layers.Layer):
    """Qwen2.5-VL vision tower: patch-embed -> windowed blocks -> 2x2 merger.

    Differs from the Qwen2-VL tower in its blocks (RMSNorm + SwiGLU instead of
    LayerNorm + quick-gelu) and, mainly, in using **windowed** attention: ``call``
    reorders the flattened patches into window-contiguous order
    (:func:`get_window_index`), runs the blocks under a block-diagonal mask that is
    per-window on most layers and per-image (full) on ``fullatt_block_indexes``,
    merges each ``spatial_merge_size`` x ``spatial_merge_size`` patch group, then
    un-permutes (``argsort``) back to row-major order.

    Args:
        embed_dim: Vision hidden width.
        depth: Number of vision blocks.
        num_heads: Vision attention heads.
        intermediate_size: SwiGLU hidden width inside the vision blocks.
        out_hidden_size: Output width of the merger (the LLM's hidden size).
        window_size: Windowed-attention window size, in pixels.
        fullatt_block_indexes: Block indices that use full (non-windowed) attention.
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
        window_size,
        fullatt_block_indexes,
        patch_size=14,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.window_size = window_size
        self.fullatt_block_indexes = tuple(fullatt_block_indexes)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads
        self.merge_unit = spatial_merge_size * spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen2_5VLVisionBlock(
                embed_dim, num_heads, intermediate_size, name=f"blocks_{i}"
            )
            for i in range(depth)
        ]
        self.merger = Qwen2_5VLPatchMerger(
            out_hidden_size,
            embed_dim,
            spatial_merge_size,
            name="merger",
        )

    def call(self, pixel_values, grid_thw):
        grid_rows = [
            tuple(int(v) for v in row)
            for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
        ]
        u = self.merge_unit
        seq = sum(t * h * w for t, h, w in grid_rows)

        hidden = self.patch_embed(pixel_values)

        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        window_index, cu_window = get_window_index(
            grid_thw, self.window_size, self.spatial_merge_size, self.patch_size
        )

        hidden = ops.reshape(hidden, (seq // u, u, self.embed_dim))
        hidden = ops.reshape(
            ops.take(hidden, window_index, axis=0), (seq, self.embed_dim)
        )
        cos = ops.reshape(
            ops.take(
                ops.reshape(cos, (seq // u, u, self.head_dim)), window_index, axis=0
            ),
            (seq, -1),
        )
        sin = ops.reshape(
            ops.take(
                ops.reshape(sin, (seq // u, u, self.head_dim)), window_index, axis=0
            ),
            (seq, -1),
        )

        cu_full = [0]
        for t, h, w in grid_rows:
            for _ in range(t):
                cu_full.append(cu_full[-1] + h * w)
        full_mask = None
        if len(cu_full) > 2:
            seg = [0] * seq
            for i in range(len(cu_full) - 1):
                for j in range(cu_full[i], cu_full[i + 1]):
                    seg[j] = i
            seg = ops.convert_to_tensor(seg, dtype="int32")
            full_mask = ops.cast(
                ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG), "float32"
            )[None, None]

        seg = [0] * seq
        for i in range(len(cu_window) - 1):
            for j in range(cu_window[i], cu_window[i + 1]):
                seg[j] = i
        seg = ops.convert_to_tensor(seg, dtype="int32")
        window_mask = ops.cast(
            ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG), "float32"
        )[None, None]

        for i, block in enumerate(self.blocks):
            mask = full_mask if i in self.fullatt_block_indexes else window_mask
            hidden = block(hidden, cos, sin, attention_mask=mask)

        merged = self.merger(hidden)
        reverse = ops.argsort(window_index)
        return ops.take(merged, reverse, axis=0)

    def compute_output_spec(self, pixel_values, grid_thw):
        # Merged-token count is grid-dependent (dynamic); the grid-iterating call
        # runs eagerly at runtime, so give the functional builder an explicit spec.
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
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5VLTextModel(layers.Layer):
    """Qwen2.5 causal decoder: ``embed -> N x Qwen2_5VLDecoderLayer -> RMSNorm``.

    Architecturally the same as the Qwen2 text decoder (GQA with qkv bias, SwiGLU).
    The token embedding lives here (``token_embedding``) and is reused (tied) as the
    LM head by :class:`Qwen2_5VLConditionalGenerate`. ``call`` takes the pre-computed
    multimodal-fused ``inputs_embeds`` and merged M-RoPE ``cos`` / ``sin``, and
    threads an optional KV cache for incremental decoding.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (grouped-query attention).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon.

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeddings.
        cos, sin: merged M-RoPE tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to ``(batch, 1, q_len, kv_len)``,
            or ``None``.
        past_key_values: optional list of per-layer ``(key, value)`` cache entries.
        use_cache: when ``True``, also return the updated per-layer cache.

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
        head_dim=None,
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
        self.head_dim = head_dim or embed_dim // num_heads
        self.norm_eps = norm_eps
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen2_5VLDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim=self.head_dim,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen2_5VLRMSNorm(eps=norm_eps, name="final_norm")

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5VLModel(Qwen2VLModel):
    """Qwen2.5-VL multimodal backbone: windowed vision tower + Qwen2.5 decoder.

    Subclasses :class:`Qwen2VLModel`, reusing its M-RoPE multimodal fusion, 3D
    position indexing, and image/video handling, but swaps in the Qwen2.5-VL
    **windowed** vision tower (:class:`Qwen2_5VLVisionModel`) and adds its extra
    configuration (``window_size``, ``fullatt_block_indexes``, ``tokens_per_second``
    and the ``vision_*`` dims). This base model returns raw features (no LM head);
    use :class:`Qwen2_5VLConditionalGenerate` for logits / text.

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

    The vision keys are optional: pass the image pair, the video pair, both, or
    neither (text-only).

    Construction:

    >>> Qwen2_5VLModel.from_weights("qwen2.5-vl-3b-instruct")
    >>> Qwen2_5VLModel.from_weights("hf:Qwen/Qwen2.5-VL-7B-Instruct")

    Reference:
        - `Qwen2.5-VL Technical Report <https://arxiv.org/abs/2502.13923>`_

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text decoder / residual-stream width
            (``head_dim = embed_dim // num_heads``).
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer (grouped-query attention).
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        mrope_section: Per-axis (temporal, height, width) channel split of the
            merged M-RoPE; sums to ``head_dim // 2``.
        tie_embeddings: Whether :class:`Qwen2_5VLConditionalGenerate` ties the LM head to the
            token embedding instead of a separate projection.
        vision_depth: Number of vision-transformer blocks.
        vision_embed_dim: Vision hidden width.
        vision_mlp_dim: Vision SwiGLU hidden width.
        vision_num_heads: Vision attention heads.
        vision_out_dim: Output width of the vision merger; defaults to
            ``embed_dim`` (the LLM hidden size).
        window_size: Windowed-attention window size, in pixels.
        fullatt_block_indexes: Vision block indices that use full (non-windowed)
            attention instead of windowed attention.
        tokens_per_second: Video temporal-position scale used by M-RoPE.
        patch_size: Vision patch size, in pixels.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).
        temporal_patch_size: Number of frames grouped into one temporal patch.
        in_channels: Image channels (``3`` for RGB).
        image_token_id: Placeholder token id replaced by image patch embeddings.
        video_token_id: Placeholder token id replaced by video patch embeddings.
        vision_start_token_id: Token id marking the start of a vision span.
        vision_end_token_id: Token id marking the end of a vision span.
    """

    HF_MODEL_TYPE = "qwen2_5_vl"
    default_load_dtype = "bfloat16"
    config_class = Qwen2_5VLConfig

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=11008,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        mrope_section=(16, 24, 24),
        tie_embeddings=True,
        vision_depth=32,
        vision_embed_dim=1280,
        vision_mlp_dim=3420,
        vision_num_heads=16,
        vision_out_dim=None,
        window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
        tokens_per_second=2,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        build_vision=True,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = embed_dim // num_heads
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        vision_out_dim = vision_out_dim or embed_dim

        visual = (
            Qwen2_5VLVisionModel(
                embed_dim=vision_embed_dim,
                depth=vision_depth,
                num_heads=vision_num_heads,
                intermediate_size=vision_mlp_dim,
                out_hidden_size=vision_out_dim,
                window_size=window_size,
                fullatt_block_indexes=fullatt_block_indexes,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
                name="visual",
            )
            if build_vision
            else None
        )
        language_model = Qwen2_5VLTextModel(
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
            hidden = qwen2_vl_multimodal_features(
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
            )
        else:
            inputs = text_input()
            hidden = qwen2_vl_text_features(
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
        self.window_size = window_size
        self.fullatt_block_indexes = tuple(fullatt_block_indexes)
        self.tokens_per_second = tokens_per_second
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_dim = patch_dim
        self.build_vision = build_vision

        with inference_scope():
            self.materialize_build()

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config.get("vision_config", {})
        rope_scaling = hf_config.get("rope_scaling") or {}
        mrope = rope_scaling.get("mrope_section", [16, 24, 24])
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config["num_key_value_heads"],
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "mrope_section": tuple(mrope),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
            "vision_depth": vc.get("depth", 32),
            "vision_embed_dim": vc.get("hidden_size", 1280),
            "vision_mlp_dim": vc.get("intermediate_size", 3420),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_out_dim": vc.get("out_hidden_size", hf_config["hidden_size"]),
            "window_size": vc.get("window_size", 112),
            "fullatt_block_indexes": tuple(
                vc.get("fullatt_block_indexes", (7, 15, 23, 31))
            ),
            "tokens_per_second": vc.get("tokens_per_second", 2),
            "patch_size": vc.get("patch_size", 14),
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
        from .convert_qwen2_5_vl_hf_to_keras import transfer_qwen2_5_vl_weights

        transfer_qwen2_5_vl_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super(Qwen2VLModel, self).get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "mrope_section": self.mrope_section,
                "tie_embeddings": self.tie_embeddings,
                "vision_depth": self.vision_depth,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_heads": self.vision_num_heads,
                "vision_out_dim": self.vision_out_dim,
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
                "tokens_per_second": self.tokens_per_second,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5VLConditionalGenerate(Qwen2_5VLModel, BaseGeneration):
    """Qwen2.5-VL with an LM head + fast ``.generate()`` (image+text -> text).

    Same fast multimodal generation as
    :class:`~kerasformers.models.qwen2_vl.qwen2_vl_model.Qwen2VLConditionalGenerate`: ``build_cache``
    runs the vision encoder + 3-axis M-RoPE prefill into a fixed KV cache (carrying
    ``rope_deltas``), then ``call_with_cache`` does text-only decode at M-RoPE position
    ``cache_idx + rope_delta``. The Qwen2.5-VL backbone (windowed vision encoder) resolves
    through :class:`Qwen2_5VLModel`. Pass pixels exactly as for :class:`Qwen2_5VLModel`:
    ``gen.generate(input_ids, pixel_values=..., image_grid_thw=...)``.
    """

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


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5VLTextGenerate(Qwen2_5VLConditionalGenerate):
    """Text-only counterpart of :class:`Qwen2_5VLConditionalGenerate` (no vision tower).

    Built with ``build_vision=False`` so the windowed ViT is never constructed;
    ``.generate()`` takes just token ids. Loads just the text backbone of a Qwen2.5-VL
    checkpoint: the target-driven ``hf:`` transfer copies only the language-model weights,
    and a kerasformers checkpoint declaring ``Qwen2_5VLConditionalGenerate`` is read via
    :attr:`CHECKPOINT_SOURCE`.

        gen = Qwen2_5VLTextGenerate.from_weights("hf:Qwen/Qwen2.5-VL-3B-Instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    config_class = Qwen2_5VLTextConfig
    CHECKPOINT_SOURCE = CheckpointSource(
        "Qwen2_5VLConditionalGenerate",
        module="kerasformers.models.qwen2_5_vl.qwen2_5_vl_model",
        match="path",
    )

    def __init__(self, *args, **kwargs):
        kwargs["build_vision"] = False
        super().__init__(*args, **kwargs)
