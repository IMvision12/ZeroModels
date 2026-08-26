import itertools

import keras
from keras import layers, ops

from kerasformers.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    CheckpointSource,
    MediaMerge,
    TiedHead,
)
from kerasformers.base.base_mixin import inference_scope

from .qwen2_vl_config import Qwen2VLConfig, Qwen2VLTextConfig
from .qwen2_vl_layers import (
    Qwen2VLDecoderLayer,
    Qwen2VLPatchEmbed,
    Qwen2VLPatchMerger,
    Qwen2VLRMSNorm,
    Qwen2VLVisionBlock,
)

MASK_NEG = -1e9


def vision_rotary_cos_sin(grid_thw, head_dim, spatial_merge_size, theta=10000.0):
    """2D vision rotary tables for the flattened patch sequence.

    Replicates HF ``Qwen2VisionTransformer.rot_pos_emb``: per image, height and
    width position ids are laid out in ``spatial_merge_size`` block order, the
    rotary table is gathered per position, and ``emb = cat(freqs, freqs)``.

    Args:
        grid_thw: iterable of ``(t, h, w)`` patch-grid sizes per image.
        head_dim: vision attention head dim (rotary covers all of it).
        spatial_merge_size: patch merge factor (block ordering).
        theta: rotary base.

    Returns:
        ``(cos, sin)`` tensors, each ``(total_patches, head_dim)``.
    """
    m = spatial_merge_size
    rotary_dim = head_dim // 2
    grid_rows = [
        tuple(int(v) for v in row)
        for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
    ]
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    pieces = []
    for t, h, w in grid_rows:
        hpos = ops.broadcast_to(ops.arange(h)[:, None], (h, w))
        hpos = ops.reshape(
            ops.transpose(ops.reshape(hpos, (h // m, m, w // m, m)), (0, 2, 1, 3)),
            (-1,),
        )
        wpos = ops.broadcast_to(ops.arange(w)[None, :], (h, w))
        wpos = ops.reshape(
            ops.transpose(ops.reshape(wpos, (h // m, m, w // m, m)), (0, 2, 1, 3)),
            (-1,),
        )
        pieces.append(ops.tile(ops.stack([hpos, wpos], axis=-1), [t, 1]))
    pos_ids = ops.concatenate(pieces, axis=0)
    total = sum(t * h * w for t, h, w in grid_rows)
    max_grid = max(max(h, w) for _, h, w in grid_rows)
    freqs = ops.arange(max_grid, dtype="float32")[:, None] * inv_freq
    rotary = ops.reshape(ops.take(freqs, pos_ids, axis=0), (total, -1))
    emb = ops.concatenate([rotary, rotary], axis=-1)
    return ops.cos(emb), ops.sin(emb)


def vision_block_mask(grid_thw):
    """Additive block-diagonal mask so patches only attend within their image.

    Returns ``None`` when there's a single block (full attention), else a
    ``(1, 1, total, total)`` float mask (0 within an image, ``MASK_NEG`` across).
    """
    grid_rows = [
        tuple(int(v) for v in row)
        for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
    ]
    seqlens = [t * h * w for t, h, w in grid_rows]
    if len(seqlens) <= 1:
        return None
    seg = ops.concatenate(
        [ops.full((n,), i, dtype="int32") for i, n in enumerate(seqlens)], axis=0
    )
    mask = ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG)
    return ops.cast(mask, "float32")[None, None]


def text_rope_cos_sin(position_ids, head_dim, theta):
    """Per-axis rotary tables from 3D position ids.

    Args:
        position_ids: ``(3, batch, seq)`` int tensor (temporal/height/width).
        head_dim: text attention head dim.
        theta: rotary base.

    Returns:
        ``(cos, sin)`` tensors, each ``(3, batch, seq, head_dim)``.
    """
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cos(emb), ops.sin(emb)


def merge_mrope(table, mrope_section):
    """Collapse the 3 position axes into one per the M-RoPE channel sections.

    ``table`` is ``(3, batch, seq, head_dim)``; for the i-th channel chunk we
    keep the ``i % 3`` position axis. Returns ``(batch, seq, head_dim)``.
    """
    sections = list(mrope_section) * 2
    pts = list(itertools.accumulate(sections))[:-1]
    splits = ops.split(table, pts, axis=-1)
    return ops.concatenate([splits[i][i % 3] for i in range(len(splits))], axis=-1)


def qwen2_vl_merged_cos_sin(position_ids, head_dim, rope_theta, mrope_section):
    # position_ids: (3, batch, seq) -> merged M-RoPE (batch, seq, head_dim).
    cos3, sin3 = text_rope_cos_sin(position_ids, head_dim, rope_theta)
    return merge_mrope(cos3, mrope_section), merge_mrope(sin3, mrope_section)


def qwen2_vl_multimodal_features(
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
):
    # Media placeholder ids are out of / reserved in the embedding range, so zero
    # them before the lookup and overwrite the slots after (a no-op merge when a
    # stream's token is absent). Vision runs unconditionally (KerasHub-style,
    # media always an input); position_ids (3-axis M-RoPE) is precomputed by the
    # host-side get_rope_index and passed in as (batch, 3, seq).
    media = ops.logical_or(
        ops.equal(input_ids, image_token_id), ops.equal(input_ids, video_token_id)
    )
    hidden = token_embedding(ops.where(media, 0, input_ids))
    hidden = image_merge(hidden, input_ids, visual(pixel_values, image_grid_thw))
    hidden = video_merge(hidden, input_ids, visual(pixel_values_videos, video_grid_thw))
    pos = ops.transpose(position_ids, (1, 0, 2))  # (batch, 3, seq) -> (3, batch, seq)
    cos, sin = qwen2_vl_merged_cos_sin(pos, head_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(hidden, cos, sin, attention_mask=mask)


def qwen2_vl_text_features(
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
    # Text-only head (no vision tower): the three M-RoPE axes collapse to plain
    # sequential positions.
    hidden = token_embedding(input_ids)
    pos1 = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    pos = ops.stack([pos1, pos1, pos1], axis=0)  # (3, batch, seq)
    cos, sin = qwen2_vl_merged_cos_sin(pos, head_dim, rope_theta, mrope_section)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(hidden, cos, sin, attention_mask=mask)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLVisionModel(layers.Layer):
    """Qwen2-VL vision tower: patch-embed -> rotary blocks -> 2x2 merger.

    A ViT over the flattened patch sequence: a Conv3d-as-Dense patch embed, ``depth``
    full-attention blocks with 2D rotary positions and a per-image block-diagonal
    mask, then a 2x2 patch merger projecting to the LLM hidden width. All images /
    video frames are processed in one packed sequence.

    Args:
        embed_dim: Vision hidden width.
        depth: Number of vision blocks.
        num_heads: Vision attention heads.
        llm_hidden_size: Output width of the merger (the LLM's hidden size).
        mlp_ratio: MLP expansion ratio inside the vision blocks.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).

    Call args:
        pixel_values: Flattened patches ``(num_patches, patch_dim)``.
        grid_thw: Per-image ``(t, h, w)`` patch-grid sizes.

    Returns:
        Merged image embeddings ``(num_merged_tokens, llm_hidden_size)``.
    """

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        llm_hidden_size,
        mlp_ratio=4,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.llm_hidden_size = llm_hidden_size
        self.mlp_ratio = mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads

        self.patch_embed = Qwen2VLPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen2VLVisionBlock(embed_dim, num_heads, mlp_ratio, name=f"blocks_{i}")
            for i in range(depth)
        ]
        self.merger = Qwen2VLPatchMerger(
            llm_hidden_size, embed_dim, spatial_merge_size, name="merger"
        )

    def call(self, pixel_values, grid_thw):
        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        mask = vision_block_mask(grid_thw)
        hidden = self.patch_embed(pixel_values)
        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_mask=mask)
        return self.merger(hidden)

    def compute_output_spec(self, pixel_values, grid_thw):
        # The merged-token count is grid-dependent (dynamic); the grid-iterating
        # call runs eagerly at runtime, so give the functional builder an explicit
        # spec instead of tracing the host-side (numpy) grid ops.
        return keras.KerasTensor((None, self.llm_hidden_size), dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "llm_hidden_size": self.llm_hidden_size,
                "mlp_ratio": self.mlp_ratio,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLTextModel(layers.Layer):
    """Qwen2 causal decoder: ``embed -> num_layers x Qwen2VLDecoderLayer -> RMSNorm``.

    The token embedding lives here (``token_embedding``) and is reused (tied) as the
    LM head by :class:`Qwen2VLModel`. A standard Qwen2 stack (GQA with qkv bias,
    SwiGLU); ``call`` takes the pre-computed multimodal-fused ``inputs_embeds`` and
    merged M-RoPE tables, and threads an optional KV cache for incremental decoding.

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
            Qwen2VLDecoderLayer(
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
        self.final_norm = Qwen2VLRMSNorm(eps=norm_eps, name="final_norm")

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
        # Decoder stack keeps the (batch, seq, embed_dim) shape; an explicit spec
        # keeps the functional builder from tracing the dynamic-shape reshapes in
        # attention (they resolve only at run time).
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
class Qwen2VLModel(BaseModel):
    """Qwen2-VL multimodal backbone: vision tower + Qwen2 decoder fused by M-RoPE.

    A ViT-style vision tower (Conv3d-as-Dense patch embed -> full-attention rotary
    blocks -> 2x2 merger) whose merged image/video embeddings are scattered into the
    ``image_token_id`` / ``video_token_id`` placeholder slots of a Qwen2 decoder,
    with 3D M-RoPE positions from :meth:`get_rope_index`. The forward pass runs
    eagerly with ``keras.ops``. This base model returns raw features (no LM head);
    use :class:`Qwen2VLConditionalGenerate` for logits / text.

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

    >>> Qwen2VLModel.from_weights("qwen2-vl-2b-instruct")
    >>> Qwen2VLModel.from_weights("hf:Qwen/Qwen2-VL-7B-Instruct")

    Reference:
        - `Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at
          Any Resolution <https://arxiv.org/abs/2409.12191>`_

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
        tie_embeddings: Whether :class:`Qwen2VLConditionalGenerate` ties the LM head to the
            token embedding instead of a separate projection.
        vision_depth: Number of vision-transformer blocks.
        vision_embed_dim: Vision hidden width.
        vision_num_heads: Vision attention heads.
        vision_mlp_ratio: MLP expansion ratio in the vision blocks.
        patch_size: Vision patch size, in pixels.
        spatial_merge_size: Spatial patch-merge factor (e.g. ``2`` -> 2x2 groups).
        temporal_patch_size: Number of frames grouped into one temporal patch.
        in_channels: Image channels (``3`` for RGB).
        image_token_id: Placeholder token id replaced by image patch embeddings.
        video_token_id: Placeholder token id replaced by video patch embeddings.
        vision_start_token_id: Token id marking the start of a vision span.
        vision_end_token_id: Token id marking the end of a vision span.
    """

    HF_MODEL_TYPE = "qwen2_vl"
    default_load_dtype = "bfloat16"
    config_class = Qwen2VLConfig
    # Qwen2VLConditionalGenerate / Qwen2VLTextGenerate flip this on to also emit
    # LM-head logits from the graph.
    output_logits = False

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=1536,
        mlp_dim=8960,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        mrope_section=(16, 24, 24),
        tie_embeddings=True,
        vision_depth=32,
        vision_embed_dim=1280,
        vision_num_heads=16,
        vision_mlp_ratio=4,
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

        visual = (
            Qwen2VLVisionModel(
                embed_dim=vision_embed_dim,
                depth=vision_depth,
                num_heads=vision_num_heads,
                llm_hidden_size=embed_dim,
                mlp_ratio=vision_mlp_ratio,
                spatial_merge_size=spatial_merge_size,
                name="visual",
            )
            if build_vision
            else None
        )
        language_model = Qwen2VLTextModel(
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

        super().__init__(
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
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
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

        # The vision tower + decoder sublayers don't all materialize during the
        # symbolic build (compute_output_spec skips their grid/dynamic call), so a
        # concrete dummy forward creates every weight before from_weights loads.
        with inference_scope():
            self.materialize_build()

    def materialize_build(self):
        if self.build_vision:
            m = self.spatial_merge_size
            h = w = 2 * m
            n = (h * w) // (m * m)
            input_ids = ops.concatenate(
                [
                    ops.zeros((1, 1), dtype="int32"),
                    ops.full((1, n), self.image_token_id, dtype="int32"),
                    ops.ones((1, 1), dtype="int32"),
                ],
                axis=1,
            )
            seq = n + 2
            self(
                {
                    "input_ids": input_ids,
                    "attention_mask": ops.ones((1, seq), dtype="int32"),
                    "position_ids": ops.zeros((1, 3, seq), dtype="int32"),
                    "pixel_values": ops.zeros((h * w, self.patch_dim), dtype="float32"),
                    "image_grid_thw": ops.convert_to_tensor([[1, h, w]], dtype="int32"),
                    "pixel_values_videos": ops.zeros(
                        (m * m, self.patch_dim), dtype="float32"
                    ),
                    "video_grid_thw": ops.convert_to_tensor([[1, m, m]], dtype="int32"),
                }
            )
        else:
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def get_rope_index(
        self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None
    ):
        m = self.spatial_merge_size
        ids_host = ops.convert_to_numpy(ops.convert_to_tensor(input_ids)).tolist()
        batch, seq = len(ids_host), len(ids_host[0])

        def _rows(grid):
            return [
                tuple(int(v) for v in row)
                for row in ops.convert_to_numpy(ops.convert_to_tensor(grid))
            ]

        grid_iters = {
            1: iter(_rows(image_grid_thw)) if image_grid_thw is not None else None,
            2: iter(_rows(video_grid_thw)) if video_grid_thw is not None else None,
        }
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
            ttype = [
                1
                if v == self.image_token_id
                else (2 if v == self.video_token_id else 0)
                for v in ids_kept
            ]
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
                    t, h, w = (int(v) for v in next(grid_iters[key]))
                    lt, lh, lw = t, h // m, w // m
                    n = lt * lh * lw
                    wpos = ops.tile(ops.arange(cur, cur + lw), [lh * lt])
                    hpos = ops.repeat(ops.arange(cur, cur + lh), lw * lt)
                    tpos = ops.full((n,), cur * self.tokens_per_second, dtype="int32")
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

    def embed_tokens(self, input_ids):
        return self.language_model.token_embedding(input_ids)

    def get_image_features(self, pixel_values, image_grid_thw):
        return self.visual(pixel_values, image_grid_thw)

    def _causal_mask(self, q_len, kv_len, offset, attention_mask=None):
        qi = ops.arange(q_len)[:, None] + offset
        ki = ops.arange(kv_len)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

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

        has_image = pixel_values is not None and image_grid_thw is not None
        has_video = pixel_values_videos is not None and video_grid_thw is not None
        image_grid = video_grid = None
        if has_image or has_video:
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            embeds_flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
            if has_image:
                image_grid = ops.cast(ops.convert_to_tensor(image_grid_thw), "int32")
                image_embeds = self.get_image_features(pixel_values, image_grid)
                idx = [j for j, v in enumerate(ids_flat) if v == self.image_token_id]
                embeds_flat = ops.scatter_update(
                    embeds_flat,
                    ops.reshape(ops.convert_to_tensor(idx, dtype="int32"), (-1, 1)),
                    ops.cast(image_embeds, embeds_flat.dtype),
                )
            if has_video:
                video_grid = ops.cast(ops.convert_to_tensor(video_grid_thw), "int32")
                video_embeds = self.get_image_features(pixel_values_videos, video_grid)
                vidx = [j for j, v in enumerate(ids_flat) if v == self.video_token_id]
                embeds_flat = ops.scatter_update(
                    embeds_flat,
                    ops.reshape(ops.convert_to_tensor(vidx, dtype="int32"), (-1, 1)),
                    ops.cast(video_embeds, embeds_flat.dtype),
                )
            inputs_embeds = ops.reshape(embeds_flat, (batch, seq, self.embed_dim))
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid, video_grid, attention_mask=attention_mask
            )
        else:
            if attention_mask is not None:
                am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
                pos = ops.where(am == 0, 0, ops.cumsum(am, axis=-1) - 1)
            else:
                pos = ops.broadcast_to(ops.arange(seq), (batch, seq))
            position_ids = ops.broadcast_to(pos, (3, batch, seq))
        return inputs_embeds, position_ids, rope_deltas, {}

    def _merged_cos_sin(self, position_ids):
        cos3, sin3 = text_rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        return (
            merge_mrope(cos3, self.mrope_section),
            merge_mrope(sin3, self.mrope_section),
        )

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
            "vision_embed_dim": vc.get("embed_dim", 1280),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_mlp_ratio": vc.get("mlp_ratio", 4),
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
        from .convert_qwen2_vl_hf_to_keras import transfer_qwen2_vl_weights

        transfer_qwen2_vl_weights(keras_model, hf_state_dict)

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
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "mrope_section": self.mrope_section,
                "tie_embeddings": self.tie_embeddings,
                "vision_depth": self.vision_depth,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
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
class Qwen2VLConditionalGenerate(Qwen2VLModel, BaseGeneration):
    """Qwen2-VL with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a vocabulary projection on top of :class:`Qwen2VLModel` (a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, else the tied token
    embedding). ``call`` returns both ``logits`` and ``last_hidden_state``. Fast
    generation comes from :class:`~kerasformers.base.BaseGeneration`'s multimodal
    path: ``build_cache`` runs the vision encoder + 3-axis M-RoPE prefill ONCE
    (consuming ``pixel_values`` / ``image_grid_thw`` / ``pixel_values_videos`` /
    ``video_grid_thw``) into a fixed KV cache, then ``call_with_cache`` does text-only
    decode with the incremental M-RoPE position ``cache_len + rope_delta`` (carried in
    the cache). Pass pixels exactly as for :class:`Qwen2VLModel`:
    ``gen.generate(input_ids, pixel_values=..., image_grid_thw=...)``.
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
        # Eager multimodal prefill: run the vision encoder + scatter image/video
        # embeddings into the placeholder slots, compute 3-axis M-RoPE positions, run
        # the text decoder, and pad each layer's K/V into a fixed (max_len) cache. The
        # ``rope_deltas`` ride in the cache so decode continues the M-RoPE positions.
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
        logits = self.project(hidden[:, -1, :])  # language_model already final-normed
        return (kv_cache, rope_deltas), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # Text-only decode step; the new token's M-RoPE position is
        # ``cache_update_index + rope_delta`` on all three (t, h, w) axes.
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
class Qwen2VLTextGenerate(Qwen2VLConditionalGenerate):
    """Text-only counterpart of :class:`Qwen2VLConditionalGenerate` (no vision tower).

    Same Qwen2 decoder and fast ``.generate()``, built with ``build_vision=False`` so the
    ViT is never constructed; ``.generate()`` takes just token ids. The Qwen2-VL
    checkpoints are multimodal, so this head loads just their text backbone: on-the-fly
    ``hf:`` conversion copies only the language-model weights (the target-driven transfer
    never reads the vision keys), and a kerasformers checkpoint declaring
    ``Qwen2VLConditionalGenerate`` is read through :attr:`CHECKPOINT_SOURCE`.

        gen = Qwen2VLTextGenerate.from_weights("hf:Qwen/Qwen2-VL-2B-Instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    config_class = Qwen2VLTextConfig
    CHECKPOINT_SOURCE = CheckpointSource(
        "Qwen2VLConditionalGenerate",
        module="kerasformers.models.qwen2_vl.qwen2_vl_model",
        match="path",
    )

    def __init__(self, *args, **kwargs):
        kwargs["build_vision"] = False
        super().__init__(*args, **kwargs)
