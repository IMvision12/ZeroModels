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

from .internvl_config import InternVLConfig
from .internvl_layers import (
    InternVLDecoderLayer,
    InternVLMultiModalProjector,
    InternVLRMSNorm,
    InternVLVisionEmbeddings,
    InternVLVisionLayer,
    make_vision_norm,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLVisionModel(layers.Layer):
    """InternViT vision tower: conv patch embed + CLS/pos embeddings ->
    layer-scaled pre-norm blocks (-> optional final LayerNorm).

    The 300M tower (1B-14B checkpoints) uses LayerNorm blocks with biased
    attention; the 6B tower (38B/78B) uses RMSNorm blocks with bias-free
    attention and full-width QK RMS-norm. With ``use_mean_pooling`` (every
    InternVL3 checkpoint) the final norm is the identity, matching HF.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of vision blocks.
        num_heads: Vision attention heads.
        image_size: Pretrained square input size in pixels.
        patch_size: Patch size in pixels.
        attention_bias: Whether vision q/k/v carry a bias.
        qk_norm: Whether vision attention RMS-normalizes full-width q/k.
        norm_type: ``"layer_norm"`` (300M) or ``"rms_norm"`` (6B).
        norm_eps: Norm epsilon.
        layer_scale_init: Initial layer-scale (overwritten by checkpoints).
        use_mean_pooling: When ``True`` the final norm is skipped.

    Call args:
        pixel_values: ``(num_tiles, H, W, 3)`` (or channels-first).

    Returns:
        ``(num_tiles, num_patches + 1, embed_dim)`` token sequence (CLS first).
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        image_size=448,
        patch_size=14,
        attention_bias=True,
        qk_norm=False,
        norm_type="layer_norm",
        norm_eps=1e-6,
        layer_scale_init=0.1,
        use_mean_pooling=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.layer_scale_init = layer_scale_init
        self.use_mean_pooling = use_mean_pooling

        self.embeddings = InternVLVisionEmbeddings(
            embed_dim, image_size, patch_size, name="embeddings"
        )
        self.blocks = [
            InternVLVisionLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                attention_bias,
                qk_norm,
                norm_type,
                norm_eps,
                layer_scale_init,
                name=f"blocks_{i}",
            )
            for i in range(num_layers)
        ]
        self.vision_norm = (
            None
            if use_mean_pooling
            else make_vision_norm(norm_type, norm_eps, "vision_norm")
        )

    def call(self, pixel_values):
        hidden = self.embeddings(pixel_values)
        for block in self.blocks:
            hidden = block(hidden)
        if self.vision_norm is not None:
            hidden = self.vision_norm(hidden)
        return hidden

    def compute_output_spec(self, pixel_values):
        # (num_tiles, num_patches + 1 CLS, embed_dim); num_tiles is dynamic. The
        # grid-dependent conv patch-embed runs eagerly at runtime; the symbolic
        # build uses this spec so the tower's sublayers still materialize.
        num_patches = (self.image_size // self.patch_size) ** 2
        return keras.KerasTensor(
            (pixel_values.shape[0], num_patches + 1, self.embed_dim),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "norm_type": self.norm_type,
                "norm_eps": self.norm_eps,
                "layer_scale_init": self.layer_scale_init,
                "use_mean_pooling": self.use_mean_pooling,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLTextModel(layers.Layer):
    """Qwen2-style causal decoder: ``embed -> num_layers x InternVLDecoderLayer
    -> RMSNorm``.

    The token embedding lives here (``token_embedding``); ``call`` takes the
    pre-computed multimodal-fused ``inputs_embeds`` and rotary tables, and
    threads an optional KV cache for incremental decoding.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon.

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeds.
        cos, sin: rotary tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_values: optional list of per-layer ``(key, value)`` entries.
        use_cache: when ``True``, also return the updated per-layer cache.

    Returns:
        ``(batch, seq, embed_dim)``, or ``(hidden, new_cache)`` when
        ``use_cache``.
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
        attention_bias=True,
        qk_norm=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
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
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = tuple(mlp_only_layers)

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            InternVLDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                norm_eps,
                attention_bias=attention_bias,
                qk_norm=qk_norm,
                use_moe=self.is_moe_layer(i),
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_mlp_dim=moe_mlp_dim,
                norm_topk_prob=norm_topk_prob,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = InternVLRMSNorm(eps=norm_eps, name="final_norm")

    def is_moe_layer(self, i):
        # Mirrors Qwen3-MoE: every ``decoder_sparse_step``-th layer is sparse
        # unless it is pinned dense in ``mlp_only_layers``. num_experts == 0
        # (the qwen2 / qwen3 dense towers) makes every layer dense.
        return (
            i not in self.mlp_only_layers
            and self.num_experts > 0
            and (i + 1) % self.decoder_sparse_step == 0
        )

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
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
            }
        )
        return config


def internvl_pixel_shuffle(vision_features, scale):
    # Port of HF InternVLModel.pixel_shuffle on (B, W, H, C) feature maps: fuse
    # each (1/scale x 1/scale) patch group channel-wise. ``-1`` carries the
    # dynamic tile-count batch so the reshapes trace symbolically.
    w = int(vision_features.shape[1])
    h = int(vision_features.shape[2])
    c = int(vision_features.shape[3])
    x = ops.reshape(vision_features, (-1, w, int(h * scale), int(c / scale)))
    x = ops.transpose(x, (0, 2, 1, 3))
    x = ops.reshape(x, (-1, int(h * scale), int(w * scale), int(c / (scale**2))))
    return ops.transpose(x, (0, 2, 1, 3))


def internvl_image_features(
    pixel_values,
    vision_tower,
    multi_modal_projector,
    downsample_ratio,
    vision_embed_dim,
    projector_input_dim,
):
    # Vision tower -> drop CLS -> spatial grid -> pixel shuffle -> project.
    features = vision_tower(pixel_values)[:, 1:, :]
    n = int(features.shape[1])
    fs = int(round(n**0.5))
    features = ops.reshape(features, (-1, fs, fs, vision_embed_dim))
    features = internvl_pixel_shuffle(features, downsample_ratio)
    features = ops.reshape(features, (-1, projector_input_dim))
    return multi_modal_projector(features)


def internvl_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def internvl_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    *,
    language_model,
    vision_tower,
    multi_modal_projector,
    image_merge,
    causal_mask,
    head_dim,
    rope_theta,
    downsample_ratio,
    vision_embed_dim,
    projector_input_dim,
    image_token_id,
    compute_dtype,
):
    # Zero the image placeholder ids before the lookup (they may sit outside the
    # embedding range) -- those slots are overwritten by the merge anyway.
    safe_ids = ops.where(ops.equal(input_ids, image_token_id), 0, input_ids)
    inputs_embeds = language_model.token_embedding(safe_ids)
    image_embeds = internvl_image_features(
        pixel_values,
        vision_tower,
        multi_modal_projector,
        downsample_ratio,
        vision_embed_dim,
        projector_input_dim,
    )
    inputs_embeds = image_merge(inputs_embeds, input_ids, image_embeds)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = internvl_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    return language_model(inputs_embeds, cos, sin, attention_mask=mask)


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLModel(BaseModel):
    """InternVL3 multimodal backbone: InternViT tower + pixel-shuffle projector
    + Qwen2-style decoder.

    Tiled 448x448 images run through the vision tower; the CLS token is
    dropped, the 32x32 patch grid is pixel-shuffled (``downsample_ratio`` 0.5,
    so 4 neighbouring patches fuse channel-wise into one of 256 tokens per
    tile), projected to the text width, and scattered into the
    ``image_token_id`` (``<IMG_CONTEXT>``) placeholder slots of the decoder
    input. Standard 1D rotary positions. A functional model: the vision tower +
    merge run inside the graph over ``{input_ids, attention_mask, pixel_values}``
    (image inputs always present; an absent image token merges as a no-op). Use
    :class:`InternVLConditionalGenerate` for logits / text.

    Construction:

    >>> InternVLModel.from_weights("zeromodels/internvl3-1b")
    >>> InternVLModel.from_weights("hf:OpenGVLab/InternVL3-1B-hf")

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer (GQA).
        norm_eps: Text RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`InternVLConditionalGenerate` ties the LM head
            (every InternVL3-hf checkpoint materializes ``lm_head``: False).
        vision_embed_dim, vision_mlp_dim, vision_num_layers, vision_num_heads:
            InternViT tower dimensions.
        image_size: Tile size in pixels (448).
        patch_size: Vision patch size in pixels (14).
        vision_attention_bias: Whether vision q/k/v carry a bias (300M: True).
        vision_qk_norm: Vision full-width QK RMS-norm (6B tower: True).
        vision_norm_type: ``"layer_norm"`` (300M) or ``"rms_norm"`` (6B).
        vision_norm_eps: Vision norm epsilon.
        vision_layer_scale_init: Initial vision layer-scale value.
        downsample_ratio: Pixel-shuffle scale factor (0.5).
        image_token_id: ``<IMG_CONTEXT>`` placeholder id replaced by projected
            vision tokens.
    """

    HF_MODEL_TYPE = "internvl"
    default_load_dtype = (
        "bfloat16"  # InternVL3/3.5 checkpoints (OpenGVLab + hosted) are bf16
    )
    config_class = InternVLConfig
    output_logits = False

    def __init__(
        self,
        vocab_size=151674,
        embed_dim=896,
        mlp_dim=4864,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=None,
        text_backbone="qwen2",
        norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_layers=24,
        vision_num_heads=16,
        image_size=448,
        patch_size=14,
        vision_attention_bias=True,
        vision_qk_norm=False,
        vision_norm_type="layer_norm",
        vision_norm_eps=1e-6,
        vision_layer_scale_init=0.1,
        downsample_ratio=0.5,
        image_token_id=151667,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads
        qk_norm = text_backbone in ("qwen3", "qwen3_moe")
        attention_bias = text_backbone == "qwen2"
        projector_input_dim = vision_embed_dim * int(1 / downsample_ratio) ** 2
        mlp_only_layers = tuple(mlp_only_layers)

        vision_tower = InternVLVisionModel(
            embed_dim=vision_embed_dim,
            mlp_dim=vision_mlp_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            image_size=image_size,
            patch_size=patch_size,
            attention_bias=vision_attention_bias,
            qk_norm=vision_qk_norm,
            norm_type=vision_norm_type,
            norm_eps=vision_norm_eps,
            layer_scale_init=vision_layer_scale_init,
            name="vision_tower",
        )
        multi_modal_projector = InternVLMultiModalProjector(
            projector_input_dim, embed_dim, name="multi_modal_projector"
        )
        language_model = InternVLTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            attention_bias=attention_bias,
            qk_norm=qk_norm,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_mlp_dim=moe_mlp_dim,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            name="language_model",
        )
        causal_mask = CausalMask(name="causal_mask")
        image_merge = MediaMerge(image_token_id, embed_dim, name="image_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        img_shape = (
            (3, image_size, image_size)
            if keras.config.image_data_format() == "channels_first"
            else (image_size, image_size, 3)
        )
        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=img_shape, dtype="float32", name="pixel_values"
            ),
        }
        hidden = internvl_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            language_model=language_model,
            vision_tower=vision_tower,
            multi_modal_projector=multi_modal_projector,
            image_merge=image_merge,
            causal_mask=causal_mask,
            head_dim=head_dim,
            rope_theta=rope_theta,
            downsample_ratio=downsample_ratio,
            vision_embed_dim=vision_embed_dim,
            projector_input_dim=projector_input_dim,
            image_token_id=image_token_id,
            compute_dtype=language_model.token_embedding.compute_dtype,
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

        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        self.language_model = language_model
        self.causal_mask_layer = causal_mask
        self.image_merge = image_merge
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.text_backbone = text_backbone
        self.qk_norm = qk_norm
        self.attention_bias = attention_bias
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_attention_bias = vision_attention_bias
        self.vision_qk_norm = vision_qk_norm
        self.vision_norm_type = vision_norm_type
        self.vision_norm_eps = vision_norm_eps
        self.vision_layer_scale_init = vision_layer_scale_init
        self.downsample_ratio = downsample_ratio
        self.image_token_id = image_token_id
        self.projector_input_dim = projector_input_dim

        # The vision tower's grid-dependent call is skipped by the symbolic
        # auto-build, so its blocks stay unbuilt; a concrete dummy forward
        # materializes every weight for from_weights (which loads before a call).
        with inference_scope():
            self(self.dummy_media_inputs())

    def dummy_media_inputs(self):
        n = int((self.image_size // self.patch_size * self.downsample_ratio) ** 2)
        return {
            "input_ids": ops.concatenate(
                [
                    ops.zeros((1, 1), dtype="int32"),
                    ops.full((1, n), self.image_token_id, dtype="int32"),
                    ops.ones((1, 1), dtype="int32"),
                ],
                axis=1,
            ),
            "attention_mask": ops.ones((1, n + 2), dtype="int32"),
            "pixel_values": ops.zeros(
                (1, 3, self.image_size, self.image_size)
                if keras.config.image_data_format() == "channels_first"
                else (1, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
        }

    def build_for_transfer(self):
        with inference_scope():
            self(self.dummy_media_inputs())

    def get_image_features(self, pixel_values):
        return internvl_image_features(
            pixel_values,
            self.vision_tower,
            self.multi_modal_projector,
            self.downsample_ratio,
            self.vision_embed_dim,
            self.projector_input_dim,
        )

    def rope_tables(self, position_ids):
        return internvl_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def prepare_inputs(self, input_ids, pixel_values, attention_mask):
        # Imperative fuse for the KV-cache prefill: handles absent media (None).
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        safe_ids = ops.where(ops.equal(input_ids, self.image_token_id), 0, input_ids)
        inputs_embeds = self.language_model.token_embedding(safe_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
            inputs_embeds = merge_media(
                inputs_embeds,
                input_ids,
                image_embeds,
                self.image_token_id,
                self.embed_dim,
            )
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        return inputs_embeds, position_ids

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        # The text tower type drives QK-norm / attention-bias / dense-vs-MoE: the
        # HF InternVL config nests a full text sub-config, so its model_type is
        # qwen2 (InternVL3), qwen3 (InternVL3.5 dense), or qwen3_moe (3.5 MoE).
        backbone = text.get("model_type", "qwen2")
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "head_dim": text.get("head_dim"),
            "text_backbone": backbone,
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 1000000.0),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
            "num_experts": text.get("num_experts", 0),
            "num_experts_per_tok": text.get("num_experts_per_tok", 0),
            "moe_mlp_dim": text.get("moe_intermediate_size", 0),
            "norm_topk_prob": bool(text.get("norm_topk_prob", True)),
            "decoder_sparse_step": text.get("decoder_sparse_step", 1),
            "mlp_only_layers": tuple(text.get("mlp_only_layers", ()) or ()),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": (
                vision["image_size"][0]
                if isinstance(vision.get("image_size"), (list, tuple))
                else vision.get("image_size", 448)
            ),
            "patch_size": (
                vision["patch_size"][0]
                if isinstance(vision.get("patch_size"), (list, tuple))
                else vision.get("patch_size", 14)
            ),
            "vision_attention_bias": vision.get("attention_bias", True),
            "vision_qk_norm": vision.get("use_qk_norm", False),
            "vision_norm_type": vision.get("norm_type", "layer_norm"),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "vision_layer_scale_init": vision.get("layer_scale_init_value", 0.1),
            "downsample_ratio": hf_config.get("downsample_ratio", 0.5),
            "image_token_id": hf_config.get("image_token_id", 151667),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_internvl_hf_to_keras import transfer_internvl_weights

        transfer_internvl_weights(keras_model, hf_state_dict)

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
                "text_backbone": self.text_backbone,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_attention_bias": self.vision_attention_bias,
                "vision_qk_norm": self.vision_qk_norm,
                "vision_norm_type": self.vision_norm_type,
                "vision_norm_eps": self.vision_norm_eps,
                "vision_layer_scale_init": self.vision_layer_scale_init,
                "downsample_ratio": self.downsample_ratio,
                "image_token_id": self.image_token_id,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLConditionalGenerate(InternVLModel, BaseGeneration):
    """InternVL3 with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a vocabulary projection on top of :class:`InternVLModel` (a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``: every
    InternVL3-hf checkpoint: else the tied token embedding). The forward graph
    returns both ``logits`` and ``last_hidden_state``. Fast generation comes from
    :class:`~zeromodels.base.BaseGeneration`'s multimodal path:
    ``build_cache`` runs the vision tower + projector + fused prefill ONCE
    (consuming ``pixel_values``) into a fixed KV cache, then
    ``call_with_cache`` does text-only decode. Pass pixels exactly as for
    :class:`InternVLModel`:

        gen.generate(input_ids, pixel_values=...)
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

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        # Multimodal prefill: vision tower + projector + placeholder merge,
        # then the text decoder writes each layer's K/V into a fixed
        # (B, num_layers, 2, num_kv_heads, max_len, head_dim) cache.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv = self.language_model.num_kv_heads
        hd = self.language_model.head_dim
        inputs_embeds, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
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
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(hidden[:, -1, :])  # language_model already final-normed
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # Text-only decode step at position ``cache_update_index``.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.language_model.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.language_model.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.language_model.final_norm(h))[:, 0, :]
        return logits, cache
