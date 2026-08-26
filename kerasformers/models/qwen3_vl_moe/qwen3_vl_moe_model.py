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
from kerasformers.models.qwen2_vl.qwen2_vl_model import Qwen2VLModel
from kerasformers.models.qwen3_moe.qwen3_moe_layers import Qwen3MoeSparseBlock
from kerasformers.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3VLMLP,
    Qwen3VLRMSNorm,
    Qwen3VLTextAttention,
)
from kerasformers.models.qwen3_vl.qwen3_vl_model import (
    Qwen3VLModel,
    Qwen3VLVisionModel,
    qwen3_vl_multimodal_features,
    qwen3_vl_text_features,
)

from .qwen3_vl_moe_config import (
    QWEN3_VL_MOE_TOKENS,
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMoeTextDecoderLayer(layers.Layer):
    """Qwen3-VL-MoE decoder block: QK-norm attention then a sparse MoE (or dense) MLP.

    Pre-norm residual block ``h = x + attn(attention_norm(x))`` then
    ``h = h + mlp(mlp_norm(h))``, where ``attn`` is the Qwen3-VL QK-norm GQA attention
    (interleaved M-RoPE) and ``mlp`` is a Qwen3-MoE sparse block (routed experts, no
    shared expert) when ``use_moe`` else a dense SwiGLU MLP. Same ``call`` / ``decode_step``
    interface as the dense Qwen3-VL decoder layer, so the shared generation loop is unchanged.

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim / norm_eps: Attention geometry.
        mlp_dim: Dense-MLP width (used when ``use_moe`` is ``False``).
        num_experts / num_experts_per_tok / moe_mlp_dim / norm_topk_prob: MoE routing.
        use_moe: Sparse MoE block when ``True``, else a dense SwiGLU MLP.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        num_experts,
        num_experts_per_tok,
        moe_mlp_dim,
        norm_topk_prob=True,
        norm_eps=1e-6,
        use_moe=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.norm_eps = norm_eps
        self.use_moe = use_moe
        self.attention_norm = Qwen3VLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen3VLTextAttention(
            embed_dim, num_heads, num_kv_heads, head_dim, norm_eps, name="attention"
        )
        self.mlp_norm = Qwen3VLRMSNorm(eps=norm_eps, name="mlp_norm")
        if use_moe:
            self.mlp = Qwen3MoeSparseBlock(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                norm_topk_prob,
                name="mlp",
            )
        else:
            self.mlp = Qwen3VLMLP(embed_dim, mlp_dim, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_out = self.attention(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "norm_eps": self.norm_eps,
                "use_moe": self.use_moe,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMoeTextModel(layers.Layer):
    """Qwen3-VL-MoE causal decoder with DeepStack visual-feature injection.

    Identical to the dense Qwen3-VL text model except each decoder layer's MLP is a
    Qwen3-MoE sparse block (per-layer dense-vs-MoE via ``decoder_sparse_step`` /
    ``mlp_only_layers``). During prefill the i-th DeepStack feature map (scattered to a
    full ``(batch, seq, embed_dim)`` tensor by the model) is added to the output of
    decoder layer ``i``.

    Args mirror the flat Qwen3-VL-MoE text hyperparameters (see [`Qwen3VLMoeModel`]).

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeddings.
        cos, sin: merged interleaved-M-RoPE tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to ``(batch, 1, q_len, kv_len)``.
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
        num_experts,
        num_experts_per_tok,
        moe_mlp_dim,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = tuple(mlp_only_layers)
        self.norm_eps = norm_eps
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3VLMoeTextDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_experts,
                num_experts_per_tok,
                moe_mlp_dim,
                norm_topk_prob,
                norm_eps,
                use_moe=(
                    i not in self.mlp_only_layers
                    and num_experts > 0
                    and (i + 1) % decoder_sparse_step == 0
                ),
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
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMoeModel(Qwen3VLModel):
    """Qwen3-VL-MoE multimodal backbone: ViT + DeepStack + Qwen3-VL MoE decoder.

    Identical to :class:`Qwen3VLModel` (vision tower with DeepStack, interleaved M-RoPE,
    QK-norm GQA attention, image/video fusion) except the text decoder's MLP is a
    sparse Mixture-of-Experts (routed experts, no shared expert). Returns raw features;
    use :class:`Qwen3VLMoeConditionalGenerate` for logits / text.

    Output dict (vision keys optional: images, video, both, or neither):

    .. code-block:: python

        out = model({
            "input_ids": ...,            # (B, L) int, image/video placeholders
            "pixel_values": ...,         # (num_patches, patch_dim) image patches
            "image_grid_thw": ...,       # (num_images, 3) per-image (t, h, w)
        })
        out["last_hidden_state"]   # (B, L, embed_dim)

    Reference:
        - `Qwen3 Technical Report <https://arxiv.org/abs/2505.09388>`_
    """

    HF_MODEL_TYPE = ("qwen3_vl_moe", "qwen3_vl_moe_text")
    default_load_dtype = "bfloat16"
    config_class = Qwen3VLMoeConfig

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=5000000.0,
        mrope_section=(24, 20, 20),
        tie_embeddings=True,
        num_experts=60,
        num_experts_per_tok=4,
        moe_mlp_dim=1408,
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
        deepstack_visual_indexes=(8, 16, 24),
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN3_VL_MOE_TOKENS["image_token_id"],
        video_token_id=QWEN3_VL_MOE_TOKENS["video_token_id"],
        vision_start_token_id=QWEN3_VL_MOE_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN3_VL_MOE_TOKENS["vision_end_token_id"],
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
        language_model = Qwen3VLMoeTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_mlp_dim=moe_mlp_dim,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
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

        # Skip Qwen3VLModel/Qwen2VLModel.__init__ (they build the dense text graph);
        # go straight to the functional keras init after Qwen2VLModel in the MRO.
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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
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
            "mlp_dim": tc.get("intermediate_size", 5632),
            "num_layers": tc["num_hidden_layers"],
            "num_heads": heads,
            "num_kv_heads": tc.get("num_key_value_heads", heads),
            "head_dim": tc.get("head_dim", hidden // heads),
            "norm_eps": tc.get("rms_norm_eps", 1e-6),
            "rope_theta": tc.get("rope_theta", 5000000.0),
            "mrope_section": tuple(mrope),
            "tie_embeddings": hf_config.get(
                "tie_word_embeddings", tc.get("tie_word_embeddings", False)
            ),
            "num_experts": tc.get("num_experts", 60),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 4),
            "moe_mlp_dim": tc.get("moe_intermediate_size", 1408),
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
            "deepstack_visual_indexes": tuple(
                vc.get("deepstack_visual_indexes", (8, 16, 24))
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
        from .convert_qwen3_vl_moe_hf_to_keras import transfer_qwen3_vl_moe_weights

        transfer_qwen3_vl_moe_weights(keras_model, hf_state_dict)

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
            "num_experts",
            "num_experts_per_tok",
            "moe_mlp_dim",
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMoeConditionalGenerate(Qwen3VLMoeModel, BaseGeneration):
    """Qwen3-VL-MoE with an LM head + fast ``.generate()`` (image+text -> text).

    Same fast multimodal generation as :class:`~kerasformers.models.qwen3_vl.Qwen3VLConditionalGenerate`:
    ``build_cache`` runs the vision encoder + M-RoPE prefill into a fixed KV cache
    (DeepStack vision features threaded through the prefill; ``rope_deltas`` carried in
    the cache), then ``call_with_cache`` does text-only decode at M-RoPE position
    ``cache_idx + rope_delta``. The MoE MLP does not change the cache structure (every
    layer is full attention).
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
class Qwen3VLMoeTextGenerate(Qwen3VLMoeConditionalGenerate):
    """Text-only counterpart of :class:`Qwen3VLMoeConditionalGenerate` (no vision tower).

    Built with ``build_vision=False`` so the DeepStack ViT is never constructed;
    ``.generate()`` takes just token ids. Loads just the MoE text backbone of a
    Qwen3-VL-MoE checkpoint: the target-driven ``hf:`` transfer copies only the
    language-model weights, and a kerasformers checkpoint declaring
    ``Qwen3VLMoeConditionalGenerate`` is read via :attr:`CHECKPOINT_SOURCE`.

        gen = Qwen3VLMoeTextGenerate.from_weights("hf:Qwen/Qwen3-VL-30B-A3B-Instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    config_class = Qwen3VLMoeTextConfig
    CHECKPOINT_SOURCE = CheckpointSource(
        "Qwen3VLMoeConditionalGenerate",
        module="kerasformers.models.qwen3_vl_moe.qwen3_vl_moe_model",
        match="path",
    )

    def __init__(self, *args, **kwargs):
        kwargs["build_vision"] = False
        super().__init__(*args, **kwargs)
