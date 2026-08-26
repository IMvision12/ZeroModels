import keras
from keras import layers, ops

from zeromodels.base import (
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
)
from zeromodels.base.base_mixin import inference_scope
from zeromodels.models.qwen2.qwen2_layers import Qwen2DecoderLayer, Qwen2RMSNorm

from .locateanything_config import LocateAnythingConfig
from .locateanything_vision import LocateAnythingVisionModel

MASK_NEG = -1e9


def locateanything_backbone_features(
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_hws,
    *,
    token_embedding,
    vision_model,
    mlp1_norm,
    mlp1_fc1,
    mlp1_fc2,
    image_merge,
    causal_mask,
    decoder_layers,
    final_norm,
    head_dim,
    rope_theta,
):
    # Always-media multimodal graph: vision runs unconditionally (no-op merge when
    # the image token is absent), then the reused Qwen2 decoder (causal mask,
    # matching the old non-magi forward). The magi block mask stays in the
    # imperative MTP/PBD generation path.
    hidden = token_embedding(
        ops.where(ops.equal(input_ids, image_merge.token_id), 0, input_ids)
    )
    vit = vision_model(pixel_values, image_grid_hws)
    vit = mlp1_norm(vit)
    vit = ops.gelu(mlp1_fc1(vit), approximate=False)
    vit = mlp1_fc2(vit)
    hidden = image_merge(hidden, input_ids, vit)
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    cos, sin = ops.cos(emb), ops.sin(emb)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class LocateAnythingModel(BaseModel):
    """LocateAnything-3B backbone (no LM head).

    MoonViT vision tokens are projected by ``mlp1`` (LayerNorm -> Linear -> GELU
    -> Linear) and spliced into the Qwen2.5-3B token-embedding stream at the
    ``image_token_index`` positions, then run through the reused Qwen2 decoder.
    The forward here uses standard causal attention; the model's "magi" block
    attention + Parallel Box Decoding live in the generation path. Returns raw
    features; use :class:`LocateAnythingConditionalGenerate` for logits.
    """

    HF_MODEL_TYPE = "locateanything"
    default_load_dtype = "bfloat16"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = LocateAnythingConfig
    HUB_REPO_SIBLINGS = frozenset(
        {"LocateAnythingModel", "LocateAnythingConditionalGenerate"}
    )
    output_logits = False

    def __init__(
        self,
        vocab_size=152681,
        embed_dim=2048,
        mlp_dim=11008,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_embeddings=True,
        vision_embed_dim=1152,
        vision_depth=27,
        vision_num_heads=16,
        vision_mlp_dim=4304,
        vision_patch_size=14,
        vision_init_pos_h=64,
        vision_init_pos_w=64,
        merge_kernel=(2, 2),
        vision_rope_theta=10000.0,
        image_token_index=151665,
        block_size=6,
        max_position_embeddings=32768,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads

        vision_model = LocateAnythingVisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_dim=vision_mlp_dim,
            patch_size=vision_patch_size,
            init_pos_h=vision_init_pos_h,
            init_pos_w=vision_init_pos_w,
            merge_kernel=merge_kernel,
            rope_theta=vision_rope_theta,
            name="vision_model",
        )
        mlp1_norm = layers.LayerNormalization(epsilon=1e-5, name="mlp1_norm")
        mlp1_fc1 = layers.Dense(embed_dim, name="mlp1_fc1")
        mlp1_fc2 = layers.Dense(embed_dim, name="mlp1_fc2")
        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Qwen2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Qwen2RMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        image_merge = MediaMerge(image_token_index, embed_dim, name="image_merge")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "pixel_values": layers.Input(
                shape=(3, vision_patch_size, vision_patch_size),
                dtype="float32",
                name="pixel_values",
            ),
            "image_grid_hws": layers.Input(
                shape=(2,), dtype="int32", name="image_grid_hws"
            ),
        }
        hidden = locateanything_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_hws"],
            token_embedding=token_embedding,
            vision_model=vision_model,
            mlp1_norm=mlp1_norm,
            mlp1_fc1=mlp1_fc1,
            mlp1_fc2=mlp1_fc2,
            image_merge=image_merge,
            causal_mask=causal_mask,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            head_dim=head_dim,
            rope_theta=rope_theta,
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

        self.vision_model = vision_model
        self.mlp1_norm = mlp1_norm
        self.mlp1_fc1 = mlp1_fc1
        self.mlp1_fc2 = mlp1_fc2
        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
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
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_patch_size = vision_patch_size
        self.vision_init_pos_h = vision_init_pos_h
        self.vision_init_pos_w = vision_init_pos_w
        self.merge_kernel = tuple(merge_kernel)
        self.vision_rope_theta = vision_rope_theta
        self.image_token_index = image_token_index
        self.block_size = block_size
        self.max_position_embeddings = max_position_embeddings

        with inference_scope():
            self.build_for_transfer()

    def get_image_features(self, pixel_values, grid_hws):
        feats = self.vision_model(pixel_values, grid_hws)  # (M, vision_embed*merge)
        x = self.mlp1_norm(feats)
        x = ops.gelu(self.mlp1_fc1(x), approximate=False)
        return self.mlp1_fc2(x)  # (M, embed_dim)

    def merge_vision(self, hidden, input_ids, vit):
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        flat_ids = ops.reshape(input_ids, (batch * seq,))
        flat_hidden = ops.reshape(hidden, (batch * seq, self.embed_dim))
        mask = ops.equal(flat_ids, self.image_token_index)
        idx = ops.cumsum(ops.cast(mask, "int32")) - 1
        gathered = ops.take(vit, ops.maximum(idx, 0), axis=0)
        flat_hidden = ops.where(
            mask[:, None], ops.cast(gathered, flat_hidden.dtype), flat_hidden
        )
        return ops.reshape(flat_hidden, (batch, seq, self.embed_dim))

    def rope_tables(self, seq, batch, position_ids=None):
        if position_ids is None:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        inv_freq = 1.0 / ops.power(
            self.rope_theta,
            ops.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return ops.cos(emb), ops.sin(emb)

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def magi_mask(self, q_len, kv_len):
        """General magi block mask, additive ``(1, 1, q_len, kv_len)``. The first
        ``q_len - block_size`` query rows are causal; the last ``block_size``
        (window) rows attend everywhere except the single ``blocked_k`` column.
        Reduces to the prefill mask when ``q_len == kv_len`` and serves the cached
        MTP decode when ``q_len < kv_len``."""
        b = self.block_size
        q_start = kv_len - q_len
        r = q_len - b
        blocked_k = kv_len - b - 1
        qi = ops.arange(q_len)[:, None]
        ki = ops.arange(kv_len)[None, :]
        causal = ki <= (q_start + qi)
        window = ops.not_equal(ki, blocked_k)
        attend = ops.where(qi < r, causal, window)
        return ops.cast(ops.where(attend, 0.0, MASK_NEG), "float32")[None, None]

    def causal_mask_cached(self, q_len, kv_len):
        """Plain causal mask for cached AR decode (``q_len <= kv_len``)."""
        q_start = kv_len - q_len
        qi = ops.arange(q_len)[:, None]
        ki = ops.arange(kv_len)[None, :]
        attend = ki <= (q_start + qi)
        return ops.cast(ops.where(attend, 0.0, MASK_NEG), "float32")[None, None]

    def magi_prefill_mask(self, seq):
        return self.magi_mask(seq, seq)

    def forward_features(self, inputs):
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        vision_embeds = inputs.get("vision_embeds")
        pixel_values = inputs.get("pixel_values")
        grid_hws = inputs.get("image_grid_hws")
        if vision_embeds is not None:
            hidden = self.merge_vision(hidden, input_ids, vision_embeds)
        elif pixel_values is not None and grid_hws is not None:
            vit = self.get_image_features(pixel_values, grid_hws)
            hidden = self.merge_vision(hidden, input_ids, vit)
        cos, sin = self.rope_tables(seq, batch, inputs.get("position_ids"))
        if inputs.get("use_magi"):
            mask = self.magi_prefill_mask(seq)
        else:
            mask = self.causal_mask(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=mask)
        return self.final_norm(hidden)

    def forward_features_cached(
        self,
        input_ids,
        position_ids,
        vision_embeds=None,
        past_caches=None,
        causal=False,
    ):
        """Cached forward over only the new tokens. ``past_caches`` is a list of
        per-layer ``(key, value)`` (or ``None`` on prefill); returns the hidden
        states for the new tokens plus the updated per-layer caches."""
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        if vision_embeds is not None:
            hidden = self.merge_vision(hidden, input_ids, vision_embeds)
        cos, sin = self.rope_tables(q_len, batch, position_ids)
        past_len = int(past_caches[0][0].shape[2]) if past_caches is not None else 0
        kv_len = past_len + q_len
        mask = (
            self.causal_mask_cached(q_len, kv_len)
            if causal
            else self.magi_mask(q_len, kv_len)
        )
        new_caches = []
        for i, layer in enumerate(self.decoder_layers):
            past = past_caches[i] if past_caches is not None else None
            hidden, kv = layer(
                hidden,
                cos,
                sin,
                attention_mask=mask,
                past_key_value=past,
                use_cache=True,
            )
            new_caches.append(kv)
        return self.final_norm(hidden), new_caches

    @classmethod
    def config_from_hf(cls, hf_config):
        t = hf_config["text_config"]
        v = hf_config["vision_config"]
        return {
            "vocab_size": t["vocab_size"],
            "embed_dim": t["hidden_size"],
            "mlp_dim": t["intermediate_size"],
            "num_layers": t["num_hidden_layers"],
            "num_heads": t["num_attention_heads"],
            "num_kv_heads": t["num_key_value_heads"],
            "head_dim": t.get("head_dim")
            or t["hidden_size"] // t["num_attention_heads"],
            "norm_eps": t.get("rms_norm_eps", 1e-6),
            "rope_theta": t.get("rope_theta", 1000000.0),
            "tie_embeddings": t.get("tie_word_embeddings", True),
            "vision_embed_dim": v["hidden_size"],
            "vision_depth": v["num_hidden_layers"],
            "vision_num_heads": v["num_attention_heads"],
            "vision_mlp_dim": v["intermediate_size"],
            "vision_patch_size": v["patch_size"],
            "vision_init_pos_h": v["init_pos_emb_height"],
            "vision_init_pos_w": v["init_pos_emb_width"],
            "merge_kernel": tuple(v["merge_kernel_size"]),
            "image_token_index": hf_config.get("image_token_index", 151665),
            "block_size": t.get("block_size", 6),
            "max_position_embeddings": t.get("max_position_embeddings", 32768),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_locateanything_hf_to_keras import transfer_locateanything_weights

        transfer_locateanything_weights(keras_model, hf_state_dict)

    def build_for_transfer(self):
        grid = ops.convert_to_tensor([[2, 2]], dtype="int64")
        ps = self.vision_patch_size
        pixel_values = ops.zeros((4, 3, ps, ps), dtype="float32")
        input_ids = ops.convert_to_tensor(
            [[self.image_token_index, 0, 0, 0]], dtype="int32"
        )
        self(
            {
                "input_ids": input_ids,
                "attention_mask": ops.ones_like(input_ids),
                "pixel_values": pixel_values,
                "image_grid_hws": grid,
            }
        )

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
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_depth": self.vision_depth,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_patch_size": self.vision_patch_size,
                "vision_init_pos_h": self.vision_init_pos_h,
                "vision_init_pos_w": self.vision_init_pos_w,
                "merge_kernel": self.merge_kernel,
                "vision_rope_theta": self.vision_rope_theta,
                "image_token_index": self.image_token_index,
                "block_size": self.block_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LocateAnythingConditionalGenerate(LocateAnythingModel):
    """LocateAnything-3B with the (tied) Qwen2 LM head -> logits."""

    eos_token_id = (151645,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def forward_logits(self, inputs):
        return ops.cast(self.project(self.forward_features(inputs)), "float32")

    def forward_logits_cached(
        self,
        input_ids,
        position_ids,
        vision_embeds=None,
        past_caches=None,
        causal=False,
    ):
        hidden, caches = self.forward_features_cached(
            input_ids, position_ids, vision_embeds, past_caches, causal
        )
        return ops.cast(self.project(hidden), "float32"), caches

    def generate(
        self,
        input_ids,
        pixel_values=None,
        image_grid_hws=None,
        tokenizer=None,
        vision_embeds=None,
        max_new_tokens=512,
        generation_mode="hybrid",
        n_future=None,
        use_cache=True,
        **kwargs,
    ):
        from .locateanything_generation import generate_loop, generate_loop_cached

        loop = generate_loop_cached if use_cache else generate_loop
        if keras.backend.backend() == "torch":
            import torch

            grad_ctx = torch.no_grad()
        else:
            from contextlib import nullcontext

            grad_ctx = nullcontext()
        with grad_ctx:
            if vision_embeds is None and pixel_values is not None:
                vision_embeds = self.get_image_features(pixel_values, image_grid_hws)
            return loop(
                self,
                input_ids,
                vision_embeds,
                tokenizer,
                n_future=n_future or self.block_size,
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
