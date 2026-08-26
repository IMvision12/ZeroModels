import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .gpt2_config import GPT2Config
from .gpt2_layers import MASK_NEG, GPT2Block

GPT2_HUB_SIBLINGS = frozenset({"GPT2Model", "GPT2TextGenerate"})


def gpt2_backbone_features(
    input_ids, attention_mask, token_embedding, wpe, blocks, ln_f, causal_mask
):
    positions = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    hidden = token_embedding(input_ids) + wpe(positions)
    mask = causal_mask(input_ids, attention_mask)
    for block in blocks:
        hidden = block(hidden, attention_mask=mask)
    return ln_f(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class GPT2Model(BaseModel):
    """GPT-2 decoder-only transformer backbone (no LM head).

    Learned token (``wte``) + absolute-position (``wpe``) embeddings, a stack of
    pre-LayerNorm causal blocks, and a final LayerNorm (``ln_f``). A functional
    model: the forward is a static graph over ``input_ids`` / ``attention_mask``.
    Returns ``last_hidden_state``; use :class:`GPT2TextGenerate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: Feed-forward hidden width per block.
        num_layers: Number of decoder blocks.
        num_heads: Attention heads per block.
        max_position_embeddings: Size of the learned position table.
        norm_eps: LayerNorm epsilon.
        tie_embeddings: Whether :class:`GPT2TextGenerate` ties the LM head to ``wte``.
    """

    HF_MODEL_TYPE = "gpt2"
    BASE_MODEL_CONFIG = None
    # Weights load by Hub repo id, e.g. from_weights("zeromodels/gpt2"), via
    # kf_config.json on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    config_class = GPT2Config
    HUB_REPO_SIBLINGS = GPT2_HUB_SIBLINGS
    # GPT2TextGenerate flips this on to also emit tied-head logits from the graph.
    output_logits = False

    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        max_position_embeddings=1024,
        norm_eps=1e-5,
        tie_embeddings=True,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        token_embedding = layers.Embedding(vocab_size, embed_dim, name="wte")
        wpe = layers.Embedding(max_position_embeddings, embed_dim, name="wpe")
        blocks = [
            GPT2Block(embed_dim, mlp_dim, num_heads, norm_eps, name=f"block_{i}")
            for i in range(num_layers)
        ]
        ln_f = layers.LayerNormalization(epsilon=norm_eps, name="ln_f")
        causal_mask = CausalMask(name="causal_mask")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = gpt2_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding,
            wpe,
            blocks,
            ln_f,
            causal_mask,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = TiedHead(token_embedding, name="lm_head")(hidden)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.token_embedding = token_embedding
        self.wpe = wpe
        self.blocks = blocks
        self.ln_f = ln_f
        self.causal_mask_layer = causal_mask
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["n_embd"],
            "mlp_dim": hf_config.get("n_inner") or 4 * hf_config["n_embd"],
            "num_layers": hf_config["n_layer"],
            "num_heads": hf_config["n_head"],
            "max_position_embeddings": hf_config["n_positions"],
            "norm_eps": hf_config.get("layer_norm_epsilon", 1e-5),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gpt2_hf_to_keras import transfer_gpt2_weights

        transfer_gpt2_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_position_embeddings": self.max_position_embeddings,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GPT2TextGenerate(GPT2Model, BaseGeneration):
    """GPT-2 backbone + a (tied) language-model head and fast ``.generate()``.

    The forward graph returns ``logits`` ``(batch, seq, vocab_size)`` and
    ``last_hidden_state``. The LM head is the transposed token embedding (GPT-2
    ties them), applied by the weightless :class:`~zeromodels.base.TiedHead` so
    no extra weight is added and the graph reads the live embedding. Fast generation
    comes from
    :class:`~zeromodels.base.BaseGeneration`, fulfilled here by ``build_cache``
    (parallel prefill into a fixed KV cache) and ``call_with_cache`` (one compiled
    decode step); GPT-2 uses learned absolute positions (``wpe``), so no rotary
    tables are threaded. Constructor ``Args`` are inherited from :class:`GPT2Model`.
    """

    output_logits = True
    eos_token_id = (50256,)  # GPT-2 <|endoftext|>

    def project(self, hidden):
        kernel = ops.transpose(ops.cast(self.token_embedding.embeddings, hidden.dtype))
        return ops.matmul(hidden, kernel)

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nh, hd = self.num_heads, self.embed_dim // self.num_heads
        positions = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        hidden = self.token_embedding(token_ids) + self.wpe(positions)
        causal = self.causal_mask(prompt_len, padding_mask)
        layer_caches = []
        for block in self.blocks:
            hidden, (k, v) = block(hidden, attention_mask=causal, use_cache=True)
            ck = ops.slice_update(
                ops.zeros((batch, nh, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nh, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.ln_f(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids) + self.wpe(positions)
        layer_caches = []
        for i, block in enumerate(self.blocks):
            h, ck, cv = block.decode_step(
                h, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.ln_f(h))[:, 0, :]
        return logits, cache
