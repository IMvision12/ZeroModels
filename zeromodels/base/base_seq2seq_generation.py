from keras import ops

from zeromodels.base.base_generation import BaseGeneration


class BaseSeq2SeqGeneration(BaseGeneration):
    """Encoder-decoder flavor of :class:`BaseGeneration` (Whisper / Speech2Text / Moonshine).

    Same optimized cross-backend decode engine as :class:`BaseGeneration`, but the prefill
    first runs an encoder over the source (audio features, source tokens, ...) and the
    decoder cross-attends to that frozen context at every step; the decoder "prompt" is
    the start / forced tokens rather than a user prompt.

    This base owns the **cache mechanics** so a model only writes its forward:

    * :meth:`build_cache` / :meth:`call_with_cache` -- generic: allocate a fixed-size
      per-layer self-attention KV cache (zeros), pull the static cross-attention KV from
      the model, then run the model's ``decode_forward`` (prefill, then one token/step).
      The opaque ``cache`` is a tuple of ``(self_k, self_v, cross_k, cross_v)`` per layer.
    * :meth:`cached_self_attention` / :meth:`cached_cross_attention` -- the per-layer cache
      primitives (write the new K/V at ``cache_update_index`` + causal mask; static cross
      KV), reusable by any Bart-style attention exposing ``query`` / ``project`` / ``attend``.

    A model implements:

    * ``encode(encoder_inputs) -> encoder_hidden_states``.
    * ``decode_cross_kv(encoder_hidden_states) -> [(cross_k, cross_v), ...]`` -- the static,
      head-split cross-attention K/V, one pair per decoder layer.
    * ``decode_forward(ids, cache, start_pos) -> (logits, new_cache)`` -- its block forward
      (embedding + positions + per-layer self/cross/FFN), using the two cache primitives.
    * ``decode_num_heads`` / ``decode_head_dim`` attributes (self-cache buffer shape).

    Speech2Text is wired onto this. :meth:`greedy_decode` is the **cacheless** fallback
    (O(n^2), uncompiled) that Whisper + Moonshine still use until they implement the hooks
    above; Whisper's forced-decoder-ids + suppress-tokens then fold into ``decode_forward``.
    """

    def greedy_decode(
        self,
        encoder_hidden_states,
        decoder_start_token_id,
        eos_token_id,
        max_new_tokens,
        forced_ids=None,
        logits_processor=None,
    ):
        forced_ids = forced_ids or {}
        batch = encoder_hidden_states.shape[0]
        generated = ops.full((batch, 1), decoder_start_token_id, dtype="int32")
        done = ops.zeros((batch,), dtype="bool")
        for step in range(max_new_tokens):
            cur_pos = generated.shape[1]
            if cur_pos in forced_ids:
                next_ids = ops.full((batch,), forced_ids[cur_pos], dtype="int32")
            else:
                logits = self.decoder(
                    {
                        "decoder_input_ids": generated,
                        "encoder_hidden_states": encoder_hidden_states,
                    }
                )
                next_logits = logits[:, -1, :]
                if logits_processor is not None:
                    next_logits = logits_processor(step, next_logits)
                next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")
            next_ids = ops.cast(ops.where(done, eos_token_id, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos_token_id))
            if bool(ops.all(done)):
                break
        return generated

    def encode(self, encoder_inputs):
        raise NotImplementedError(f"{type(self).__name__} must implement encode().")

    def decode_cross_kv(self, encoder_hidden_states):
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_cross_kv()."
        )

    def decode_forward(self, ids, cache, start_pos):
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_forward()."
        )

    @staticmethod
    def cached_self_attention(
        attn, hidden_states, cache_k, cache_v, update_index, rotary=None
    ):
        q = attn.query(hidden_states)
        k_new, v_new = attn.project(hidden_states)
        if rotary is not None:
            q = rotary(q, update_index)
            k_new = rotary(k_new, update_index)
        cache_k = ops.slice_update(cache_k, (0, 0, update_index, 0), k_new)
        cache_v = ops.slice_update(cache_v, (0, 0, update_index, 0), v_new)
        n = hidden_states.shape[1]
        max_len = cache_k.shape[2]
        q_pos = update_index + ops.arange(n)
        k_pos = ops.arange(max_len)
        mask = ops.where(k_pos[None, :] <= q_pos[:, None], 0.0, -1e9)[None, None]
        return attn.attend(q, cache_k, cache_v, mask), cache_k, cache_v

    @staticmethod
    def cached_cross_attention(attn, hidden_states, cross_k, cross_v):
        return attn.attend(attn.query(hidden_states), cross_k, cross_v, None)

    def build_cache(self, decoder_start_ids, encoder_hidden_states, max_len):
        batch = decoder_start_ids.shape[0]
        heads = self.decode_num_heads
        head_dim = self.decode_head_dim
        cache = tuple(
            (
                ops.zeros((batch, heads, max_len, head_dim)),
                ops.zeros((batch, heads, max_len, head_dim)),
                cross_k,
                cross_v,
            )
            for cross_k, cross_v in self.decode_cross_kv(encoder_hidden_states)
        )
        logits, cache = self.decode_forward(decoder_start_ids, cache, 0)
        return cache, logits[:, -1, :]

    def call_with_cache(self, token_ids, cache, cache_update_index):
        logits, cache = self.decode_forward(token_ids, cache, cache_update_index)
        return logits[:, -1, :], cache

    def generate_step(
        self, encoder_inputs, decoder_start_ids, noise, max_new_tokens, eos, sampler
    ):
        decoder_start_ids = ops.cast(ops.convert_to_tensor(decoder_start_ids), "int32")
        prompt_len = int(decoder_start_ids.shape[1])
        encoder_hidden_states = self.encode(encoder_inputs)
        cache, logits = self.build_cache(
            decoder_start_ids, encoder_hidden_states, prompt_len + max_new_tokens
        )
        return self.decode_loop(
            cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
        )

    def generate(
        self,
        encoder_inputs,
        decoder_input_ids,
        max_new_tokens=None,
        eos_token_id=None,
        sampler=None,
        seed=None,
    ):
        max_new_tokens, eos, sampler, seed = self.resolve_generation_args(
            max_new_tokens, eos_token_id, sampler, seed
        )
        decoder_input_ids = ops.cast(ops.convert_to_tensor(decoder_input_ids), "int32")
        batch = int(decoder_input_ids.shape[0])
        sampler_key = (
            type(sampler).__name__,
            tuple(sorted(sampler.get_config().items())),
        )
        cache_key = (max_new_tokens, eos, sampler_key)
        fn = self.cached_generate_function(cache_key, max_new_tokens, eos, sampler)
        noise = self.draw_noise(sampler, max_new_tokens, batch, seed)
        return self.run_compiled(fn, (encoder_inputs, decoder_input_ids), noise)
