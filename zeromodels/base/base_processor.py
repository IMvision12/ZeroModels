import keras

from zeromodels.base.base_mixin import PreprocessorMixin


class BaseProcessor(PreprocessorMixin):
    """Base class for zeromodels multi-modal processors.

    A processor **composes** (has-a) a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseAudioFeatureExtractor`; it is not a
    subclass of them. Each subclass declares the component classes it uses via the
    ``TOKENIZER_CLS`` / ``IMAGE_PROCESSOR_CLS`` / ``FEATURE_EXTRACTOR_CLS`` class
    attributes and stores the built instances on ``self.tokenizer`` /
    ``self.image_processor`` / ``self.feature_extractor``. ``__init__`` accepts
    pre-built components (used by the loaders) or builds them from kwargs.

    The base then provides, generically over whatever components are declared:

    * ``from_hf(repo)``: loads **every** component from the HF ``repo`` (tokenizer
      files + image processor / feature extractor), so ``from_weights("hf:org/repo")``
      returns a complete processor.
    * ``get_config`` / ``from_config``: serialize/deserialize the components.
    * ``decode`` / ``batch_decode``: wired through to ``self.tokenizer``.
    * ``render_conversations`` / ``deal_per_text``: batching support, rendering
      one conversation or a list of them, then handing each prompt only the
      vision inputs its own markers claim.

    Subclasses implement ``call`` (the modality dispatch) and, if they carry extra
    scalar state, extend ``get_config``. The loading API + ``__call__`` -> ``call``
    forwarder are inherited from :class:`PreprocessorMixin`.
    """

    TOKENIZER_CLS = None
    IMAGE_PROCESSOR_CLS = None
    FEATURE_EXTRACTOR_CLS = None
    COMPONENTS = ("tokenizer", "image_processor", "feature_extractor")

    @classmethod
    def from_hf(cls, repo, **kwargs):
        parts = {}
        if cls.TOKENIZER_CLS is not None:
            parts["tokenizer"] = cls.TOKENIZER_CLS.from_hf(repo)
        if cls.IMAGE_PROCESSOR_CLS is not None:
            parts["image_processor"] = cls.IMAGE_PROCESSOR_CLS.from_hf(repo)
        if cls.FEATURE_EXTRACTOR_CLS is not None:
            parts["feature_extractor"] = cls.FEATURE_EXTRACTOR_CLS.from_hf(repo)
        return cls(**parts, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        """Load every component from a Hub repo id.

        Mirrors :meth:`from_hf`, but for a self-describing zeromodels repo: each
        component loads by repo id, so the tokenizer pulls the repo's
        ``tokenizer.json`` and the image processor / feature extractor its
        ``zm_preprocessor.json`` (``Processor.from_weights("zeromodels/repo")``).
        """
        parts = {}
        if cls.TOKENIZER_CLS is not None:
            parts["tokenizer"] = cls.TOKENIZER_CLS.from_weights(repo_id)
        if cls.IMAGE_PROCESSOR_CLS is not None:
            parts["image_processor"] = cls.IMAGE_PROCESSOR_CLS.from_weights(repo_id)
        if cls.FEATURE_EXTRACTOR_CLS is not None:
            parts["feature_extractor"] = cls.FEATURE_EXTRACTOR_CLS.from_weights(repo_id)
        return cls(**parts, **kwargs)

    def is_conversation_batch(self, conversation):
        """True when ``conversation`` holds several conversations, not one.

        One conversation is a list of ``{"role", "content"}`` dicts; a batch is a
        list of those lists.
        """
        return (
            isinstance(conversation, (list, tuple))
            and len(conversation) > 0
            and isinstance(conversation[0], (list, tuple))
        )

    def normalize_conversations(self, conversation):
        """One conversation or a batch of them, always as a list of conversations."""
        if self.is_conversation_batch(conversation):
            return list(conversation)
        return [conversation]

    def collect_across(self, conversations, extract):
        """Flatten what ``extract`` finds in each conversation, in marker order."""
        found = []
        for c in conversations:
            found.extend(extract(c) or [])
        return found or None

    def render_conversations(self, conversation, add_generation_prompt=True):
        """Render one conversation, or a batch of them, to prompts plus images.

        Returns ``(texts, images)``: one prompt per conversation, and every image
        across the batch flattened into a single list in marker order (``None``
        when there are no images), which is the order ``deal_per_text`` reverses.
        Processors carrying more modalities than images (video, audio) drive
        ``normalize_conversations`` + ``collect_across`` directly instead.
        """
        conversations = self.normalize_conversations(conversation)
        texts = [
            self.apply_chat_template(c, add_generation_prompt) for c in conversations
        ]
        return texts, self.collect_across(conversations, self.extract_images)

    def deal_per_text(self, texts, marker, items):
        """Hand each text the slice of ``items`` its own markers claim.

        Vision inputs arrive as one flat batch-wide list, so each prompt has to
        take its own share: without this every prompt would expand against the
        whole list and land the wrong geometry (or a mismatch) in the batch.

        Raises when the markers and the inputs do not add up, rather than dealing
        a short slice: a silent truncation would leave the extra inputs with no
        placeholders to scatter into and fail later, inside the model.
        """
        dealt = []
        start = 0
        for text in texts:
            count = text.count(marker)
            dealt.append(items[start : start + count])
            start += count
        if start != len(items):
            raise ValueError(
                f"{start} {marker} placeholder(s) across {len(texts)} prompt(s) "
                f"but {len(items)} vision input(s) were given."
            )
        return dealt

    def decode(self, *args, **kwargs) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.decode() requires `self.tokenizer` to be set."
            )
        return tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.batch_decode() requires "
                "`self.tokenizer` to be set."
            )
        return tokenizer.batch_decode(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        for attr in self.COMPONENTS:
            component = getattr(self, attr, None)
            if component is not None:
                config[attr] = keras.saving.serialize_keras_object(component)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for attr in cls.COMPONENTS:
            if isinstance(config.get(attr), dict):
                config[attr] = keras.saving.deserialize_keras_object(config[attr])
        return cls(**config)
