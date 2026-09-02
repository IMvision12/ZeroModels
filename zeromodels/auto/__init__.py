"""Auto loaders: resolve a repo to its class by ``model_type`` (transformers-style).

See :mod:`zeromodels.auto.auto_factory`. The classes are named ``AutoZM*`` (``AutoZModel``
for the backbone) so they never collide with the transformers ``AutoModel`` /
``AutoTokenizer`` / ... when both libraries are imported. The ``AutoZM<Task>`` model
classes (``AutoZModel``, ``AutoZMDetect``, ``AutoZMImageClassify``, ``AutoZMTextGenerate``,
...) plus ``AutoZMConfig`` / ``AutoZMTokenizer`` / ``AutoZMProcessor`` /
``AutoZMImageProcessor`` are generated from :mod:`zeromodels.models`, so this package
re-exports whatever the factory built.
"""

from zeromodels.auto import auto_factory
from zeromodels.auto.auto_factory import (  # noqa: F401
    AutoZMConfig,
    AutoZMImageProcessor,
    AutoZMProcessor,
    AutoZMTokenizer,
    all_mappings,
    read_model_type,
)

# Pull in the dynamically-created AutoZM<Task> model classes (AutoZModel, AutoZMDetect, ...).
globals().update(auto_factory._TASK_AUTOS)

__all__ = list(auto_factory.__all__)
