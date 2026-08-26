"""Typed model configuration base, in the spirit of a transformers config.

A ``BaseConfig`` subclass declares each hyperparameter as an annotated class
attribute with its default. Model constructors stay **flat** (every field is a
keyword), but a config **serializes nested**, mirroring upstream configs. There are
two ways to declare the nesting:

* **Composite (transformers-style)** ``sub_configs``: the config holds real
  sub-config objects (``text_config`` / ``vision_config``), each its own
  ``BaseConfig`` with native field names. ``to_dict`` recurses into them; global
  fields (``image_token_id``) stay top-level. ``sub_config_prefixes`` says how each
  sub-config's fields flatten to the (flat) model constructor ("" for the primary
  tower, ``"vision_"`` for a vision tower); ``group_extras`` lists a sub-config's
  fields that keep their own name in the flat constructor. This is the preferred
  form for multi-tower models.

* **Prefix auto-grouping** ``config_groups``: a flat config whose fields sharing a
  prefix collapse into a nested sub-config on serialize (``"vision_"`` moves
  ``vision_embed_dim`` -> ``vision_config: {"embed_dim": ...}``). Lighter for a
  single optional group.

Either way the primary block is ``text_config`` (a config with a token
``vocab_size``) or ``vision_config`` (a pure-vision model). Parsing accepts the
nested (v2) form and the older flat (v1) form, so existing repos keep loading.
"""

import typing


def _is_tuple_type(annotation):
    """True when ``annotation`` is ``tuple`` / ``tuple[...]`` / ``tuple | None``."""
    if annotation is tuple or typing.get_origin(annotation) is tuple:
        return True
    return any(
        arg is tuple or typing.get_origin(arg) is tuple
        for arg in typing.get_args(annotation)
    )


class BaseConfig:
    """Base for typed zeromodels model configs: flat constructor, nested serialize."""

    model_type = None
    sub_configs = {}
    sub_config_prefixes = {}
    optional_sub_configs = ()
    config_groups = {}
    group_extras = {}
    top_level_fields = ()
    main_config_key = None

    def __init__(self, **kwargs):
        annotations = self._annotations()
        for name in annotations:
            value = kwargs.pop(name, getattr(type(self), name, None))
            setattr(self, name, self._coerce(name, value, annotations[name]))
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, sub_cls in self.sub_configs.items():
            value = getattr(self, key, None)
            if isinstance(value, dict):
                setattr(self, key, sub_cls(**value))
            elif value is None and key not in self.optional_sub_configs:
                setattr(self, key, sub_cls())

    @classmethod
    def _annotations(cls):
        merged = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "__annotations__", {}) or {})
        return merged

    @classmethod
    def field_names(cls):
        return list(cls._annotations().keys())

    @classmethod
    def _defaults(cls):
        return {name: getattr(cls, name, None) for name in cls._annotations()}

    @classmethod
    def _main_key(cls):
        if cls.main_config_key is not None:
            return cls.main_config_key
        if cls.sub_configs:
            for key, prefix in cls.sub_config_prefixes.items():
                if prefix == "":
                    return key
            if "text_config" in cls.sub_configs:
                return "text_config"
            return next(iter(cls.sub_configs))
        return "text_config" if "vocab_size" in cls._annotations() else "vision_config"

    @classmethod
    def _sub_flat_name(cls, key, field):
        """Flat-constructor kwarg name for ``field`` of sub-config ``key``."""
        prefix = cls.sub_config_prefixes.get(key, "")
        if not prefix or field in cls.group_extras.get(key, ()):
            return field
        return prefix + field

    @staticmethod
    def _coerce(name, value, annotation):
        if isinstance(value, list) and _is_tuple_type(annotation):
            return tuple(value)
        return value

    def constructor_kwargs(self):
        if self.sub_configs:
            flat = {
                name: getattr(self, name)
                for name in self.field_names()
                if name not in self.sub_configs
            }
            for key in self.sub_configs:
                obj = getattr(self, key)
                if obj is None:
                    continue
                for field in obj.field_names():
                    flat[self._sub_flat_name(key, field)] = getattr(obj, field)
            return flat
        return {name: getattr(self, name) for name in self.field_names()}

    def _group_members(self, key):
        """``{sub_field_name: flat_field}`` for a ``config_groups`` group."""
        prefix = self.config_groups[key]
        members = {
            f[len(prefix) :]: f for f in self.field_names() if f.startswith(prefix)
        }
        for f in self.group_extras.get(key, ()):
            members[f] = f
        return members

    def _composite_to_dict(self):
        data = {}
        if self.model_type is not None:
            data["model_type"] = self.model_type
        main = self._main_key()
        active_secondary = False
        for key, sub_cls in self.sub_configs.items():
            obj = getattr(self, key)
            if obj is None:
                continue  # absent optional tower (kept as None)
            sub = {f: getattr(obj, f) for f in obj.field_names()}
            if key == main:
                data[key] = sub
                continue
            all_default = sub == {
                f: getattr(sub_cls, f, None) for f in sub_cls.field_names()
            }
            if key in self.optional_sub_configs and all_default:
                continue  # optional tower left all-default (legacy sentinel style)
            data[key] = sub
            active_secondary = True
        if active_secondary:  # glue only present alongside a secondary tower
            for name in self.field_names():
                if name not in self.sub_configs:
                    data[name] = getattr(self, name)
        return data

    def to_dict(self):
        if self.sub_configs:
            return self._composite_to_dict()
        grouped = set()
        active = {}
        for key in self.config_groups:
            members = self._group_members(key)
            grouped.update(members.values())
            active[key] = {sub: getattr(self, f) for sub, f in members.items()}

        data = {}
        if self.model_type is not None:
            data["model_type"] = self.model_type
        main_key = self._main_key()
        main_fields = [
            f
            for f in self.field_names()
            if f not in grouped and f not in self.top_level_fields
        ]
        if main_key not in self.config_groups:
            data[main_key] = {f: getattr(self, f) for f in main_fields}
        for key, sub in active.items():
            data[key] = sub
        if active:  # glue only makes sense alongside an active group
            for f in self.top_level_fields:
                data[f] = getattr(self, f)
        return data

    @classmethod
    def _composite_from_dict(cls, data):
        if any(key in data for key in cls.sub_configs):  # nested input
            init = {key: data[key] for key in cls.sub_configs if key in data}
            for name in cls.field_names():
                if name not in cls.sub_configs and name in data:
                    init[name] = data[name]
            return cls(**init)
        init = {}
        for key, sub_cls in cls.sub_configs.items():
            sub = {}
            for field in sub_cls.field_names():
                flat_name = cls._sub_flat_name(key, field)
                if flat_name in data:
                    sub[field] = data[flat_name]
            if sub or key not in cls.optional_sub_configs:
                init[key] = sub
        for name in cls.field_names():
            if name not in cls.sub_configs and name in data:
                init[name] = data[name]
        return cls(**init)

    @classmethod
    def from_dict(cls, data):
        if cls.sub_configs:
            return cls._composite_from_dict(data)
        fields = set(cls.field_names())
        main_key = cls._main_key()
        if main_key in data or "text_config" in data or "vision_config" in data:
            flat = {k: v for k, v in data.items() if k in fields}
            if main_key not in cls.config_groups:
                main = data.get(main_key) or {}
                flat.update({k: v for k, v in main.items() if k in fields})
            for key, prefix in cls.config_groups.items():
                extras = set(cls.group_extras.get(key, ()))
                for sub, value in (data.get(key) or {}).items():
                    flat_name = sub if sub in extras else prefix + sub
                    if flat_name in fields:
                        flat[flat_name] = value
            return cls(**flat)
        return cls(**{k: v for k, v in data.items() if k in fields})

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self.constructor_kwargs().items())
        return f"{type(self).__name__}({inner})"
