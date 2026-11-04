"""Portable YAML configuration with inheritance and environment variables."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

import yaml


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read(path: Path, stack: tuple[Path, ...]) -> Dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    base = config.pop("_base", None)
    if base:
        config = _deep_merge(_read(path.parent / base, (*stack, path)), config)
    return config


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.groups()
        if os.environ.get(name):
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Set the {name} environment variable before loading this config")

    return os.path.expanduser(_ENV.sub(replace, value))


def load_config(path: str) -> Dict:
    """Merge relative ``_base`` files, then expand ``${VAR:-default}`` strings."""
    return _expand(_read(Path(path), ()))
