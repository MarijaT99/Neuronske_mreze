"""YAML config loading with light schema validation and dotted access.

Every experiment is driven by a YAML file (no hardcoded hyperparameters). A
loaded config is a plain nested dict wrapped in :class:`Config`, which supports
attribute access (``cfg.training.lr``) and ``cfg.get("training.lr", default)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Read-only-ish wrapper over a nested config dict with dotted access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ---- access -------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"No config key {name!r}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a possibly-nested key like ``"training.optimizer.lr"``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    # ---- conversion ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _deep_copy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return Config(data)


def save_config(cfg: Config | dict[str, Any], path: str | Path) -> None:
    """Persist a config (snapshot it next to every experiment's checkpoints)."""
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
