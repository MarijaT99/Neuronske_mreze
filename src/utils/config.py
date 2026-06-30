"""Učitavanje YAML config-a uz laku validaciju šeme i pristup tačka-notacijom.

Svaki eksperiment je vođen YAML fajlom (bez zakucanih hiperparametara).
Učitani config je običan ugnežđeni dict obavijen u :class:`Config`, koji podržava
pristup preko atributa (``cfg.training.lr``) i ``cfg.get("training.lr", default)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Uglavnom read-only omotač nad ugnežđenim config dict-om sa pristupom tačka-notacijom."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ---- pristup ------------------------------------------------------------
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
        """Dohvati potencijalno ugnežđeni ključ poput ``"training.optimizer.lr"``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    # ---- konverzija ---------------------------------------------------------
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Rekurzivno spoji ``override`` u ``base`` (override pobeđuje na listovima)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_raw(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_config(path: str | Path) -> Config:
    """Učitaj YAML config, poštujući opcioni ``extends:`` bazni fajl.

    Config može deklarisati ``extends: <relativna-putanja>`` (razrešena u odnosu
    na sopstveni direktorijum config-a). Baza se prvo učitava (rekurzivno), a
    trenutni fajl se duboko spaja preko nje, tako da config-i eksperimenata
    navode samo override vrednosti. Ključ ``extends`` se uklanja iz finalnog config-a.
    """
    path = Path(path)
    data = _load_raw(path)

    base_ref = data.pop("extends", None)
    if base_ref is not None:
        base_path = (path.parent / base_ref).resolve()
        base_data = load_config(base_path).to_dict()
        data = _deep_merge(base_data, data)

    return Config(data)


def save_config(cfg: Config | dict[str, Any], path: str | Path) -> None:
    """Sačuvaj config (snimi ga uz checkpoint-e svakog eksperimenta)."""
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
