"""
config/settings.py
------------------
Centralised settings loader.

Loads config/config.yaml, then overrides any leaf value with matching
environment variables (e.g. CRAWLER__MAX_WORKERS=8 overrides
crawler.max_workers).  All downstream modules import `settings` from here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(cfg: dict, prefix: str = "") -> dict:
    """
    Walk cfg and look for matching env vars.
    Nested keys are addressed with double underscore, e.g.:
        CRAWLER__MAX_WORKERS  -> cfg["crawler"]["max_workers"]
    """
    for key, value in cfg.items():
        env_key = (f"{prefix}__{key}" if prefix else key).upper()
        if isinstance(value, dict):
            cfg[key] = _apply_env_overrides(value, prefix=env_key.lower().rstrip("_"))
        else:
            env_val = os.getenv(env_key)
            if env_val is not None:
                # attempt type coercion to match original type
                if isinstance(value, bool):
                    cfg[key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(value, int):
                    cfg[key] = int(env_val)
                elif isinstance(value, float):
                    cfg[key] = float(env_val)
                else:
                    cfg[key] = env_val
    return cfg


class _Settings:
    """Dot-access wrapper around the merged configuration dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _Settings(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Settings({list(self._data.keys())})"


def load_settings(extra_config: dict | None = None) -> _Settings:
    """Load, merge, and return settings."""
    base = _load_yaml(_CONFIG_PATH)
    if extra_config:
        base = _deep_merge(base, extra_config)
    base = _apply_env_overrides(base)
    return _Settings(base)


# Module-level singleton – import this everywhere:
#   from config.settings import settings
settings = load_settings()
