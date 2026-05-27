"""Runtime-mutable settings.

A small subset of operator-tunable knobs that an admin can flip from
the UI without restarting the service. Persists to
``web/data/runtime_settings.json`` so changes survive restarts.

Right now only ``auto_update_override`` lives here. The env-level
``DECKWEAVER_AUTO_UPDATE`` still acts as the default; the override
only takes effect when explicitly set (None = use the env default).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import DATA_ROOT, get_settings

_PATH = DATA_ROOT / "runtime_settings.json"
_lock = threading.Lock()
_state: dict[str, object] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    try:
        with _PATH.open("r", encoding="utf-8") as f:
            _state.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception:
        # Corrupt file shouldn't crash the service; fall back to defaults.
        pass
    _loaded = True


def _save() -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_state, f, ensure_ascii=False, indent=2)
    tmp.replace(_PATH)


def get_auto_update() -> bool:
    """Effective auto-update value: override if set, else env default."""
    with _lock:
        _load()
        ov = _state.get("auto_update_override")
        if isinstance(ov, bool):
            return ov
    return get_settings().auto_update


def set_auto_update(value: bool) -> None:
    with _lock:
        _load()
        _state["auto_update_override"] = bool(value)
        _save()
