"""Periodic GitHub poller — fetch, and if behind, pull + restart.

Restart strategy: `os.execv` the current Python interpreter with the
same argv. This re-runs uvicorn from scratch, picking up any code
changes. Jobs that were still running at the time get marked failed
on the next boot (queue.boot_recover).

Toggle with DECKWEAVER_AUTO_UPDATE=false to disable.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT, get_settings
from . import queue as job_queue
from . import runtime_settings
from .ws import broker


_state = {
    "commit": "",
    "short_commit": "",
    "behind": 0,
    "ahead": 0,
    "branch": "",
    "updating": False,
    "last_check": None,
}


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True).strip()


def _safe_run(cmd: list[str]) -> str:
    try:
        return _run(cmd)
    except Exception as exc:
        return f"ERR: {exc}"


def snapshot() -> dict:
    """Return a copy of the current sync state (also refreshes commit)."""
    try:
        _state["commit"] = _run(["git", "rev-parse", "HEAD"])
        _state["short_commit"] = _state["commit"][:7]
        _state["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    return dict(_state)


async def _refresh_counts() -> None:
    s = get_settings()
    remote_ref = f"{s.git_remote}/{s.git_branch}"
    try:
        await asyncio.to_thread(_run, ["git", "fetch", s.git_remote, s.git_branch])
        behind = await asyncio.to_thread(
            _run, ["git", "rev-list", "--count", f"HEAD..{remote_ref}"]
        )
        ahead = await asyncio.to_thread(
            _run, ["git", "rev-list", "--count", f"{remote_ref}..HEAD"]
        )
        _state["behind"] = int(behind)
        _state["ahead"] = int(ahead)
        _state["last_check"] = datetime.now(timezone.utc)
    except Exception as exc:
        _state["last_check"] = datetime.now(timezone.utc)
        _state["error"] = str(exc)


def _wait_queue_drained() -> None:
    # Synchronous polling — runs in to_thread.
    import time as _t
    while job_queue.queue_size() > 0:
        _t.sleep(1)


async def _broadcast_state() -> None:
    snap = snapshot()
    await broker.broadcast(
        {
            "type": "system",
            "commit": snap["commit"],
            "short_commit": snap["short_commit"],
            "behind": snap["behind"],
            "ahead": snap["ahead"],
            "updating": snap["updating"],
        }
    )


async def _try_update() -> bool:
    s = get_settings()
    if _state["behind"] <= 0:
        return False
    _state["updating"] = True
    job_queue.pause()
    await _broadcast_state()
    try:
        await asyncio.to_thread(_wait_queue_drained)
        # Capture requirements hashes so we know whether to pip install.
        def _sha(p: Path) -> str:
            try:
                return _safe_run(["git", "hash-object", str(p)])
            except Exception:
                return ""
        req_before = _sha(REPO_ROOT / "requirements.txt")
        web_req_before = _sha(REPO_ROOT / "web" / "backend" / "requirements.txt")

        await asyncio.to_thread(_run, ["git", "pull", "--ff-only", s.git_remote, s.git_branch])

        req_after = _sha(REPO_ROOT / "requirements.txt")
        web_req_after = _sha(REPO_ROOT / "web" / "backend" / "requirements.txt")
        if req_before != req_after:
            await asyncio.to_thread(
                subprocess.run,
                [s.python_bin, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")],
                False,
            )
        if web_req_before != web_req_after:
            await asyncio.to_thread(
                subprocess.run,
                [s.python_bin, "-m", "pip", "install", "-r",
                 str(REPO_ROOT / "web" / "backend" / "requirements.txt")],
                False,
            )
        await _broadcast_state()
        # Re-exec the current process with same argv. uvicorn will rebind
        # the listening socket from scratch.
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, *sys.argv])
        return True
    except Exception as exc:
        _state["error"] = str(exc)
        _state["updating"] = False
        job_queue.resume()
        await _broadcast_state()
        return False


async def poll_loop() -> None:
    s = get_settings()
    snapshot()
    while True:
        try:
            await _refresh_counts()
            await _broadcast_state()
            # Read the runtime override each iteration so admin toggles
            # take effect on the next poll without a restart.
            if runtime_settings.get_auto_update():
                await _try_update()
        except Exception:
            pass
        await asyncio.sleep(s.update_poll_seconds)


def is_updating() -> bool:
    return bool(_state.get("updating"))
