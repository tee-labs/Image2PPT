"""Subprocess wrapper around scripts/convert.py.

Streams stdout, parses progress markers, calls back into the queue.
The subprocess is launched with CWD = repo root so its relative paths
behave identically to a manual CLI invocation, and run through the
sandbox module so it can only write to its own job dirs.
"""
from __future__ import annotations

import asyncio
import re
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

from .config import REPO_ROOT, get_settings
from . import sandbox

STAGE_RE = re.compile(r"^===\s*(\d+)/(\d+)\s+")
PAGE_RE = re.compile(r"^page\s+(\d+):")
LOG_TAIL_LINES = 200


async def run_convert(
    *,
    source: Path,
    work_dir: Path,
    upload_dir: Path,
    mode: str,
    on_stage: Callable[[int, int], Awaitable[None]],
    on_page: Callable[[int], Awaitable[None]],
    on_line: Callable[[str], Awaitable[None]],
) -> tuple[int, str]:
    """Run convert.py. Returns (exit_code, full_tail_log)."""
    s = get_settings()
    cmd = [
        s.python_bin,
        str(s.convert_script),
        "--source", str(source),
        "--work-dir", str(work_dir),
        # EasyOCR + Tesseract are optional cross-verifiers; skipping
        # them keeps the prod install (and the sandbox) lean. Set
        # DECKWEAVER_CROSS_VERIFY=true if you've installed them and
        # want belt-and-suspenders OCR confidence.
    ]
    if not s.cross_verify:
        cmd.append("--skip-cross-verify")
    if mode != "full":
        cmd += ["--mode", mode]

    wrapped, cleanup = sandbox.wrap_command(
        cmd, upload_dir=upload_dir, output_dir=work_dir,
    )
    env = sandbox.safe_env()
    preexec = sandbox.make_preexec(
        memory_mb=s.subprocess_memory_mb,
        cpu_seconds=s.subprocess_cpu_seconds,
        output_mb=s.subprocess_output_mb,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *wrapped,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            preexec_fn=preexec,
            start_new_session=False,  # preexec already does setsid
        )

        tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            tail.append(line)
            await on_line(line)
            m = STAGE_RE.match(line)
            if m:
                await on_stage(int(m.group(1)), int(m.group(2)))
                continue
            m = PAGE_RE.match(line)
            if m:
                await on_page(int(m.group(1)))

        code = await proc.wait()
        return code, "\n".join(tail)
    finally:
        for p in cleanup:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
