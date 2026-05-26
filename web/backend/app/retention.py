"""Background sweeper that deletes finished jobs older than N days."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from .config import get_settings
from .db import SessionLocal
from .models import Job
from . import queue as job_queue

log = logging.getLogger("deckweaver.retention")


def _sweep_once() -> int:
    s = get_settings()
    if s.job_retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=s.job_retention_days)
    deleted = 0
    db = SessionLocal()
    try:
        stale = (
            db.query(Job)
            .filter(Job.status.in_(("done", "failed", "canceled")))
            .filter(Job.finished_at.is_not(None))
            .filter(Job.finished_at < cutoff)
            .all()
        )
        for j in stale:
            job_queue.cleanup_job_dirs(j)
            db.delete(j)
            deleted += 1
        db.commit()
    finally:
        db.close()
    return deleted


async def sweeper_loop() -> None:
    s = get_settings()
    while True:
        try:
            n = await asyncio.to_thread(_sweep_once)
            if n:
                log.info("retention: deleted %d stale job(s)", n)
        except Exception as exc:  # pragma: no cover
            log.warning("retention sweep failed: %s", exc)
        await asyncio.sleep(max(60, s.retention_sweep_seconds))
