"""ETA estimator — rolling average of seconds-per-page by mode.

History is recomputed on demand from the DB so it survives restarts
without a separate persistence layer.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from sqlalchemy.orm import Session

from .models import Job

# Conservative defaults for cold start (no history yet). These are
# rough order-of-magnitude — refined as soon as one real job lands.
DEFAULT_SECONDS_PER_PAGE = {"full": 60, "text-only": 25}
HISTORY_WINDOW = 20


def _seconds_per_page(jobs: Iterable[Job]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for j in jobs:
        if j.status != "done" or j.page_count <= 0 or j.duration_seconds <= 0:
            continue
        buckets[j.mode].append(j.duration_seconds / j.page_count)
    out: dict[str, float] = {}
    for mode, samples in buckets.items():
        recent = samples[-HISTORY_WINDOW:]
        out[mode] = sum(recent) / len(recent)
    return out


def per_page_seconds(db: Session, mode: str) -> float:
    history = _seconds_per_page(db.query(Job).filter(Job.status == "done").all())
    return history.get(mode) or DEFAULT_SECONDS_PER_PAGE.get(mode, 60)


def estimate_job(db: Session, job: Job, all_jobs: list[Job]) -> tuple[int, int]:
    """Return (queue_position, eta_seconds) for the given job.

    queue_position counts jobs ahead in queued/running state. ETA is the
    expected time until this job FINISHES (not until it starts).
    """
    sec_per_page = per_page_seconds(db, job.mode)

    if job.status in ("done", "failed", "canceled"):
        return 0, 0

    ahead = [
        j for j in all_jobs
        if j.status in ("running", "queued")
        and (j.created_at < job.created_at or (j.status == "running" and j.id != job.id))
        and j.id != job.id
    ]
    queue_position = sum(1 for _ in ahead)

    ahead_seconds = 0.0
    for j in ahead:
        remaining_pages = max(j.page_count - j.current_page, 1)
        ahead_seconds += remaining_pages * per_page_seconds(db, j.mode)

    own_seconds = max(job.page_count - job.current_page, 1) * sec_per_page
    return queue_position, int(ahead_seconds + own_seconds)
