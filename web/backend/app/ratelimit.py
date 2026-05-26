"""Tiny in-memory token bucket. Sized for single-worker uvicorn.

If you scale to multiple workers, replace this with Redis. We keep it
in-process here because the worker queue is single-process anyway.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class _Window:
    times: deque[float]


class RateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self.max = max_per_minute
        self._windows: dict[str, _Window] = defaultdict(lambda: _Window(deque()))

    def allow(self, key: str) -> bool:
        if self.max <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        w = self._windows[key]
        while w.times and w.times[0] < cutoff:
            w.times.popleft()
        if len(w.times) >= self.max:
            return False
        w.times.append(now)
        return True

    def retry_after(self, key: str) -> int:
        if self.max <= 0:
            return 0
        w = self._windows.get(key)
        if not w or not w.times:
            return 0
        return max(1, int(60 - (time.monotonic() - w.times[0])))
