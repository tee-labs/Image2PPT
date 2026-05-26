"""WebSocket broker — push job state changes to connected clients."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import WebSocket


@dataclass(eq=False)
class _Client:
    """`eq=False` keeps the default identity hash so instances are
    hashable and addable to a set."""
    ws: WebSocket
    user_id: int
    is_admin: bool


class WSBroker:
    def __init__(self) -> None:
        self._clients: set[_Client] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, user_id: int, is_admin: bool) -> _Client:
        await ws.accept()
        client = _Client(ws=ws, user_id=user_id, is_admin=is_admin)
        async with self._lock:
            self._clients.add(client)
        return client

    async def disconnect(self, client: _Client) -> None:
        async with self._lock:
            self._clients.discard(client)

    async def broadcast(self, payload: dict, owner_id: int | None = None) -> None:
        dead: list[_Client] = []
        async with self._lock:
            targets = list(self._clients)
        for c in targets:
            if owner_id is not None and not c.is_admin and c.user_id != owner_id:
                continue
            try:
                await c.ws.send_json(payload)
            except Exception:
                dead.append(c)
        if dead:
            async with self._lock:
                for c in dead:
                    self._clients.discard(c)


broker = WSBroker()
