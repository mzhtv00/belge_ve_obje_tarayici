import asyncio
import logging
from typing import Any

from fastapi import WebSocketDisconnect

from app.storage import CaptureStore

logger = logging.getLogger(__name__)


class EventHub:
    def __init__(self) -> None:
        self._clients: dict[str, set[Any]] = {}

    async def connect(self, session_id: str, websocket: Any) -> None:
        await websocket.accept()
        self._clients.setdefault(session_id, set()).add(websocket)

    def disconnect(self, session_id: str, websocket: Any) -> None:
        clients = self._clients.get(session_id)
        if clients is None:
            return
        clients.discard(websocket)
        if not clients:
            self._clients.pop(session_id, None)

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        for websocket in tuple(self._clients.get(session_id, ())):
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(session_id, websocket)


class CapturePipeline:
    def __init__(self, store: CaptureStore, hub: EventHub, worker_count: int) -> None:
        self.store = store
        self.hub = hub
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.worker_count = worker_count
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._workers = [
            asyncio.create_task(self._worker(), name=f"capture-worker-{index}")
            for index in range(self.worker_count)
        ]

    async def stop(self) -> None:
        for _ in self._workers:
            await self.queue.put(("", ""))
        if self._workers:
            await asyncio.gather(*self._workers)
        self._workers.clear()

    async def enqueue(self, session_id: str, capture_id: str) -> None:
        await self.queue.put((session_id, capture_id))

    async def _worker(self) -> None:
        while True:
            session_id, capture_id = await self.queue.get()
            try:
                if not capture_id:
                    return
                record = await asyncio.to_thread(self.store.create_thumbnail, capture_id)
                if record is None:
                    continue
                await self.hub.publish(session_id, {
                    "type": "capture.processed",
                    "capture": record,
                    "message": "Image stored and thumbnail generated. No OCR or 3D reconstruction service is configured.",
                })
            except Exception:
                logger.exception("Capture background processing failed for %s", capture_id)
                record = self.store.update_record(capture_id, status="error")
                await self.hub.publish(session_id, {
                    "type": "capture.error",
                    "capture": record or {"id": capture_id},
                    "message": "The capture was saved, but thumbnail processing failed.",
                })
            finally:
                self.queue.task_done()
