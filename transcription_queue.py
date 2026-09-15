import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi.concurrency import run_in_threadpool


@dataclass
class TranscriptionJob:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any | None = None
    error: str | None = None


class TranscriptionQueue:
    """Serializes transcription requests so only one runs at a time."""

    def __init__(self, max_queue_size: int = 10):
        self.pipeline = None
        self._queue: asyncio.Queue[tuple[str, TranscriptionJob]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker: asyncio.Task | None = None
        self._running = False

    async def start(self):
        self._running = True
        self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._running = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def submit(self, tmp_path: str) -> dict:
        """Submit a job and wait for it to complete."""
        job = TranscriptionJob()
        try:
            self._queue.put_nowait((tmp_path, job))
        except asyncio.QueueFull:
            return {
                "status": "queue_full",
                "detail": f"Queue is full (max {self._queue.maxsize})",
            }
        position = self._queue.qsize()
        await job.event.wait()

        if job.error:
            return {"status": "error", "detail": job.error}
        return {"status": "completed", "result": job.result, "queue_position": position}

    async def _worker_loop(self):
        while self._running:
            try:
                tmp_path, job = await self._queue.get()
                try:
                    result = await run_in_threadpool(self.pipeline, tmp_path)
                    job.result = result
                except Exception as exc:
                    job.error = str(exc)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    job.event.set()
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)
