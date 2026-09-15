import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


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
        logger.info("Transcription worker started")

    async def stop(self):
        self._running = False
        if self._worker:
            self._worker.cancel()
            logger.info("Transcription worker stopped")
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
            logger.warning(
                "Queue full (max %d), rejecting job for %s",
                self._queue.maxsize,
                tmp_path,
            )
            return {
                "status": "queue_full",
                "detail": f"Queue is full (max {self._queue.maxsize})",
            }
        position = self._queue.qsize()
        logger.info("Job submitted for %s (position %d)", tmp_path, position)
        await job.event.wait()

        if job.error:
            logger.error("Job failed for %s: %s", tmp_path, job.error)
            return {"status": "error", "detail": job.error}
        logger.info("Job completed for %s", tmp_path)
        return {"status": "completed", "result": job.result, "queue_position": position}

    async def _worker_loop(self):
        logger.debug("Worker loop running")
        while self._running:
            try:
                tmp_path, job = await self._queue.get()
                logger.info("Processing %s", tmp_path)
                try:
                    result = await run_in_threadpool(self.pipeline, tmp_path)
                    job.result = result
                except Exception as exc:
                    job.error = str(exc)
                    logger.error(
                        "Error processing %s: %s", tmp_path, exc, exc_info=True
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        logger.warning("Failed to clean up %s", tmp_path)
                    job.event.set()
                    self._queue.task_done()
            except asyncio.CancelledError:
                logger.debug("Worker loop cancelled")
                break
            except Exception:
                await asyncio.sleep(0.1)
