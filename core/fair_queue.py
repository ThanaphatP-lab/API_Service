"""Bounded, process-local round-robin jobs; exactly one inference call at a time.

The queue owns submitted resources, including after caller timeout/cancellation.
An active native inference cannot be interrupted: cleanup waits for its return.
"""
from __future__ import annotations

import asyncio
import logging
import time
from core.telemetry import current_request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.contracts import ModelAPIError

logger = logging.getLogger("model_api")


@dataclass(eq=False)
class _Job:
    items: list[Any]
    run: Callable[[list[Any]], Any]
    cleanup: Callable[[], None]
    future: asyncio.Future
    cost_bytes: int
    label: str
    cursor: int = 0
    cancelled: bool = False
    results: list[Any] = field(default_factory=list)
    queued_at: float = field(default_factory=time.perf_counter)


class FairInferenceQueue:
    def __init__(self, *, quantum: int = 20, max_jobs: int = 16,
                 max_items: int = 256, max_bytes: int = 256 * 1024 * 1024,
                 timeout: float = 180):
        if min(quantum, max_jobs, max_items, max_bytes, timeout) <= 0:
            raise ValueError("queue limits must be positive")
        self.quantum, self.max_jobs = quantum, max_jobs
        self.max_items, self.max_bytes, self.timeout = max_items, max_bytes, timeout
        self._ready: deque[_Job] = deque()
        self._jobs: set[_Job] = set()
        self._items = self._bytes = 0
        self._wake = asyncio.Event()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rec-inference")
        self._worker = asyncio.create_task(self._work())

    async def submit(self, items: list[Any], run: Callable, *, cleanup: Callable = lambda: None,
                     cost_bytes: int = 0, label: str = "") -> list[Any]:
        # Ownership transfers on entry, including rejection.
        if not items or cost_bytes < 0:
            cleanup()
            raise ValueError("items must be nonempty and cost_bytes nonnegative")
        if (self._closed or len(self._jobs) >= self.max_jobs
                or self._items + len(items) > self.max_items
                or self._bytes + cost_bytes > self.max_bytes):
            cleanup()
            raise ModelAPIError(503, "SERVICE_BUSY", "The inference queue is full or stopping.",
                                headers={"Retry-After": "1"})
        future = asyncio.get_running_loop().create_future()
        job = _Job(list(items), run, cleanup, future, cost_bytes, label)
        self._jobs.add(job)
        self._items += len(items)
        self._bytes += cost_bytes
        self._ready.append(job)
        self._wake.set()
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            job.cancelled = True
            # Consume any completion racing with a timeout; do not leak futures.
            if future.done() and not future.cancelled():
                future.exception()
            future.cancel()
            self._wake.set()
            if isinstance(exc, asyncio.TimeoutError):
                raise ModelAPIError(504, "INFERENCE_TIMEOUT", "The queued inference exceeded its time limit.") from exc
            raise

    def _finish(self, job: _Job, error: Exception | None = None) -> None:
        self._jobs.remove(job)
        self._items -= len(job.items)
        self._bytes -= job.cost_bytes
        try:
            job.cleanup()
        except Exception:
            logger.exception("Inference resource cleanup failed request_id=%s", job.label)
        if not job.future.done():
            if error is not None:
                job.future.set_exception(error)
            else:
                job.future.set_result(job.results)

    async def _work(self) -> None:
        while True:
            await self._wake.wait()
            while self._ready:
                job = self._ready.popleft()
                if job.cancelled or self._closed:
                    self._finish(job, ModelAPIError(503, "SERVICE_BUSY", "Inference service is stopping."))
                    continue
                chunk = job.items[job.cursor:job.cursor + self.quantum]
                turn_start = time.perf_counter()
                logging.getLogger("uvicorn.error").info(
                    "fair_queue_turn request_id=%s queue_wait_ms=%.2f items=%s active_jobs=%s",
                    job.label, (turn_start-job.queued_at)*1000, len(chunk), len(self._jobs),
                )
                def run_chunk(job=job, chunk=chunk):
                    token = current_request.set(job.label)
                    try:
                        return job.run(chunk)
                    finally:
                        current_request.reset(token)
                try:
                    result = await asyncio.get_running_loop().run_in_executor(self._executor, run_chunk)
                except Exception as exc:
                    self._finish(job, exc)
                    continue
                job.cursor += len(chunk)
                logger.info("inference_turn request_id=%s items=%s completed=%s total=%s",
                            job.label, len(chunk), job.cursor, len(job.items))
                if job.cancelled or self._closed:
                    self._finish(job, ModelAPIError(503, "SERVICE_BUSY", "Inference service is stopping."))
                else:
                    job.results.append(result)
                    if job.cursor == len(job.items):
                        self._finish(job)
                    else:
                        job.queued_at = time.perf_counter()
                        self._ready.append(job)
                # Admit arrivals and deliver finished responses before the next turn.
                await asyncio.sleep(0)
            self._wake.clear()
            if self._closed:
                return

    async def aclose(self) -> None:
        self._closed = True
        self._wake.set()
        await self._worker
        self._executor.shutdown(wait=True)
