import asyncio
import threading

import pytest

from core.fair_queue import FairInferenceQueue
from shared.contracts import ModelAPIError


def test_round_robin_100_and_60_items_short_job_finishes_first():
    async def scenario():
        queue = FairInferenceQueue(quantum=20)
        turns, completed, cleaned = [], [], []
        def run(items):
            turns.append(items[0][0])
            return items
        async def job(name, count):
            result = await queue.submit([(name, i) for i in range(count)], run,
                                        cleanup=lambda: cleaned.append(name))
            completed.append(name)
            return [item for chunk in result for item in chunk]
        try:
            a, b = await asyncio.gather(job("A", 100), job("B", 60))
            assert turns == ["A", "B", "A", "B", "A", "B", "A", "A"]
            assert completed == ["B", "A"]
            assert cleaned == ["B", "A"]
            assert a == [("A", i) for i in range(100)]
            assert b == [("B", i) for i in range(60)]
        finally:
            await queue.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["jobs", "items", "bytes"])
def test_backpressure_includes_active_job_and_cleans_rejected(limit):
    async def scenario():
        queue = FairInferenceQueue(max_jobs=1 if limit == "jobs" else 5,
                                   max_items=1 if limit == "items" else 5,
                                   max_bytes=1 if limit == "bytes" else 100)
        started, release = threading.Event(), threading.Event()
        cleaned = []
        def run(items):
            started.set()
            assert release.wait(3)
            return items
        active = asyncio.create_task(queue.submit([1], run, cost_bytes=1))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            with pytest.raises(ModelAPIError) as caught:
                await queue.submit([2], run, cost_bytes=1, cleanup=lambda: cleaned.append(2))
            assert caught.value.code == "SERVICE_BUSY"
            assert cleaned == [2]
        finally:
            release.set()
            await active
            await queue.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [False, True])
def test_cancellation_waits_for_active_chunk_before_cleanup(timeout):
    async def scenario():
        queue = FairInferenceQueue(quantum=1, timeout=0.05 if timeout else 3)
        started, release = threading.Event(), threading.Event()
        cleaned, calls = [], []
        def run(items):
            calls.extend(items)
            started.set()
            assert release.wait(3)
            assert not cleaned
            return items
        job = asyncio.create_task(queue.submit([1, 2], run, cleanup=lambda: cleaned.append(True)))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            if timeout:
                with pytest.raises(ModelAPIError) as caught:
                    await job
                assert caught.value.code == "INFERENCE_TIMEOUT"
            else:
                job.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await job
            assert cleaned == []
        finally:
            release.set()
            await queue.aclose()
        assert calls == [1]
        assert cleaned == [True]
    asyncio.run(scenario())


def test_error_does_not_stop_other_jobs_or_event_loop():
    async def scenario():
        queue = FairInferenceQueue(quantum=1)
        threads, cleaned = [], []
        main_thread = threading.get_ident()
        def fail(items):
            threads.append(threading.get_ident())
            raise RuntimeError("fake GPU failure")
        def good(items):
            threads.append(threading.get_ident())
            return items
        try:
            results = await asyncio.gather(
                queue.submit([1], fail, cleanup=lambda: cleaned.append(1)),
                queue.submit([2, 3], good, cleanup=lambda: cleaned.append(2)),
                return_exceptions=True,
            )
            assert isinstance(results[0], RuntimeError)
            assert results[1] == [[2], [3]]
            assert len(set(threads)) == 1 and threads[0] != main_thread
            assert cleaned == [1, 2]
        finally:
            await queue.aclose()
    asyncio.run(scenario())


def test_shutdown_finishes_active_chunk_and_rejects_remaining_work():
    async def scenario():
        queue = FairInferenceQueue(quantum=1)
        started, release = threading.Event(), threading.Event()
        cleaned, calls = [], []
        def run(items):
            calls.extend(items)
            started.set()
            assert release.wait(3)
            return items
        first = asyncio.create_task(queue.submit([1, 2], run, cleanup=lambda: cleaned.append(1)))
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(queue.submit([3], run, cleanup=lambda: cleaned.append(3)))
        await asyncio.sleep(0)
        closing = asyncio.create_task(queue.aclose())
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        await closing
        assert all(isinstance(result, ModelAPIError) for result in results)
        assert calls == [1]
        assert sorted(cleaned) == [1, 3]
        with pytest.raises(ModelAPIError):
            await queue.submit([4], run)
    asyncio.run(scenario())
