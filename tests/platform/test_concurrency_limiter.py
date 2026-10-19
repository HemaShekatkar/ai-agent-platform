import asyncio

import pytest

from app.platform.concurrency_limiter import (
    ConcurrencyLimiter,
    OverloadedError,
)


@pytest.mark.asyncio
async def test_limits_concurrent_operations():

    limiter = ConcurrencyLimiter(
        max_concurrent=2,
        max_waiting=5,
    )

    active = 0
    peak = 0

    async def operation():
        nonlocal active, peak

        active += 1
        peak = max(peak, active)

        await asyncio.sleep(0.02)

        active -= 1
        return "done"

    results = await asyncio.gather(
        *[
            limiter.execute(
                operation,
                queue_timeout_seconds=1.0,
            )
            for _ in range(5)
        ]
    )

    assert len(results) == 5
    assert peak <= 2


@pytest.mark.asyncio
async def test_rejects_when_queue_is_full():

    limiter = ConcurrencyLimiter(
        max_concurrent=1,
        max_waiting=0,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_operation():
        started.set()
        await release.wait()
        return "done"

    first = asyncio.create_task(
        limiter.execute(slow_operation)
    )

    await started.wait()

    with pytest.raises(OverloadedError):
        await limiter.execute(slow_operation)

    release.set()
    assert await first == "done"


@pytest.mark.asyncio
async def test_queue_timeout():

    limiter = ConcurrencyLimiter(
        max_concurrent=1,
        max_waiting=1,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_operation():
        started.set()
        await release.wait()

    first = asyncio.create_task(
        limiter.execute(slow_operation)
    )

    await started.wait()

    with pytest.raises(OverloadedError):
        await limiter.execute(
            slow_operation,
            queue_timeout_seconds=0.01,
        )

    release.set()
    await first


@pytest.mark.asyncio
async def test_operation_exception_releases_capacity():

    limiter = ConcurrencyLimiter(
        max_concurrent=1,
        max_waiting=0,
    )

    async def failing_operation():
        raise ValueError("Provider failure")

    with pytest.raises(ValueError):
        await limiter.execute(failing_operation)

    async def successful_operation():
        return "recovered"

    assert (
        await limiter.execute(successful_operation)
        == "recovered"
    )


def test_invalid_configuration():

    with pytest.raises(ValueError):
        ConcurrencyLimiter(
            max_concurrent=0,
            max_waiting=1,
        )

    with pytest.raises(ValueError):
        ConcurrencyLimiter(
            max_concurrent=1,
            max_waiting=-1,
        )
