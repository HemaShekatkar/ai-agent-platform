import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class OverloadedError(Exception):
    pass


class ConcurrencyLimiter:
    def __init__(
        self,
        max_concurrent: int,
        max_waiting: int,
    ):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")

        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Tracks requests waiting for a concurrency slot.
        self._waiting_slots = asyncio.Semaphore(max_waiting)

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        queue_timeout_seconds: float = 1.0,
    ) -> T:

        if queue_timeout_seconds <= 0:
            raise ValueError("queue timeout must be positive")

        # Fast path: acquire immediately if capacity exists.
        # asyncio.Semaphore has no public nonblocking acquire,
        # so we use a bounded wait instead.
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=0.001,
            )
        except TimeoutError:

            # No execution capacity. Try entering the wait queue.
            try:
                await asyncio.wait_for(
                    self._waiting_slots.acquire(),
                    timeout=0.001,
                )
            except TimeoutError:
                raise OverloadedError(
                    "Concurrency limit and wait queue are full"
                )

            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=queue_timeout_seconds,
                )
            except TimeoutError:
                raise OverloadedError(
                    "Timed out waiting for execution capacity"
                )
            finally:
                self._waiting_slots.release()

        try:
            return await operation()
        finally:
            self._semaphore.release()
