from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any
import time
import uuid
from contextlib import contextmanager
from collections.abc import Iterator


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str

    start_time_ms: float
    end_time_ms: float | None = None

    attributes: dict[str, Any] = field(default_factory=dict)

    status: str = "OK"

    @property
    def latency_ms(self) -> float | None:
        if self.end_time_ms is None:
            return None

        return self.end_time_ms - self.start_time_ms


class Tracer:

    def __init__(self, trace: Trace):
        self.trace = trace

    @contextmanager
    def start_span(
        self,
        name: str,
    ) -> Iterator[Span]:

        parent_context = get_trace_context()

        span_id = str(uuid.uuid4())

        span = Span(
            trace_id=self.trace.trace_id,
            span_id=span_id,
            parent_span_id=(
                parent_context.span_id
                if parent_context is not None
                else None
            ),
            name=name,
            start_time_ms=time.perf_counter() * 1000,
        )

        token = set_trace_context(
            TraceContext(
                trace_id=self.trace.trace_id,
                span_id=span_id,
            )
        )

        try:
            yield span

        except Exception as exc:
            span.status = "ERROR"
            span.attributes["error.type"] = type(exc).__name__
            raise

        finally:
            span.end_time_ms = time.perf_counter() * 1000

            reset_trace_context(token)

            self.trace.add_span(span)


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def start_span(
        self,
        name: str,
    ) -> Iterator[Span]:

        parent_context = get_trace_context()

        span_id = str(uuid.uuid4())

        span = Span(
            trace_id=self.trace.trace_id,
            span_id=span_id,
            parent_span_id=(
                parent_context.span_id
                if parent_context is not None
                else None
            ),
            name=name,
            start_time_ms=time.perf_counter() * 1000,
        )

        token = set_trace_context(
            TraceContext(
                trace_id=self.trace.trace_id,
                span_id=span_id,
            )
        )

        try:
            yield span

        except Exception as exc:
            span.status = "ERROR"
            span.attributes["error.type"] = type(exc).__name__
            raise

        finally:
            span.end_time_ms = time.perf_counter() * 1000

            reset_trace_context(token)

            self.trace.add_span(span)

    def add_span(self, span: Span) -> None:
        if span.trace_id != self.trace_id:
            raise ValueError(
                "Span trace_id must match Trace trace_id."
            )

        self.spans.append(span)


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str | None = None


_current_trace_context: ContextVar[TraceContext | None] = ContextVar(
    "current_trace_context",
    default=None,
)


def get_trace_context() -> TraceContext | None:
    return _current_trace_context.get()


def set_trace_context(
    context: TraceContext,
):
    return _current_trace_context.set(context)


def reset_trace_context(token) -> None:
    _current_trace_context.reset(token)
