import pytest

from app.observability.tracing import Span, Tracer, Trace, TraceContext, get_trace_context, set_trace_context, reset_trace_context


def make_span(trace_id="A", start_time_ms=100, end_time_ms=None):
    return Span(
        trace_id=trace_id,
        span_id="span-1",
        parent_span_id=None,
        name="test-span",
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )


def test_span_latency_is_end_minus_start():
    span = make_span(start_time_ms=100, end_time_ms=175)

    assert span.latency_ms == 75


def test_open_span_has_no_latency():
    span = make_span(start_time_ms=100, end_time_ms=None)

    assert span.latency_ms is None


def test_trace_rejects_span_from_other_trace():
    trace = Trace(trace_id="A")
    span = make_span(trace_id="B")

    with pytest.raises(ValueError):
        trace.add_span(span)

    assert trace.spans == []


def test_trace_context_can_be_set_and_restored():

    original = get_trace_context()

    context = TraceContext(
        trace_id="trace-123",
        span_id="span-1",
    )

    token = set_trace_context(context)

    try:
        assert get_trace_context() == context
    finally:
        reset_trace_context(token)

    assert get_trace_context() == original


def test_start_span_records_completed_span():

    trace = Trace(trace_id="trace-123")
    tracer = Tracer(trace)

    with tracer.start_span("retrieval.search") as span:
        span.attributes["returned_count"] = 5

    assert len(trace.spans) == 1

    recorded = trace.spans[0]

    assert recorded.name == "retrieval.search"
    assert recorded.status == "OK"
    assert recorded.attributes["returned_count"] == 5
    assert recorded.latency_ms is not None
    assert recorded.latency_ms >= 0


def test_nested_spans_create_parent_child_relationship():

    trace = Trace(trace_id="trace-123")
    tracer = Tracer(trace)

    with tracer.start_span("agent.run") as parent:

        with tracer.start_span("retrieval.search") as child:
            pass

    assert child.parent_span_id == parent.span_id


def test_span_records_error_and_reraises_exception():

    trace = Trace(trace_id="trace-123")
    tracer = Tracer(trace)

    with pytest.raises(RuntimeError):

        with tracer.start_span("model.generate"):
            raise RuntimeError("provider failed")

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.status == "ERROR"
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.latency_ms is not None
