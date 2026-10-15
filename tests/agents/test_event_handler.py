import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agents.agent_lock import InMemoryAgentRunLock
from app.agents.event_handler import AgentWorkflowEventHandler
from app.agents.event_inbox import EventProcessingStatus, InMemoryEventInbox
from app.agents.events import WorkflowCompletionEvent, WorkflowEventType
from app.agents.onboarding_graph import build_onboarding_graph
from app.agents.temporal_client import ToolWorkflowRunner
from app.tools.base import BaseTool
from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.registry import ToolRegistry
from app.workflows.models import ToolWorkflowResult, ToolWorkflowStatus


AGENT_RUN_ID = "agent-run-123"


class PartnerInput(BaseModel):
    partner_id: str


class ApprovePartnerTool(BaseTool[PartnerInput, str]):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="approve_partner",
            description="Approve a partner for onboarding.",
            input_schema=PartnerInput,
            risk=ToolRisk.HIGH,
            required_permissions={"partner.approve"},
            approval_requirement=ApprovalRequirement.HUMAN,
        )

    def execute(self, input: PartnerInput, runtime: ToolRuntimeContext) -> str:
        raise AssertionError("tools run inside the workflow, never the graph")


class FakeTemporalRunner(ToolWorkflowRunner):
    # Pending until a test sets complete = True, e.g. once approval happens

    def __init__(self):
        self.complete = False
        self.fail_next_check = False
        # Optional gate: when set, is_complete() parks until the test opens
        # it, holding the caller inside the LangGraph resume
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.start_count = 0
        self.is_complete_count = 0
        self.get_result_count = 0
        self.requests = {}

    async def start(self, request):
        self.start_count += 1
        workflow_id = f"tool-{request.execution_id}"
        self.requests[workflow_id] = request
        return workflow_id

    async def is_complete(self, workflow_id):
        self.is_complete_count += 1

        if self.gate is not None:
            self.entered.set()
            await self.gate.wait()

        if self.fail_next_check:
            # One-off transient failure, e.g. Temporal briefly unreachable
            self.fail_next_check = False
            raise RuntimeError("Temporal unavailable")

        return self.complete

    async def get_result(self, workflow_id):
        self.get_result_count += 1
        request = self.requests[workflow_id]
        return ToolWorkflowResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            status=ToolWorkflowStatus.COMPLETED,
            output={"approved": True},
        )


class CountingProposer:

    def __init__(self):
        self.calls = 0

    async def __call__(self, state):
        self.calls += 1
        return {
            "proposed_tool": "approve_partner",
            "proposed_arguments": {"partner_id": "partner-123"},
        }


@pytest.fixture
def fake_runner() -> FakeTemporalRunner:
    return FakeTemporalRunner()


@pytest.fixture
def proposer() -> CountingProposer:
    return CountingProposer()


@pytest.fixture
def graph(fake_runner, proposer):
    registry = ToolRegistry()
    registry.register(ApprovePartnerTool())

    return build_onboarding_graph(
        workflow_runner=fake_runner,
        registry=registry,
        action_proposer=proposer,
        checkpointer=InMemorySaver(),
    )


@pytest.fixture
def inbox() -> InMemoryEventInbox:
    return InMemoryEventInbox()


@pytest.fixture
def agent_lock() -> InMemoryAgentRunLock:
    return InMemoryAgentRunLock()


@pytest.fixture
def handler(graph, inbox, agent_lock) -> AgentWorkflowEventHandler:
    return AgentWorkflowEventHandler(
        graph=graph,
        inbox=inbox,
        agent_lock=agent_lock,
    )


@pytest.fixture
def config() -> dict:
    return {"configurable": {"thread_id": AGENT_RUN_ID}}


@pytest.fixture
def initial_state() -> dict:
    return {
        "agent_run_id": AGENT_RUN_ID,
        "agent_id": "onboarding-agent",
        "user_id": "user-1",
    }


async def start_agent(graph, initial_state, config) -> str:
    # 1. Start agent; Temporal is still running, so the graph pauses
    await graph.ainvoke(initial_state, config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("wait_for_execution",)

    return snapshot.values["temporal_workflow_id"]


def completion_event(workflow_id: str, **overrides) -> WorkflowCompletionEvent:
    values = {
        "event_id": "event-1",
        "agent_run_id": AGENT_RUN_ID,
        "workflow_id": workflow_id,
        "event_type": WorkflowEventType.COMPLETED,
    }
    values.update(overrides)
    return WorkflowCompletionEvent(**values)


@pytest.mark.asyncio
async def test_completion_event_resumes_agent(
    graph, handler, fake_runner, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)

    # 2. Temporal completes
    fake_runner.complete = True

    # 3. Deliver completion event
    event = completion_event(workflow_id)
    result = await handler.handle(event)

    # 4. Verify graph resumed
    assert result["workflow_status"] == "COMPLETED"
    assert result["execution_phase"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}

    # And the outcome is checkpointed, not only returned
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["execution_phase"] == "COMPLETED"


@pytest.mark.asyncio
async def test_event_does_not_restart_temporal(
    graph, handler, fake_runner, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    await handler.handle(completion_event(workflow_id))

    assert fake_runner.start_count == 1


@pytest.mark.asyncio
async def test_event_does_not_repropose_action(
    graph, handler, fake_runner, proposer, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    await handler.handle(completion_event(workflow_id))

    assert proposer.calls == 1


@pytest.mark.asyncio
async def test_wrong_workflow_id_is_rejected(
    graph, handler, fake_runner, initial_state, config,
):
    await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    checks_before = fake_runner.is_complete_count

    with pytest.raises(ValueError, match="Workflow ID mismatch"):
        await handler.handle(completion_event("tool-some-other-workflow"))

    # Rejected before resuming: Temporal untouched, run still paused
    assert fake_runner.is_complete_count == checks_before
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("wait_for_execution",)
    assert snapshot.values["execution_phase"] == "WAITING"


@pytest.mark.asyncio
async def test_missing_checkpoint_is_rejected(handler, fake_runner):
    event = completion_event(
        "tool-anything",
        agent_run_id="agent-run-unknown",
    )

    with pytest.raises(ValueError, match="agent-run-unknown"):
        await handler.handle(event)

    assert fake_runner.start_count == 0
    assert fake_runner.is_complete_count == 0


@pytest.mark.asyncio
async def test_duplicate_event_after_completion(
    graph, handler, fake_runner, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    event = completion_event(workflow_id)

    first_result = await handler.handle(event)
    checks_after_first = fake_runner.is_complete_count

    # Event delivery is at-least-once; the redelivery must be a no-op
    second_result = await handler.handle(event)

    assert second_result["execution_phase"] == "COMPLETED"
    assert second_result["tool_output"] == first_result["tool_output"]
    assert fake_runner.start_count == 1
    assert fake_runner.is_complete_count == checks_after_first


@pytest.mark.asyncio
async def test_handled_event_is_marked_processed(
    graph, handler, fake_runner, inbox, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    await handler.handle(completion_event(workflow_id))

    record = inbox.records["event-1"]
    assert record.status == EventProcessingStatus.PROCESSED
    assert record.attempts == 1


@pytest.mark.asyncio
async def test_duplicate_event_is_not_counted_as_another_attempt(
    graph, handler, fake_runner, inbox, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    event = completion_event(workflow_id)

    await handler.handle(event)
    await handler.handle(event)

    assert inbox.records["event-1"].attempts == 1


@pytest.mark.asyncio
async def test_failed_event_is_recorded_and_can_be_retried(
    graph, handler, fake_runner, inbox, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    # Same event ID, first delivered with a bad workflow ID
    with pytest.raises(ValueError):
        await handler.handle(completion_event("tool-some-other-workflow"))

    record = inbox.records["event-1"]
    assert record.status == EventProcessingStatus.FAILED
    assert record.last_error == "Workflow ID mismatch"

    # A failed event is not treated as done: redelivery is processed
    result = await handler.handle(completion_event(workflow_id))

    assert result["execution_phase"] == "COMPLETED"
    assert record.status == EventProcessingStatus.PROCESSED
    assert record.attempts == 2


# Integration: handler + inbox + checkpointed graph


def count_resumes(graph, monkeypatch) -> list:
    # Spy on graph.ainvoke to count how many times the handler resumes
    resumes = []
    original = graph.ainvoke

    async def spy(input, *args, **kwargs):
        resumes.append(input)
        return await original(input, *args, **kwargs)

    monkeypatch.setattr(graph, "ainvoke", spy)
    return resumes


@pytest.mark.asyncio
async def test_duplicate_delivery_resumes_graph_once(
    graph, handler, fake_runner, inbox, initial_state, config, monkeypatch,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    resumes = count_resumes(graph, monkeypatch)
    event = completion_event(workflow_id)

    first = await handler.handle(event)
    second = await handler.handle(event)

    assert len(resumes) == 1
    assert fake_runner.get_result_count == 1
    assert first["execution_phase"] == second["execution_phase"] == "COMPLETED"
    assert inbox.records["event-1"].status == EventProcessingStatus.PROCESSED


@pytest.mark.asyncio
async def test_transient_failure_is_recorded_and_retry_succeeds(
    graph, handler, fake_runner, inbox, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    fake_runner.fail_next_check = True
    event = completion_event(workflow_id)

    with pytest.raises(RuntimeError, match="Temporal unavailable"):
        await handler.handle(event)

    record = inbox.records["event-1"]
    assert record.status == EventProcessingStatus.FAILED
    assert record.last_error == "Temporal unavailable"

    # The failed resume left the graph mid-run, not at the interrupt
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("check_execution",)

    # Redelivery must actually advance the agent, not just mark it handled
    result = await handler.handle(event)

    assert result["execution_phase"] == "COMPLETED"
    assert result["workflow_status"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    assert record.status == EventProcessingStatus.PROCESSED
    assert record.attempts == 2
    assert fake_runner.start_count == 1

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["execution_phase"] == "COMPLETED"


@pytest.mark.asyncio
async def test_new_event_for_completed_agent_does_not_restart_workflow(
    graph, handler, fake_runner, inbox, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    await handler.handle(completion_event(workflow_id, event_id="event-1"))
    checks_before = fake_runner.is_complete_count

    # Different event ID, same business workflow: event_id deduplication
    # can't catch this, so the agent's own state must
    result = await handler.handle(
        completion_event(workflow_id, event_id="event-2")
    )

    assert result["execution_phase"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    assert fake_runner.start_count == 1
    assert fake_runner.is_complete_count == checks_before
    assert fake_runner.get_result_count == 1

    # Both events are recorded as processed; only the first did any work
    assert inbox.records["event-2"].status == EventProcessingStatus.PROCESSED


@pytest.mark.asyncio
async def test_handler_does_not_acknowledge_after_lease_expires(
    graph, fake_runner, initial_state, config,
):
    # The worker is slow: its lease runs out while it resumes the graph,
    # so another worker may already own the event
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    inbox = InMemoryEventInbox(clock=lambda: now[0])
    handler = AgentWorkflowEventHandler(
        graph=graph,
        inbox=inbox,
        agent_lock=InMemoryAgentRunLock(),
    )

    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    original_get_result = fake_runner.get_result

    async def slow_get_result(workflow_id):
        now[0] += timedelta(seconds=120)
        return await original_get_result(workflow_id)

    fake_runner.get_result = slow_get_result

    with pytest.raises(RuntimeError, match="lease ownership lost"):
        await handler.handle(completion_event(workflow_id))

    # Not acknowledged: the event stays claimable for redelivery
    record = inbox.records["event-1"]
    assert record.status == EventProcessingStatus.PROCESSING
    assert await inbox.claim("event-1", owner="worker-b") is True


@pytest.mark.asyncio
async def test_event_is_refused_while_another_worker_holds_the_run(
    graph, handler, fake_runner, inbox, agent_lock, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    # Another worker is mid-way through this agent run (a different event)
    assert await agent_lock.acquire(AGENT_RUN_ID, owner="other-worker")
    checks_before = fake_runner.is_complete_count

    with pytest.raises(RuntimeError, match="currently being processed"):
        await handler.handle(completion_event(workflow_id))

    # The graph was not touched, and the event is left retryable
    assert fake_runner.is_complete_count == checks_before
    assert (await graph.aget_state(config)).next == ("wait_for_execution",)
    assert inbox.records["event-1"].status == EventProcessingStatus.FAILED

    # The other worker's lock is not released by the refused handler
    assert agent_lock.leases[AGENT_RUN_ID][0] == "other-worker"

    # Once the run is free, redelivery completes it
    await agent_lock.release(AGENT_RUN_ID, owner="other-worker")
    result = await handler.handle(completion_event(workflow_id))
    assert result["execution_phase"] == "COMPLETED"


@pytest.mark.asyncio
async def test_lock_is_released_after_success(
    graph, handler, fake_runner, agent_lock, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    await handler.handle(completion_event(workflow_id))

    assert AGENT_RUN_ID not in agent_lock.leases


@pytest.mark.asyncio
async def test_lock_is_released_after_failure(
    graph, handler, fake_runner, agent_lock, initial_state, config,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True
    fake_runner.fail_next_check = True

    with pytest.raises(RuntimeError, match="Temporal unavailable"):
        await handler.handle(completion_event(workflow_id))

    # A failed attempt must not leave the run locked until lease expiry
    assert AGENT_RUN_ID not in agent_lock.leases


@pytest.mark.asyncio
async def test_two_events_for_same_run_cannot_resume_concurrently(
    graph, handler, fake_runner, inbox, initial_state, config, monkeypatch,
):
    workflow_id = await start_agent(graph, initial_state, config)
    fake_runner.complete = True

    # Track how many LangGraph resumes are in flight at once
    in_flight = 0
    max_in_flight = 0
    resumes = 0
    original_ainvoke = graph.ainvoke

    async def tracking_ainvoke(input, *args, **kwargs):
        nonlocal in_flight, max_in_flight, resumes
        resumes += 1
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            return await original_ainvoke(input, *args, **kwargs)
        finally:
            in_flight -= 1

    monkeypatch.setattr(graph, "ainvoke", tracking_ainvoke)

    # Two distinct events (different event IDs, so the inbox can't dedupe
    # them) about the same agent run
    first_event = completion_event(workflow_id, event_id="event-1")
    second_event = completion_event(workflow_id, event_id="event-2")

    # The first handler enters LangGraph and is parked holding the run lock
    fake_runner.gate = asyncio.Event()
    first = asyncio.create_task(handler.handle(first_event))
    await asyncio.wait_for(fake_runner.entered.wait(), timeout=5)

    # The second arrives mid-resume and is refused rather than interleaving.
    # Bounded: without the run lock it would enter LangGraph and park at
    # the gate too, so the timeout turns that into a clear failure
    with pytest.raises(RuntimeError, match="currently being processed"):
        await asyncio.wait_for(handler.handle(second_event), timeout=2)

    fake_runner.gate.set()
    first_result = await asyncio.wait_for(first, timeout=5)

    assert first_result["execution_phase"] == "COMPLETED"
    assert max_in_flight == 1
    assert resumes == 1

    # The refused event is retryable, and its redelivery finds the run
    # already complete: no second resume, no restart
    assert inbox.records["event-2"].status == EventProcessingStatus.FAILED
    fake_runner.gate = None

    second_result = await handler.handle(second_event)

    assert second_result["execution_phase"] == "COMPLETED"
    assert resumes == 1
    assert fake_runner.start_count == 1
