from uuid import uuid4

from langgraph.types import Command

from app.agents.agent_lock import AgentRunLock
from app.agents.events import WorkflowCompletionEvent
from app.agents.event_inbox import EventInbox


class AgentWorkflowEventHandler:

    def __init__(
        self,
        graph,
        inbox: EventInbox,
        agent_lock: AgentRunLock,
    ):
        self.graph = graph
        self.inbox = inbox
        self.agent_lock = agent_lock

    async def handle(self, event: WorkflowCompletionEvent):
        await self.inbox.receive(event.event_id)

        owner = str(uuid4())
        if not await self.inbox.claim(event.event_id, owner):
            # Duplicate (already processed or in flight): don't resume again,
            # but still give the caller the agent's current state
            return await self._current_state(event)

        lock_acquired = False

        try:
            # The inbox lease guards one event; this lock guards the agent
            # run, so two different events for the same run can't resume
            # its graph concurrently
            lock_acquired = await self.agent_lock.acquire(
                agent_run_id=event.agent_run_id,
                owner=owner,
            )

            if not lock_acquired:
                raise RuntimeError(
                    "Agent run is currently being processed"
                )

            result = await self._resume_graph(event)

            acknowledged = await self.inbox.mark_processed(
                event.event_id,
                owner,
            )

            if not acknowledged:
                raise RuntimeError("Event lease ownership lost")

            return result

        except Exception as exc:
            await self.inbox.mark_failed(
                event.event_id,
                owner,
                str(exc),
            )
            raise

        finally:
            if lock_acquired:
                await self.agent_lock.release(
                    event.agent_run_id,
                    owner,
                )

    @staticmethod
    def _config(event: WorkflowCompletionEvent) -> dict:
        return {
            "configurable": {
                "thread_id": event.agent_run_id,
            }
        }

    async def _current_state(self, event: WorkflowCompletionEvent):
        snapshot = await self.graph.aget_state(self._config(event))
        return snapshot.values

    async def _resume_graph(
        self,
        event: WorkflowCompletionEvent,
    ):
        config = self._config(event)

        # Load checkpoint
        snapshot = await self.graph.aget_state(config)

        if not snapshot.values:
            raise ValueError(
                f"Agent checkpoint not found: {event.agent_run_id}"
            )

        # Validate workflow identity
        expected_workflow_id = snapshot.values.get(
            "temporal_workflow_id"
        )

        if expected_workflow_id != event.workflow_id:
            raise ValueError(
                "Workflow ID mismatch"
            )

        # Finished run: nothing to resume (semantic idempotency, even for a
        # new event ID about the same workflow)
        if not snapshot.next:
            return snapshot.values

        # Paused at the expected interrupt: resume it
        if "wait_for_execution" in snapshot.next:
            return await self.graph.ainvoke(
                Command(
                    resume={
                        "workflow_id": event.workflow_id,
                    }
                ),
                config=config,
            )

        # Stopped mid-run, e.g. an earlier delivery resumed the graph and
        # check_execution then hit a transient error. Continue from the
        # checkpoint so the failed step re-runs; returning the stale state
        # here would mark the event processed while the agent never advances.
        return await self.graph.ainvoke(None, config=config)
