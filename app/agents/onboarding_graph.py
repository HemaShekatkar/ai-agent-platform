from collections.abc import Awaitable, Callable
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END, START, StateGraph

from app.agents.messages import (
    ACTION_AWAITING_COMPLETION,
    ACTION_COMPLETED,
    action_ended,
)
from app.agents.state import OnboardingAgentState, AgentExecutionPhase
from app.agents.temporal_client import ToolWorkflowRunner
from app.tools.fingerprint import action_fingerprint
from app.tools.models import ApprovalRequirement
from app.tools.registry import ToolRegistry
from app.workflows.models import ToolWorkflowRequest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

checkpointer = InMemorySaver()


# Returns {"proposed_tool": ..., "proposed_arguments": ...}; untrusted output
# (eventually from the model), so everything downstream re-derives policy
# from the registry rather than from this proposal
ActionProposer = Callable[[OnboardingAgentState], Awaitable[dict[str, Any]]]


def build_onboarding_graph(
    workflow_runner: ToolWorkflowRunner,
    registry: ToolRegistry,
    action_proposer: ActionProposer | None = None,
    checkpointer=None,
):

    async def placeholder_propose_action(
        state: OnboardingAgentState,
    ):
        # Deterministic placeholder.
        # Later replaced by Model Gateway / LLM.
        return {
            "proposed_tool": "approve_partner",
            "proposed_arguments": {
                "partner_id": "partner-123",
            },
        }

    async def prepare_execution(
        state: OnboardingAgentState,
    ):
        tool = registry.get(state["proposed_tool"])
        definition = tool.definition

        fingerprint = action_fingerprint(
            definition.name,
            state["proposed_arguments"],
        )

        # Same run + same action -> same execution ID (and workflow ID), so a
        # retried graph step maps to the same Temporal workflow rather than
        # minting a new identity for an action already in flight
        execution_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{state['agent_run_id']}:{fingerprint}",
            )
        )

        return {
            "execution_id": execution_id,
            "requires_approval": (
                definition.approval_requirement
                == ApprovalRequirement.HUMAN
            ),
        }

    async def start_execution(
        state: OnboardingAgentState,
    ):
        request = ToolWorkflowRequest(
            execution_id=state["execution_id"],
            tool_name=state["proposed_tool"],
            arguments=state["proposed_arguments"],
            agent_id=state["agent_id"],
            user_id=state["user_id"],
            requires_approval=state["requires_approval"],
        )

        workflow_id = await workflow_runner.start(request)

        return {
            "temporal_workflow_id": workflow_id,
            "workflow_status": "RUNNING",
        }

    async def check_execution(
        state: OnboardingAgentState,
    ):
        workflow_id = state["temporal_workflow_id"]

        if not await workflow_runner.is_complete(workflow_id):
            return {"workflow_status": "RUNNING"}

        result = await workflow_runner.get_result(workflow_id)

        return {
            "workflow_status": result.status.value,
            "tool_output": result.output,
            "error": result.error,
        }

    async def waiting(state: OnboardingAgentState):
        return {
            "execution_phase": AgentExecutionPhase.WAITING.value,
            "final_response": ACTION_AWAITING_COMPLETION,
        }

    def route_after_check(state: OnboardingAgentState):
        if state["workflow_status"] == "RUNNING":
            return "waiting"
        return "observe_result"

    async def observe_result(state: OnboardingAgentState):
        if state["workflow_status"] == "COMPLETED":
            return {
                "execution_phase": AgentExecutionPhase.COMPLETED.value,
                "final_response": ACTION_COMPLETED,
            }

        return {
            "execution_phase": AgentExecutionPhase.FAILED.value,
            "final_response": (
                action_ended(state["workflow_status"])
            ),
        }

    async def wait_for_execution(state: OnboardingAgentState):
        request = {
            "execution_id": state["execution_id"],
            "workflow_id": state["temporal_workflow_id"],
            "reason": "WAITING_FOR_TEMPORAL",
        }

        # Reject a bad resume by pausing again rather than raising. LangGraph
        # persists each resume value and replays it to the matching
        # interrupt() call on re-entry, so raising here would replay the bad
        # payload on every later resume and wedge the run permanently.
        # A new interrupt() call gets the next resume value instead.
        while True:
            resume_payload = interrupt(request)

            if not isinstance(resume_payload, dict):
                rejection = "Invalid resume payload"
            elif (
                resume_payload.get("workflow_id")
                != state["temporal_workflow_id"]
            ):
                rejection = "Workflow identity mismatch"
            else:
                return {"execution_phase": AgentExecutionPhase.WAITING.value}

            request = {**request, "rejected_resume": rejection}

    graph = StateGraph(OnboardingAgentState)

    # 1. Register nodes
    graph.add_node(
        "propose_action",
        action_proposer or placeholder_propose_action,
    )
    graph.add_node("prepare_execution", prepare_execution)
    graph.add_node("start_execution", start_execution)
    graph.add_node("check_execution", check_execution)
    graph.add_node("waiting", waiting)
    graph.add_node("wait_for_execution", wait_for_execution)
    graph.add_node("observe_result", observe_result)

    # 2. Define execution flow
    graph.add_edge(START, "propose_action")
    graph.add_edge("propose_action", "prepare_execution")
    graph.add_edge("prepare_execution", "start_execution")
    graph.add_edge("start_execution", "check_execution")

    # 3. Route based on Temporal workflow status
    graph.add_conditional_edges(
        "check_execution",
        route_after_check,
        {
            "waiting": "waiting",
            "observe_result": "observe_result",
        },
    )

    # 4. Pause while Temporal runs. "waiting" records the WAITING phase as
    # its own step: an interrupted node's return value is not checkpointed,
    # so the phase must be saved before wait_for_execution pauses. A resume
    # re-checks Temporal; the loop repeats until the workflow finishes.
    graph.add_edge("waiting", "wait_for_execution")
    graph.add_edge("wait_for_execution", "check_execution")

    # 5. Terminal node
    graph.add_edge("observe_result", END)

    # 6. Compile with checkpointing
    return graph.compile(checkpointer=checkpointer)
