import pytest
from pydantic import BaseModel

from app.tools.approval import ApprovalDecision, ApprovalPolicy, ApprovalStatus
from app.tools.base import BaseTool
from app.tools.exceptions import (
    ToolApprovalRejectedError,
    ToolApprovalRequiredError,
    ToolAuthorizationError,
    ToolInputValidationError,
    ToolNotFoundError,
)
from app.tools.executor import ToolExecutor
from app.tools.fingerprint import action_fingerprint
from app.tools.models import (
    ApprovalRequirement,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.policy import ToolAuthorizer
from app.tools.registry import ToolRegistry


class PartnerInput(BaseModel):
    partner_id: str


class CountingTool(BaseTool[PartnerInput, str]):
    # Records every execution so tests can prove no side effect occurred

    def __init__(self, definition: ToolDefinition):
        self._definition = definition
        self.call_count = 0
        self.received_inputs: list[PartnerInput] = []
        self.received_runtimes: list[ToolRuntimeContext] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, input: PartnerInput, runtime: ToolRuntimeContext) -> str:
        self.call_count += 1
        self.received_inputs.append(input)
        self.received_runtimes.append(runtime)
        return f"{self._definition.name}:{input.partner_id}"


def lookup_tool() -> CountingTool:
    return CountingTool(
        ToolDefinition(
            name="lookup_partner",
            description="Look up a partner's onboarding status.",
            input_schema=PartnerInput,
            risk=ToolRisk.READ_ONLY,
            required_permissions={"partner.read"},
            approval_requirement=ApprovalRequirement.NONE,
        )
    )


def approve_tool() -> CountingTool:
    return CountingTool(
        ToolDefinition(
            name="approve_partner",
            description="Approve a partner for onboarding.",
            input_schema=PartnerInput,
            risk=ToolRisk.HIGH,
            required_permissions={"partner.approve"},
            approval_requirement=ApprovalRequirement.HUMAN,
        )
    )


# Fingerprint of the action every approval in this file is granted for
APPROVED_ACTION = action_fingerprint(
    "approve_partner",
    {"partner_id": "partner-123"},
)


def decision(execution_id: str, status: ApprovalStatus) -> ApprovalDecision:
    return ApprovalDecision(
        execution_id=execution_id,
        action_fingerprint=APPROVED_ACTION,
        status=status,
        approved_by="reviewer-1",
    )


def make_executor(*tools: CountingTool) -> ToolExecutor:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    return ToolExecutor(
        registry=registry,
        authorizer=ToolAuthorizer(),
        approval_policy=ApprovalPolicy(),
    )


def make_context(permissions: set[str]) -> ToolExecutionContext:
    # Agent and user hold the same permissions unless a test says otherwise
    return ToolExecutionContext(
        agent_id="onboarding-agent",
        user_id="user-123",
        agent_permissions=set(permissions),
        user_permissions=set(permissions),
    )


def call(tool_name: str, **arguments) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments or {"partner_id": "partner-123"},
    )


def test_executor_executes_authorized_read_only_tool():
    tool = lookup_tool()
    executor = make_executor(tool)

    result = executor.execute(
        call("lookup_partner"),
        execution_id="exec-1",
        context=make_context({"partner.read"}),
    )

    assert result.success is True
    assert result.tool_name == "lookup_partner"
    assert result.output == "lookup_partner:partner-123"
    assert tool.call_count == 1

    # The tool receives the validated model, not the raw arguments
    assert tool.received_inputs == [PartnerInput(partner_id="partner-123")]

    # and the execution ID it runs under, e.g. for idempotency keys
    assert [runtime.execution_id for runtime in tool.received_runtimes] == [
        "exec-1",
    ]


def test_executor_rejects_unknown_tool():
    tool = lookup_tool()
    executor = make_executor(tool)

    with pytest.raises(ToolNotFoundError):
        executor.execute(
            call("delete_partner"),
            execution_id="exec-1",
            context=make_context({"partner.read", "partner.delete"}),
        )

    assert tool.call_count == 0


def test_executor_rejects_unauthorized_tool():
    tool = lookup_tool()
    executor = make_executor(tool)

    with pytest.raises(ToolAuthorizationError):
        executor.execute(
            call("lookup_partner"),
            execution_id="exec-1",
            context=make_context(set()),
        )

    assert tool.call_count == 0


def test_executor_rejects_invalid_input():
    tool = lookup_tool()
    executor = make_executor(tool)

    with pytest.raises(ToolInputValidationError):
        executor.execute(
            call("lookup_partner", partner_ref="partner-123"),
            execution_id="exec-1",
            context=make_context({"partner.read"}),
        )

    assert tool.call_count == 0


def test_executor_blocks_when_approval_required():
    tool = approve_tool()
    executor = make_executor(tool)

    with pytest.raises(ToolApprovalRequiredError):
        executor.execute(
            call("approve_partner"),
            execution_id="exec-1",
            context=make_context({"partner.approve"}),
        )

    assert tool.call_count == 0


def test_executor_blocks_when_approval_rejected():
    tool = approve_tool()
    executor = make_executor(tool)

    with pytest.raises(ToolApprovalRejectedError):
        executor.execute(
            call("approve_partner"),
            execution_id="exec-1",
            context=make_context({"partner.approve"}),
            approval=decision("exec-1", ApprovalStatus.REJECTED),
        )

    assert tool.call_count == 0


def test_executor_executes_with_valid_approval():
    tool = approve_tool()
    executor = make_executor(tool)

    result = executor.execute(
        call("approve_partner"),
        execution_id="exec-1",
        context=make_context({"partner.approve"}),
        approval=decision("exec-1", ApprovalStatus.APPROVED),
    )

    assert result.success is True
    assert result.output == "approve_partner:partner-123"
    assert tool.call_count == 1

    # The tool runs under the same execution ID the approval was granted for
    assert tool.received_runtimes[0].execution_id == "exec-1"


# Execution-boundary invariant: no side effect occurs unless every control
# before the boundary succeeds. Each test below satisfies every control
# except one, so the tool would run if that single control were missing.


def test_tool_is_never_called_when_authorization_fails():
    tool = approve_tool()
    executor = make_executor(tool)

    # Valid input and a matching approval; only the user lacks permission
    context = ToolExecutionContext(
        agent_id="onboarding-agent",
        user_id="user-123",
        agent_permissions={"partner.approve"},
        user_permissions=set(),
    )

    with pytest.raises(ToolAuthorizationError):
        executor.execute(
            call("approve_partner"),
            execution_id="exec-1",
            context=context,
            approval=decision("exec-1", ApprovalStatus.APPROVED),
        )

    assert tool.call_count == 0


def test_tool_is_never_called_when_validation_fails():
    tool = approve_tool()
    executor = make_executor(tool)

    # Authorized and approved; only the arguments are invalid
    with pytest.raises(ToolInputValidationError):
        executor.execute(
            ToolCall(tool_name="approve_partner", arguments={"partner_id": 123}),
            execution_id="exec-1",
            context=make_context({"partner.approve"}),
            approval=decision("exec-1", ApprovalStatus.APPROVED),
        )

    assert tool.call_count == 0


def test_tool_is_never_called_when_approval_is_pending():
    tool = approve_tool()
    executor = make_executor(tool)

    # Authorized with valid input; the only approval is for another execution
    with pytest.raises(ToolApprovalRequiredError):
        executor.execute(
            call("approve_partner"),
            execution_id="exec-1",
            context=make_context({"partner.approve"}),
            approval=decision("exec-2", ApprovalStatus.APPROVED),
        )

    assert tool.call_count == 0


def test_tool_is_never_called_when_arguments_changed_after_approval():
    tool = approve_tool()
    executor = make_executor(tool)

    # Approved for partner-123, but the executed call targets partner-999:
    # same tool, same execution, different action
    with pytest.raises(ToolApprovalRequiredError):
        executor.execute(
            call("approve_partner", partner_id="partner-999"),
            execution_id="exec-1",
            context=make_context({"partner.approve"}),
            approval=decision("exec-1", ApprovalStatus.APPROVED),
        )

    assert tool.call_count == 0
