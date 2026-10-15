import pytest
from pydantic import BaseModel
from temporalio.exceptions import ApplicationError

from app.tools.approval import ApprovalDecision, ApprovalPolicy, ApprovalStatus
from app.tools.base import BaseTool
from app.tools.exceptions import PermanentToolError, TransientToolError
from app.tools.executor import ToolExecutor
from app.tools.fingerprint import action_fingerprint
from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolExecutionContext,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.policy import ToolAuthorizer
from app.tools.registry import ToolRegistry
from app.workflows.activities import (
    ExecuteToolActivityRequest,
    ToolExecutionActivities,
)


class PartnerInput(BaseModel):
    partner_id: str


class CountingTool(BaseTool[PartnerInput, str]):

    def __init__(self, approval_requirement: ApprovalRequirement):
        self._definition = ToolDefinition(
            name="approve_partner",
            description="Approve a partner for onboarding.",
            input_schema=PartnerInput,
            risk=ToolRisk.HIGH,
            required_permissions={"partner.approve"},
            approval_requirement=approval_requirement,
        )
        self.call_count = 0
        self.received_runtimes: list[ToolRuntimeContext] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, input: PartnerInput, runtime: ToolRuntimeContext) -> str:
        self.call_count += 1
        self.received_runtimes.append(runtime)
        return f"approved:{input.partner_id}"


class FakeIdentityResolver:
    # A mutable permission store standing in for the identity provider

    def __init__(
        self,
        agent_permissions: dict[str, set[str]],
        user_permissions: dict[str, set[str]],
    ):
        self.agent_permissions = agent_permissions
        self.user_permissions = user_permissions
        self.calls: list[tuple[str, str]] = []

    def resolve(self, agent_id: str, user_id: str) -> ToolExecutionContext:
        self.calls.append((agent_id, user_id))
        return ToolExecutionContext(
            agent_id=agent_id,
            user_id=user_id,
            agent_permissions=set(self.agent_permissions.get(agent_id, set())),
            user_permissions=set(self.user_permissions.get(user_id, set())),
        )

    def revoke_user_permission(self, user_id: str, permission: str) -> None:
        self.user_permissions[user_id].discard(permission)


def make_resolver() -> FakeIdentityResolver:
    return FakeIdentityResolver(
        agent_permissions={"onboarding-agent": {"partner.approve"}},
        user_permissions={"user-123": {"partner.approve"}},
    )


def make_activities(
    resolver: FakeIdentityResolver,
    tool: CountingTool,
) -> ToolExecutionActivities:
    registry = ToolRegistry()
    registry.register(tool)

    return ToolExecutionActivities(
        identity_resolver=resolver,
        executor=ToolExecutor(
            registry=registry,
            authorizer=ToolAuthorizer(),
            approval_policy=ApprovalPolicy(),
        ),
    )


def make_request(
    execution_id: str = "exec-1",
    approval: ApprovalDecision | None = None,
) -> ExecuteToolActivityRequest:
    return ExecuteToolActivityRequest(
        execution_id=execution_id,
        tool_name="approve_partner",
        arguments={"partner_id": "partner-123"},
        agent_id="onboarding-agent",
        user_id="user-123",
        approval=approval,
    )


# Fingerprint of the action every approval in this file is granted for
APPROVED_ACTION = action_fingerprint(
    "approve_partner",
    {"partner_id": "partner-123"},
)


def approved_for(execution_id: str) -> ApprovalDecision:
    return ApprovalDecision(
        execution_id=execution_id,
        action_fingerprint=APPROVED_ACTION,
        status=ApprovalStatus.APPROVED,
        approved_by="reviewer-1",
    )


def test_activity_resolves_current_identity():
    resolver = make_resolver()
    tool = CountingTool(ApprovalRequirement.NONE)
    activities = make_activities(resolver, tool)
    request = make_request()

    # The request carries IDs only; permissions come from the resolver
    assert not hasattr(request, "user_permissions")

    activities.execute_tool(request)

    assert resolver.calls == [("onboarding-agent", "user-123")]

    # Each execution resolves again rather than caching a snapshot
    activities.execute_tool(request)

    assert len(resolver.calls) == 2


def test_activity_passes_execution_id():
    resolver = make_resolver()
    tool = CountingTool(ApprovalRequirement.HUMAN)
    activities = make_activities(resolver, tool)

    activities.execute_tool(
        make_request(execution_id="exec-1", approval=approved_for("exec-1"))
    )

    # The tool runs under the request's execution ID
    assert tool.received_runtimes[0].execution_id == "exec-1"

    # and the approval check uses it: an approval for another execution fails
    with pytest.raises(ApplicationError) as exc_info:
        activities.execute_tool(
            make_request(execution_id="exec-2", approval=approved_for("exec-1"))
        )

    assert exc_info.value.type == "ToolApprovalRequiredError"
    assert tool.call_count == 1


def test_activity_executes_authorized_tool():
    resolver = make_resolver()
    tool = CountingTool(ApprovalRequirement.NONE)
    activities = make_activities(resolver, tool)

    result = activities.execute_tool(make_request())

    assert result.success is True
    assert result.tool_name == "approve_partner"
    assert result.output == "approved:partner-123"
    assert tool.call_count == 1


def test_activity_blocks_when_current_permission_revoked():
    resolver = make_resolver()
    tool = CountingTool(ApprovalRequirement.HUMAN)
    activities = make_activities(resolver, tool)

    # The same activity instance succeeds while the user holds the permission
    activities.execute_tool(
        make_request(execution_id="exec-1", approval=approved_for("exec-1"))
    )
    assert tool.call_count == 1

    # The permission is revoked, e.g. during a later approval wait
    resolver.revoke_user_permission("user-123", "partner.approve")

    # A fresh, valid approval does not override the current revocation
    with pytest.raises(ApplicationError) as exc_info:
        activities.execute_tool(
            make_request(execution_id="exec-2", approval=approved_for("exec-2"))
        )

    assert exc_info.value.type == "ToolAuthorizationError"

    # Retrying cannot restore a revoked permission, so Temporal must not retry
    assert exc_info.value.non_retryable is True
    assert tool.call_count == 1


def test_activity_does_not_retry_unknown_tool():
    resolver = make_resolver()
    tool = CountingTool(ApprovalRequirement.NONE)
    activities = make_activities(resolver, tool)

    request = ExecuteToolActivityRequest(
        execution_id="exec-1",
        tool_name="delete_partner",
        arguments={"partner_id": "partner-123"},
        agent_id="onboarding-agent",
        user_id="user-123",
    )

    with pytest.raises(ApplicationError) as exc_info:
        activities.execute_tool(request)

    assert exc_info.value.type == "ToolNotFoundError"

    # An unregistered tool name cannot succeed on retry
    assert exc_info.value.non_retryable is True
    assert tool.call_count == 0


# Exception translation: the activity converts tool errors into Temporal
# ApplicationErrors, and non_retryable decides whether Temporal retries


class RaisingExecutor:
    # Stands in for ToolExecutor and fails the way a real tool call would

    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def execute(self, call, execution_id, context, approval=None):
        self.calls += 1
        raise self.error


@pytest.fixture
def activity_request() -> ExecuteToolActivityRequest:
    return make_request()


@pytest.fixture
def tool_activities_raising():
    def build(error: Exception) -> ToolExecutionActivities:
        return ToolExecutionActivities(
            identity_resolver=make_resolver(),
            executor=RaisingExecutor(error),
        )

    return build


@pytest.fixture
def counting_tool() -> CountingTool:
    return CountingTool(ApprovalRequirement.NONE)


@pytest.fixture
def resolver() -> FakeIdentityResolver:
    return make_resolver()


@pytest.fixture
def tool_activities(resolver, counting_tool) -> ToolExecutionActivities:
    # Real executor, so authorization and validation actually run
    return make_activities(resolver, counting_tool)


def test_transient_tool_error_is_retryable(
    tool_activities_raising,
    activity_request,
):
    tool_activities = tool_activities_raising(
        TransientToolError("partner service timed out")
    )

    with pytest.raises(ApplicationError) as exc_info:
        tool_activities.execute_tool(activity_request)

    error = exc_info.value

    # A timeout may clear up, so Temporal should apply the retry policy
    assert error.type == "TransientToolError"
    assert error.non_retryable is False
    assert "partner service timed out" in str(error)
    assert isinstance(error.__cause__, TransientToolError)


def test_permanent_tool_error_is_non_retryable(
    tool_activities_raising,
    activity_request,
):
    tool_activities = tool_activities_raising(
        PermanentToolError("partner does not exist")
    )

    with pytest.raises(ApplicationError) as exc_info:
        tool_activities.execute_tool(activity_request)

    error = exc_info.value

    assert error.type == "PermanentToolError"
    assert error.non_retryable is True
    assert isinstance(error.__cause__, PermanentToolError)


def test_authorization_error_is_non_retryable(
    tool_activities,
    resolver,
    counting_tool,
    activity_request,
):
    resolver.revoke_user_permission("user-123", "partner.approve")

    with pytest.raises(ApplicationError) as exc_info:
        tool_activities.execute_tool(activity_request)

    error = exc_info.value

    assert error.type == "ToolAuthorizationError"
    assert error.non_retryable is True
    assert counting_tool.call_count == 0


def test_validation_error_is_non_retryable(
    tool_activities,
    counting_tool,
):
    request = ExecuteToolActivityRequest(
        execution_id="exec-1",
        tool_name="approve_partner",
        arguments={"partner_ref": "partner-123"},
        agent_id="onboarding-agent",
        user_id="user-123",
    )

    with pytest.raises(ApplicationError) as exc_info:
        tool_activities.execute_tool(request)

    error = exc_info.value

    # The same malformed arguments fail every attempt
    assert error.type == "ToolInputValidationError"
    assert error.non_retryable is True
    assert counting_tool.call_count == 0
