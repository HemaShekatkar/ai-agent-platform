
from app.rag.multi_hop import (
    BlockerResolution,
    OnboardingRecord,
    ResolutionStatus,
    resolve_launch_blocker,
)


def onboarding(_):
    return OnboardingRecord(
        onboarding_id="ONB-123",
        blocker_code="DR-042",
    )


def blocker(_):
    return BlockerResolution(
        blocker_code="DR-042",
        description="Data residency approval required",
        policy_id="EU-DATA-2026-011",
        approval_role="Regional Data Protection Officer",
        directory_key="EU_DPO",
    )


def test_successful_multi_hop_resolution():
    result = resolve_launch_blocker(
        "ONB-123",
        onboarding,
        blocker,
        validate_policy=lambda _: True,
        get_approver=lambda _: "Verified Approver",
    )

    assert result.status == ResolutionStatus.COMPLETE
    assert result.assigned_approver == "Verified Approver"
    assert result.unresolved == ()


def test_directory_failure_returns_partial_result():
    attempts = 0

    def unavailable_directory(_):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("Directory unavailable")

    result = resolve_launch_blocker(
        "ONB-123",
        onboarding,
        blocker,
        validate_policy=lambda _: True,
        get_approver=unavailable_directory,
        max_directory_attempts=3,
    )

    assert attempts == 3
    assert result.status == ResolutionStatus.PARTIAL
    assert result.assigned_approver is None
    assert "Assigned approver could not be verified" in result.unresolved


def test_empty_approver_name_is_not_verified():
    attempts = 0

    def blank_directory(_):
        nonlocal attempts
        attempts += 1
        return ""

    result = resolve_launch_blocker(
        "ONB-123",
        onboarding,
        blocker,
        validate_policy=lambda _: True,
        get_approver=blank_directory,
        max_directory_attempts=3,
    )

    assert attempts == 3
    assert result.status == ResolutionStatus.PARTIAL
    assert result.assigned_approver is None
    assert "Assigned approver could not be verified" in result.unresolved


def test_invalid_policy_requires_escalation():
    result = resolve_launch_blocker(
        "ONB-123",
        onboarding,
        blocker,
        validate_policy=lambda _: False,
        get_approver=lambda _: "Someone",
    )

    assert result.status == ResolutionStatus.ESCALATE
    assert result.assigned_approver is None


def test_directory_recovers_after_transient_failure():
    attempts = 0

    def flaky_directory(_):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("Temporary failure")
        return "Verified Approver"

    result = resolve_launch_blocker(
        "ONB-123",
        onboarding,
        blocker,
        validate_policy=lambda _: True,
        get_approver=flaky_directory,
        max_directory_attempts=3,
    )

    assert attempts == 3
    assert result.status == ResolutionStatus.COMPLETE
    assert result.assigned_approver == "Verified Approver"
