
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class ResolutionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class OnboardingRecord:
    onboarding_id: str
    blocker_code: str


@dataclass(frozen=True)
class BlockerResolution:
    blocker_code: str
    description: str
    policy_id: str
    approval_role: str
    directory_key: str


@dataclass(frozen=True)
class ResolutionResult:
    status: ResolutionStatus
    blocker_description: str
    approval_role: str
    assigned_approver: str | None
    evidence_refs: tuple[str, ...]
    unresolved: tuple[str, ...]


def resolve_launch_blocker(
    onboarding_id: str,
    get_onboarding: Callable[[str], OnboardingRecord],
    get_blocker: Callable[[str], BlockerResolution],
    validate_policy: Callable[[str], bool],
    get_approver: Callable[[str], str | None],
    max_directory_attempts: int = 3,
) -> ResolutionResult:
    if max_directory_attempts < 1:
        raise ValueError("max_directory_attempts must be >= 1")

    onboarding = get_onboarding(onboarding_id)
    blocker = get_blocker(onboarding.blocker_code)

    if blocker.blocker_code != onboarding.blocker_code:
        raise ValueError("Blocker registry returned mismatched code")

    if not validate_policy(blocker.policy_id):
        return ResolutionResult(
            status=ResolutionStatus.ESCALATE,
            blocker_description=blocker.description,
            approval_role=blocker.approval_role,
            assigned_approver=None,
            evidence_refs=(onboarding_id, blocker.blocker_code),
            unresolved=("Applicable policy could not be validated",),
        )

    approver = None

    for _ in range(max_directory_attempts):
        try:
            approver = get_approver(blocker.directory_key)
            if approver:
                break
        except (TimeoutError, ConnectionError):
            continue

    # An empty name is not a verified approver
    if not approver:
        return ResolutionResult(
            status=ResolutionStatus.PARTIAL,
            blocker_description=blocker.description,
            approval_role=blocker.approval_role,
            assigned_approver=None,
            evidence_refs=(
                onboarding_id,
                blocker.blocker_code,
                blocker.policy_id,
            ),
            unresolved=("Assigned approver could not be verified",),
        )

    return ResolutionResult(
        status=ResolutionStatus.COMPLETE,
        blocker_description=blocker.description,
        approval_role=blocker.approval_role,
        assigned_approver=approver,
        evidence_refs=(
            onboarding_id,
            blocker.blocker_code,
            blocker.policy_id,
        ),
        unresolved=(),
    )
