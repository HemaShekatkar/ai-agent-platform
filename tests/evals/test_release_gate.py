import pytest

from app.evals.release_gate import (
    ReleaseDecision,
    ReleaseEvidence,
    ReleasePolicy,
    evaluate_release,
)


def make_evidence(
    evaluated_cases=100,
    successful_tasks=98,
    correct_tool_decisions=99,
    enforcement_tests_run=50,
    critical_enforcement_failures=0,
):
    return ReleaseEvidence(
        evaluated_cases=evaluated_cases,
        successful_tasks=successful_tasks,
        correct_tool_decisions=correct_tool_decisions,
        enforcement_tests_run=enforcement_tests_run,
        critical_enforcement_failures=critical_enforcement_failures,
    )


def test_release_passes():
    result = evaluate_release(
        make_evidence(),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.PASS


def test_critical_failure_blocks_release():
    result = evaluate_release(
        make_evidence(critical_enforcement_failures=1),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.BLOCK


def test_insufficient_eval_data():
    result = evaluate_release(
        make_evidence(
            evaluated_cases=10,
            successful_tasks=10,
            correct_tool_decisions=10,
        ),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.INSUFFICIENT_EVIDENCE


def test_missing_enforcement_evidence():
    result = evaluate_release(
        make_evidence(enforcement_tests_run=0),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.INSUFFICIENT_EVIDENCE


def test_low_task_success_blocks_release():
    result = evaluate_release(
        make_evidence(successful_tasks=90),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.BLOCK


def test_low_tool_accuracy_blocks_release():
    result = evaluate_release(
        make_evidence(correct_tool_decisions=95),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.BLOCK


def test_inconsistent_evidence_rejected():
    with pytest.raises(ValueError):
        evaluate_release(
            make_evidence(successful_tasks=101),
            ReleasePolicy(),
        )
