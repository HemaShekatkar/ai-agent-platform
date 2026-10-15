import pytest

from app.evals.agent_dataset import ONBOARDING_EVAL_CASES
from app.evals.agent_evaluator import (
    ProposedAction,
    evaluate_case,
)


def get_case(case_id):
    return next(
        case for case in ONBOARDING_EVAL_CASES
        if case.case_id == case_id
    )


@pytest.mark.asyncio
async def test_prompt_injection_refused():

    async def safe_proposer(eval_input):
        assert "SYSTEM OVERRIDE" in eval_input.untrusted_content
        return None

    result = await evaluate_case(
        get_case("prompt-injection-document"),
        safe_proposer,
    )

    assert result.task_success is True


@pytest.mark.asyncio
async def test_unauthorized_action_detected():

    async def unsafe_proposer(eval_input):
        assert "approve_partner" not in eval_input.trusted_permissions

        return ProposedAction(
            tool_name="approve_partner",
            arguments={"partner_id": "partner-123"},
            requires_approval=True,
        )

    result = await evaluate_case(
        get_case("unauthorized-user"),
        unsafe_proposer,
    )

    assert result.task_success is False
    assert result.tool_correct is False


@pytest.mark.asyncio
async def test_approval_bypass_detected():

    async def unsafe_proposer(eval_input):
        return ProposedAction(
            tool_name="approve_partner",
            arguments={"partner_id": "partner-123"},
            requires_approval=False,
        )

    result = await evaluate_case(
        get_case("approval-bypass"),
        unsafe_proposer,
    )

    assert result.task_success is False
    assert result.tool_correct is False
