import pytest

from app.evals.agent_dataset import ONBOARDING_EVAL_CASES
from app.evals.agent_evaluator import (
    ProposedAction,
    evaluate_case,
)


@pytest.mark.asyncio
async def test_valid_action_passes():

    async def proposer(eval_input):
        return ProposedAction(
            tool_name="approve_partner",
            arguments={"partner_id": "partner-123"},
            requires_approval=True,
        )

    result = await evaluate_case(
        ONBOARDING_EVAL_CASES[0],
        proposer,
    )

    assert result.task_success is True
    assert result.tool_correct is True


@pytest.mark.asyncio
async def test_missing_security_review_requires_refusal():

    async def proposer(eval_input):
        return None

    result = await evaluate_case(
        ONBOARDING_EVAL_CASES[1],
        proposer,
    )

    assert result.task_success is True


@pytest.mark.asyncio
async def test_unsafe_proposal_fails():

    async def proposer(eval_input):
        return ProposedAction(
            tool_name="approve_partner",
            arguments={"partner_id": "partner-123"},
            requires_approval=False,
        )

    result = await evaluate_case(
        ONBOARDING_EVAL_CASES[1],
        proposer,
    )

    assert result.task_success is False
    assert result.tool_correct is False
