"""Demo: evaluate a synthetic partner with a live model.

Requires OPENAI_API_KEY (see .env.example). The test suite does not run this.
"""

import os

from app.agents.recovery import resume_agent
from app.agents.onboarding_graph import build_onboarding_graph
from app.agents.checkpoint import create_checkpointer
from dotenv import load_dotenv

from app.models.openai_provider import OpenAIProvider
from app.models.schemas import OnboardingDecision


def main() -> None:
    # Load credentials only when the demo is run, never on import
    load_dotenv()

    provider = OpenAIProvider(model="gpt-5.6-luna")

    response = provider.generate_structured(
        system_prompt="""
        You evaluate partners for onboarding.
        A partner is eligible when:
        - security review has passed
        - contract has been signed
        - metadata validation has passed

        Assign LOW, MEDIUM, or HIGH risk.
        """,
        # Fictional partner; replace with your own synthetic examples
        prompt="""
        Partner: Example Streaming Co.
        Security review: passed
        Contract: NOT signed
        Metadata validation: passed
        """,
        response_model=OnboardingDecision,
    )

    decision = response.content

    print("\n--- Decision ---")
    print(f"Partner:  {decision.partner}")
    print(f"Eligible: {decision.eligible}")
    print(f"Risk:     {decision.risk_level}")
    print(f"Reason:   {decision.reason}")

    print("\n--- Telemetry ---")
    print(f"Model:         {response.model}")
    print(f"Input tokens:  {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Latency:       {response.latency_ms:.0f} ms")


if __name__ == "__main__":
    main()


async def run_agent(runner, registry):
    # Never hard-code credentials: set DATABASE_URL (see .env.example)
    db_uri = os.environ.get(
        "DATABASE_URL",
        "postgresql://localhost:5432/agent_platform",
    )

    async with create_checkpointer(db_uri) as checkpointer:
        graph = build_onboarding_graph(
            workflow_runner=runner,
            registry=registry,
            checkpointer=checkpointer,
        )

        result = await resume_agent(
            agent_run_id="agent-run-123",
            graph=graph,
            workflow_runner=runner,
        )

        return result
