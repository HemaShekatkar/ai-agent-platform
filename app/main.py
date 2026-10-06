"""Demo: evaluate a synthetic partner with a live model.

Requires OPENAI_API_KEY (see .env.example). The test suite does not run this.
"""

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
