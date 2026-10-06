from typing import Literal

from pydantic import BaseModel, Field


class OnboardingDecision(BaseModel):
    partner: str = Field(description="Name of the partner being evaluated")
    eligible: bool = Field(description="Whether the partner meets all onboarding criteria")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    reason: str = Field(description="Short explanation of the decision")
