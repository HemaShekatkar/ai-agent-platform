
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    EXACT_POLICY = "exact_policy"
    GENERAL_FAQ = "general_faq"
    MULTI_DOCUMENT = "multi_document"
    HIGH_RISK = "high_risk"


@dataclass(frozen=True)
class RerankDecision:
    required: bool
    reason: str
    max_candidates: int


def decide_reranking(
    query_type: QueryType,
    evidence_ambiguous: bool,
    authoritative_policy_found: bool,
) -> RerankDecision:

    if query_type == QueryType.EXACT_POLICY:
        if authoritative_policy_found:
            return RerankDecision(
                required=False,
                reason="Authoritative policy resolved directly",
                max_candidates=0,
            )

        return RerankDecision(
            required=False,
            reason="Targeted policy recovery required",
            max_candidates=0,
        )

    if query_type == QueryType.HIGH_RISK:
        if not authoritative_policy_found:
            return RerankDecision(
                required=False,
                reason="Authoritative evidence required before proceeding",
                max_candidates=0,
            )

        return RerankDecision(
            required=True,
            reason="Rerank supporting evidence for high-risk query",
            max_candidates=20,
        )

    if query_type == QueryType.MULTI_DOCUMENT:
        return RerankDecision(
            required=True,
            reason="Multi-document evidence synthesis",
            max_candidates=20,
        )

    return RerankDecision(
        required=evidence_ambiguous,
        reason=(
            "Ambiguous evidence requires reranking"
            if evidence_ambiguous
            else "Initial evidence sufficient"
        ),
        max_candidates=20 if evidence_ambiguous else 0,
    )
