
from app.rag.context_compression import (
    EvidenceClause,
    assemble_context,
)


def clause(
    clause_id,
    tokens,
    conditions,
    authoritative=True,
):
    return EvidenceClause(
        clause_id=clause_id,
        policy_id="SEC-2026-018",
        text=f"Exact policy text for {clause_id}",
        token_count=tokens,
        covers_conditions=frozenset(conditions),
        authoritative=authoritative,
    )


def test_preserves_all_mandatory_conditions():
    clauses = [
        clause("general_rule", 150, ["approval_sequence"]),
        clause(
            "eu_exception",
            350,
            ["regional_exception", "officer_approval"],
        ),
        clause("audit_rule", 150, ["approval_recorded"]),
    ]

    result = assemble_context(
        clauses,
        required_conditions=frozenset({
            "approval_sequence",
            "regional_exception",
            "officer_approval",
            "approval_recorded",
        }),
        token_budget=700,
    )

    assert result.complete
    assert result.missing_conditions == frozenset()
    assert result.tokens_used == 650


def test_fails_when_mandatory_evidence_exceeds_budget():
    clauses = [
        clause("required_policy", 650, ["all_conditions"])
    ]

    result = assemble_context(
        clauses,
        required_conditions=frozenset({"all_conditions"}),
        token_budget=500,
    )

    assert not result.complete
    assert "all_conditions" in result.missing_conditions


def test_rejects_non_authoritative_evidence():
    clauses = [
        clause(
            "outdated_policy",
            100,
            ["security_approval"],
            authoritative=False,
        )
    ]

    result = assemble_context(
        clauses,
        required_conditions=frozenset({"security_approval"}),
        token_budget=500,
    )

    assert not result.complete
    assert result.selected_clauses == ()


def test_selects_compact_complete_evidence():
    clauses = [
        clause("large_section", 900, ["A", "B"]),
        clause("small_a", 150, ["A"]),
        clause("small_b", 200, ["B"]),
    ]

    result = assemble_context(
        clauses,
        required_conditions=frozenset({"A", "B"}),
        token_budget=500,
    )

    assert result.complete
    assert result.tokens_used == 350
    assert {
        c.clause_id for c in result.selected_clauses
    } == {"small_a", "small_b"}
