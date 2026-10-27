
from app.rag.budgeted_context import (
    EvidenceDocument,
    select_context,
)


def test_selects_complete_context_within_budget():
    docs = [
        EvidenceDocument(
            "global-security",
            frozenset({"US-security", "EU-security", "CA-security"}),
            0.96, 900, True,
        ),
        EvidenceDocument(
            "global-legal",
            frozenset({"US-legal", "EU-legal", "CA-legal"}),
            0.94, 850, True,
        ),
        EvidenceDocument(
            "us-residency",
            frozenset({"US-residency"}),
            0.93, 450, True,
        ),
        EvidenceDocument(
            "eu-residency",
            frozenset({"EU-residency"}),
            0.98, 550, True,
        ),
        EvidenceDocument(
            "ca-residency",
            frozenset({"CA-residency"}),
            0.95, 450, True,
        ),
    ]

    required = set().union(
        *(doc.requirements for doc in docs)
    )

    result = select_context(
        candidates=docs,
        required=required,
        token_budget=4000,
    )

    assert result.complete
    assert result.missing == frozenset()
    assert result.tokens_used == 3200


def test_non_authoritative_document_is_excluded():
    docs = [
        EvidenceDocument(
            "outdated-guide",
            frozenset({"security"}),
            0.99, 100, False,
        ),
        EvidenceDocument(
            "official-policy",
            frozenset({"security"}),
            0.90, 300, True,
        ),
    ]

    result = select_context(
        docs,
        required={"security"},
        token_budget=500,
    )

    assert [doc.id for doc in result.documents] == [
        "official-policy"
    ]


def test_insufficient_budget_reports_missing_evidence():
    docs = [
        EvidenceDocument(
            "policy-a",
            frozenset({"security"}),
            0.95, 500, True,
        ),
        EvidenceDocument(
            "policy-b",
            frozenset({"legal"}),
            0.95, 500, True,
        ),
    ]

    result = select_context(
        docs,
        required={"security", "legal"},
        token_budget=600,
    )

    assert result.complete is False
    assert len(result.missing) == 1
    assert result.tokens_used <= 600
