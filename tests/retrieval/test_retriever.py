from datetime import date

from app.retrieval.models import DocumentChunk, RetrievalRequest
from app.retrieval.retriever import is_eligible, InMemoryRetriever
from app.routing.models import DataClassification


def make_chunk(**overrides):
    values = {
        "id": "chunk-1",
        "document_id": "security-policy",
        "text": "PII partners require an approved security review.",
        "document_type": "SECURITY",
        "region": "US",
        "effective_date": date(2026, 1, 1),
        "expiration_date": None,
        "access_groups": {"partner-security"},
        "data_classification": DataClassification.INTERNAL,
    }

    values.update(overrides)

    return DocumentChunk(**values)


def make_request(**overrides):
    values = {
        "query": "What security review is required?",
        "region": "US",
        "document_type": "SECURITY",
        "caller_groups": {"partner-security"},
        "as_of_date": date(2026, 10, 1),
    }

    values.update(overrides)

    return RetrievalRequest(**values)


def test_eligible_chunk():
    assert is_eligible(
        make_chunk(),
        make_request(),
    )


def test_unauthorized_caller_is_not_eligible():
    assert not is_eligible(
        make_chunk(),
        make_request(caller_groups={"partner-sales"}),
    )


def test_chunk_without_acl_groups_is_not_eligible():
    assert not is_eligible(
        make_chunk(access_groups=set()),
        make_request(),
    )


def test_wrong_region_is_not_eligible():
    assert not is_eligible(
        make_chunk(region="EU"),
        make_request(),
    )


def test_wrong_document_type_is_not_eligible():
    assert not is_eligible(
        make_chunk(document_type="FINANCE"),
        make_request(),
    )


def test_policy_not_yet_effective_is_not_eligible():
    assert not is_eligible(
        make_chunk(effective_date=date(2026, 10, 2)),
        make_request(),
    )


def test_expired_policy_is_not_eligible():
    assert not is_eligible(
        make_chunk(expiration_date=date(2026, 9, 30)),
        make_request(),
    )


def test_policy_expiring_on_as_of_date_is_not_eligible():
    assert not is_eligible(
        make_chunk(expiration_date=date(2026, 10, 1)),
        make_request(),
    )


def test_policy_effective_on_as_of_date_is_eligible():
    assert is_eligible(
        make_chunk(effective_date=date(2026, 10, 1)),
        make_request(),
    )


def test_retriever_filters_before_ranking():
    chunks = [
        make_chunk(
            id="authorized",
            text="Security review is required before onboarding.",
        ),
        make_chunk(
            id="unauthorized",
            text=(
                "Security security security review review "
                "required required onboarding."
            ),
            access_groups={"internal-security"},
        ),
    ]

    retriever = InMemoryRetriever(chunks)

    response = retriever.retrieve(
        make_request(
            query="security review required onboarding",
        )
    )

    results = response.results

    assert len(results) == 1
    assert results[0].chunk.id == "authorized"

    # The unauthorized chunk is filtered out before scoring, so it never
    # competes for a ranked slot despite its higher keyword overlap
    assert response.diagnostics.corpus_count == 2
    assert response.diagnostics.eligible_count == 1


def test_retriever_ranks_by_relevance():
    chunks = [
        make_chunk(
            id="less-relevant",
            text="Security documentation for partners.",
        ),
        make_chunk(
            id="more-relevant",
            text=(
                "Security review is required "
                "before onboarding."
            ),
        ),
    ]

    retriever = InMemoryRetriever(chunks)

    results = retriever.retrieve(
        make_request(
            query="security review required onboarding",
        )
    ).results

    assert results[0].chunk.id == "more-relevant"
    assert results[0].score > results[1].score


def test_retriever_respects_top_k():
    chunks = [
        make_chunk(
            id=f"chunk-{i}",
            text=f"Security review required {i}",
        )
        for i in range(10)
    ]

    retriever = InMemoryRetriever(chunks)

    response = retriever.retrieve(
        make_request(
            query="security review required",
            top_k=3,
        )
    )

    assert len(response.results) == 3

    # top_k truncates the ranked list, not the eligible set
    assert response.diagnostics.corpus_count == 10
    assert response.diagnostics.eligible_count == 10


def test_retriever_reports_diagnostics_when_nothing_is_eligible():
    retriever = InMemoryRetriever([
        make_chunk(id="eu-only", region="EU"),
        make_chunk(id="expired", expiration_date=date(2026, 6, 1)),
    ])

    response = retriever.retrieve(make_request())

    assert response.results == []
    assert response.diagnostics.corpus_count == 2
    assert response.diagnostics.eligible_count == 0


def test_retriever_reports_provenance_in_diagnostics():
    retriever = InMemoryRetriever(
        [make_chunk()],
        retrieval_version="lexical-v2",
        index_version="policy-index-17",
        embedding_model_version="none",
    )

    diagnostics = retriever.retrieve(make_request()).diagnostics

    assert diagnostics.retrieval_version == "lexical-v2"
    assert diagnostics.index_version == "policy-index-17"
    assert diagnostics.embedding_model_version == "none"
