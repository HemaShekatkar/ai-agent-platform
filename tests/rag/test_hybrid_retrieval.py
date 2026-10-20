from dataclasses import replace
from datetime import date

from app.rag.hybrid_retrieval import (
    Document,
    RetrievalContext,
    reciprocal_rank_fusion,
)


CTX = RetrievalContext(
    tenant_id="tenant-a",
    principal_id="analyst-1",
    as_of=date(2026, 10, 1),
)


def make_doc(doc_id: str, **overrides) -> Document:
    # Eligible for CTX unless a test overrides a field
    values = {
        "id": doc_id,
        "tenant_id": "tenant-a",
        "content": f"Onboarding policy {doc_id}",
        "policy_status": "ACTIVE",
        "effective_date": date(2026, 1, 1),
        "expires_at": None,
        "allowed_principals": frozenset({"analyst-1"}),
    }
    values.update(overrides)
    return Document(**values)


def ids(documents: list[Document]) -> list[str]:
    return [doc.id for doc in documents]


def test_superseded_policy_is_excluded_even_when_ranked_first():
    superseded = make_doc("policy-v1", policy_status="SUPERSEDED")
    current = make_doc("policy-v2")

    # The old version is the top hit in both retrievers
    result = reciprocal_rank_fusion(
        bm25_results=[superseded, current],
        dense_results=[superseded, current],
        ctx=CTX,
    )

    assert ids(result) == ["policy-v2"]


def test_document_from_another_tenant_is_excluded():
    own = make_doc("own-policy")
    foreign = make_doc("foreign-policy", tenant_id="tenant-b")

    result = reciprocal_rank_fusion(
        bm25_results=[foreign, own],
        dense_results=[foreign, own],
        ctx=CTX,
    )

    assert ids(result) == ["own-policy"]


def test_unauthorized_principal_cannot_retrieve_document():
    restricted = make_doc(
        "restricted-policy",
        allowed_principals=frozenset({"security-lead"}),
    )
    shared = make_doc("shared-policy")

    result = reciprocal_rank_fusion(
        bm25_results=[restricted, shared],
        dense_results=[restricted, shared],
        ctx=CTX,
    )

    assert ids(result) == ["shared-policy"]

    # The principal it was shared with can retrieve it
    authorized = replace(CTX, principal_id="security-lead")
    result = reciprocal_rank_fusion(
        bm25_results=[replace(restricted)],
        dense_results=[],
        ctx=authorized,
    )
    assert ids(result) == ["restricted-policy"]


def test_document_in_both_result_lists_appears_once():
    shared = make_doc("doc-shared")
    bm25_only = make_doc("doc-bm25")
    dense_only = make_doc("doc-dense")

    result = reciprocal_rank_fusion(
        bm25_results=[shared, bm25_only],
        dense_results=[dense_only, shared],
        ctx=CTX,
    )

    assert ids(result).count("doc-shared") == 1
    assert sorted(ids(result)) == ["doc-bm25", "doc-dense", "doc-shared"]


def test_duplicate_within_one_list_is_counted_once():
    # A retriever returning the same chunk twice must not double its score
    repeated = make_doc("doc-repeated")
    other = make_doc("doc-other")

    result = reciprocal_rank_fusion(
        bm25_results=[repeated, repeated, repeated],
        dense_results=[other],
        ctx=CTX,
    )

    assert ids(result).count("doc-repeated") == 1

    # Rank 1 in one list each, so the IDs tie and break alphabetically
    assert ids(result) == ["doc-other", "doc-repeated"]


def test_document_ranked_in_both_lists_outranks_single_list_leader():
    consensus = make_doc("doc-consensus")
    bm25_leader = make_doc("doc-bm25-leader")
    dense_leader = make_doc("doc-dense-leader")

    # consensus is 2nd in both lists; each leader is 1st in only one list.
    # RRF: 2 * 1/(60+2) ~= 0.0323 beats 1/(60+1) ~= 0.0164
    result = reciprocal_rank_fusion(
        bm25_results=[bm25_leader, consensus],
        dense_results=[dense_leader, consensus],
        ctx=CTX,
    )

    assert ids(result)[0] == "doc-consensus"
