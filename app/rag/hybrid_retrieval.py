
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Document:
    id: str
    tenant_id: str
    content: str
    policy_status: str
    effective_date: date
    expires_at: date | None = None
    allowed_principals: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RetrievalContext:
    tenant_id: str
    principal_id: str
    as_of: date


def is_eligible(doc: Document, ctx: RetrievalContext) -> bool:
    return (
        doc.tenant_id == ctx.tenant_id
        and ctx.principal_id in doc.allowed_principals
        and doc.policy_status == "ACTIVE"
        and doc.effective_date <= ctx.as_of
        and (
            doc.expires_at is None
            or ctx.as_of < doc.expires_at
        )
    )


def reciprocal_rank_fusion(
    bm25_results: list[Document],
    dense_results: list[Document],
    ctx: RetrievalContext,
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    if k <= 0 or top_n < 0:
        raise ValueError("Invalid ranking parameters")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for results in (bm25_results, dense_results):
        seen: set[str] = set()

        for rank, doc in enumerate(results, start=1):
            if doc.id in seen:
                continue
            seen.add(doc.id)

            if not is_eligible(doc, ctx):
                continue

            # A document ID must refer to consistent metadata.
            if doc.id in documents and documents[doc.id] != doc:
                raise ValueError("Conflicting document metadata")

            documents[doc.id] = doc
            scores[doc.id] = (
                scores.get(doc.id, 0.0)
                + 1.0 / (k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )

    return [documents[doc_id] for doc_id in ranked_ids[:top_n]]
