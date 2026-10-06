from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.retrieval.models import (
    DocumentChunk,
    RetrievalRequest,
    RetrievalResult,
)


@dataclass
class RetrievalDiagnostics:
    corpus_count: int
    eligible_count: int
    retrieval_version: str
    index_version: str
    embedding_model_version: str


@dataclass
class RetrievalResponse:
    results: list[RetrievalResult]
    diagnostics: RetrievalDiagnostics


def is_eligible(
    chunk: DocumentChunk,
    request: RetrievalRequest,
) -> bool:
    # ACL: caller must share at least one group; chunks with no groups are denied
    if not chunk.access_groups & request.caller_groups:
        return False

    if chunk.region != request.region:
        return False

    if chunk.document_type != request.document_type:
        return False

    if chunk.effective_date > request.as_of_date:
        return False

    # Expiration date is exclusive: a chunk is no longer valid on that day
    if (
        chunk.expiration_date is not None
        and chunk.expiration_date <= request.as_of_date
    ):
        return False

    return True


class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        ...


class InMemoryRetriever(Retriever):

    def __init__(
        self,
        chunks: list[DocumentChunk],
        retrieval_version: str = "lexical-v1",
        index_version: str = "in-memory-v1",
        embedding_model_version: str = "none",
    ):
        self.chunks = chunks
        self.retrieval_version = retrieval_version
        self.index_version = index_version
        self.embedding_model_version = embedding_model_version

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:

        eligible = [
            chunk
            for chunk in self.chunks
            if is_eligible(chunk, request)
        ]

        scored = [
            RetrievalResult(
                chunk=chunk,
                score=relevance_score(request.query, chunk.text),
            )
            for chunk in eligible
        ]

        scored.sort(key=lambda result: result.score, reverse=True)

        results = scored[:request.top_k]

        return RetrievalResponse(
            results=results,
            diagnostics=RetrievalDiagnostics(
                corpus_count=len(self.chunks),
                eligible_count=len(eligible),
                retrieval_version=self.retrieval_version,
                index_version=self.index_version,
                embedding_model_version=self.embedding_model_version,
            ),
        )


def relevance_score(
    query: str,
    text: str,
) -> float:
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())

    if not query_terms:
        return 0.0

    overlap = query_terms & text_terms

    return len(overlap) / len(query_terms)
