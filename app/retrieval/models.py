from dataclasses import dataclass, field
from datetime import date
from typing import Any
from app.routing.models import DataClassification


@dataclass
class DocumentChunk:
    id: str
    document_id: str
    text: str

    document_type: str
    region: str

    effective_date: date
    expiration_date: date | None

    access_groups: set[str]
    data_classification: DataClassification

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRequest:
    query: str

    region: str
    document_type: str

    caller_groups: set[str]

    as_of_date: date

    top_k: int = 5


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
