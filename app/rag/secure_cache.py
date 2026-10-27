
from dataclasses import dataclass
from typing import Callable

from app.rag.hybrid_retrieval import Document


@dataclass(frozen=True)
class CacheKey:
    tenant_id: str
    query_fingerprint: str
    retrieval_config_version: str


class SecureRetrievalCache:
    def __init__(self):
        self._cache: dict[CacheKey, list[Document]] = {}

    def put(
        self,
        key: CacheKey,
        documents: list[Document],
    ) -> None:
        if any(doc.tenant_id != key.tenant_id for doc in documents):
            raise ValueError("Cross-tenant cache entry")

        self._cache[key] = list(documents)

    def get_authorized(
        self,
        key: CacheKey,
        principal_id: str,
        authorize: Callable[[str, Document], bool],
    ) -> list[Document]:
        documents = self._cache.get(key, [])

        # Current authorization is evaluated on every read.
        return [
            doc for doc in documents
            if doc.tenant_id == key.tenant_id
            and authorize(principal_id, doc)
        ]

    def get_current_authorized(
        self,
        key: CacheKey,
        principal_id: str,
        authorize: Callable[[str, Document], bool],
        is_current_policy: Callable[[Document], bool],
    ) -> list[Document] | None:
        """
        None means the cache is missing or stale
        and authoritative retrieval is required.

        An empty list means the cache was checked,
        but no documents are authorized.
        """
        if key not in self._cache:
            return None

        documents = self._cache[key]

        # Check policy freshness before returning results.
        # The callback must consult trusted current state.
        try:
            if not all(is_current_policy(doc) for doc in documents):
                self._cache.pop(key, None)
                return None

            return [
                doc for doc in documents
                if doc.tenant_id == key.tenant_id
                and authorize(principal_id, doc)
            ]
        except Exception:
            # Do not expose cached evidence if validation fails.
            self._cache.pop(key, None)
            return None
