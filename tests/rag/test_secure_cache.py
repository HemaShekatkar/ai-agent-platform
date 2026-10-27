
from datetime import date

from app.rag.hybrid_retrieval import Document
from app.rag.secure_cache import (
    CacheKey,
    SecureRetrievalCache,
)


def test_revoked_permission_blocks_cached_document():
    cache = SecureRetrievalCache()

    key = CacheKey(
        tenant_id="tenant-a",
        query_fingerprint="security-approval",
        retrieval_config_version="v1",
    )

    document = Document(
        id="SEC-2026-017",
        tenant_id="tenant-a",
        content="Security approval is required.",
        policy_status="ACTIVE",
        effective_date=date(2026, 1, 1),
    )

    cache.put(key, [document])

    permissions = {"engineer-1": True}

    def authorize(principal_id, doc):
        return permissions.get(principal_id, False)

    assert cache.get_authorized(
        key, "engineer-1", authorize
    ) == [document]

    # Simulate permission revocation.
    permissions["engineer-1"] = False

    assert cache.get_authorized(
        key, "engineer-1", authorize
    ) == []


def test_stale_policy_is_rejected_even_without_invalidation():
    cache = SecureRetrievalCache()

    key = CacheKey(
        tenant_id="tenant-a",
        query_fingerprint="security-approval",
        retrieval_config_version="v1",
    )

    old_policy = Document(
        id="SEC-2026-017",
        tenant_id="tenant-a",
        content="Old approval sequence",
        policy_status="ACTIVE",
        effective_date=date(2026, 1, 1),
    )

    cache.put(key, [old_policy])

    # Simulate the authoritative registry moving to v018.
    # The cache never received an invalidation event.
    active_policy_id = "SEC-2026-018"

    result = cache.get_current_authorized(
        key=key,
        principal_id="engineer-1",
        authorize=lambda principal, doc: True,
        is_current_policy=lambda doc: (
            doc.id == active_policy_id
        ),
    )

    assert result is None

    # The stale entry was also removed.
    assert cache.get_current_authorized(
        key=key,
        principal_id="engineer-1",
        authorize=lambda principal, doc: True,
        is_current_policy=lambda doc: True,
    ) is None
