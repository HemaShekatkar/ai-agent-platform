
import pytest

from app.platform.tenant_admission import (
    TenantAdmissionController,
    TenantQuota,
)


@pytest.fixture
def controller():
    return TenantAdmissionController(
        global_limit=3,
        quotas=[
            TenantQuota("tenant-a", max_concurrent=2),
            TenantQuota("tenant-b", max_concurrent=2),
        ],
    )


def test_tenant_quota_enforced(controller):
    assert controller.acquire("tenant-a")
    assert controller.acquire("tenant-a")
    assert not controller.acquire("tenant-a")


def test_global_limit_enforced(controller):
    assert controller.acquire("tenant-a")
    assert controller.acquire("tenant-a")
    assert controller.acquire("tenant-b")
    assert not controller.acquire("tenant-b")


def test_capacity_released(controller):
    assert controller.acquire("tenant-a")
    controller.release("tenant-a")

    assert controller.acquire("tenant-a")


def test_unknown_tenant_rejected(controller):
    assert not controller.acquire("tenant-unknown")


def test_release_without_acquire_fails(controller):
    with pytest.raises(ValueError):
        controller.release("tenant-a")


def test_invalid_global_limit():
    with pytest.raises(ValueError):
        TenantAdmissionController(
            global_limit=0,
            quotas=[],
        )


def test_duplicate_tenant_rejected():
    with pytest.raises(ValueError):
        TenantAdmissionController(
            global_limit=3,
            quotas=[
                TenantQuota("tenant-a", 1),
                TenantQuota("tenant-a", 2),
            ],
        )
